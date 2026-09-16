#!/usr/bin/env python3
"""Harness integrity audit (Exp660).

Motivation: Exp659 lost two full days of hidden risk to two harness defects that both
produced *numbers* instead of errors - (a) run_rtf_multi.sh had been silently deleted by an
auto-revert because it was untracked, so past sweeps were unreproducible, and (b) a restored
copy parsed DEV from config.json, which never contained it, so every adb call degraded into
a malformed command and each arm reported an empty result rather than a failure. A third
variant showed up the same day: an invalid (decimal) affinity mask made taskset fail, and
the arm again looked like a null result.

This script makes that class self-reporting. It checks, in order of how badly a failure
would corrupt results:
  1. syntax of every harness script (a broken script that never runs is better than one
     that half-runs),
  2. that every path a script references on the host exists,
  3. that the device holds the artifacts the scripts expect, byte-identical for small
     files and size-checked for the multi-GB models,
  4. that config values (DEV) are parsed from a source that actually contains them,
  5. that the frozen transcript references the A/B harness diffs against are present.

Exit code is non-zero if any check fails, so it can gate a session.

Usage: .auto/audit_harness.py [--skip-device]
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
RDIR = '/data/local/tmp/vibeasr'
fails, warns, oks = [], [], []


def ok(msg):
    oks.append(msg)


def bad(msg):
    fails.append(msg)


def warn(msg):
    warns.append(msg)


def sh(cmd, timeout=120):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


scripts = sorted(f for f in os.listdir(HERE) if f.endswith(('.sh', '.py'))
                 and not f.startswith('audit_harness'))

# ---- 1. syntax -----------------------------------------------------------------
for s in scripts:
    p = os.path.join(HERE, s)
    if s.endswith('.sh'):
        r = sh(f'bash -n "{p}"')
    else:
        r = sh(f'"{sys.executable}" -c "import py_compile,sys; py_compile.compile(sys.argv[1], doraise=True)" "{p}"')
    (bad if r.returncode else ok)(f"syntax {s}: " + (r.stderr.strip().splitlines()[-1][:90] if r.returncode else 'clean'))

# ---- 2. host paths referenced by scripts ---------------------------------------
PATH_RE = re.compile(r'(?<![\w/.])((?:\./)?[\w\.\-\$\{\}]+/[\w\.\-\$\{\}]+\.(?:sh|py|json|gguf|wav|txt|md|log|tsv))')
SKIP_PREFIX = ('/data', '/proc', '/sys', '/system', '/tmp', '$RDIR', '${RDIR}', '.auto/multi-', '.auto/last_')
seen = set()
for s in scripts:
    txt = open(os.path.join(HERE, s), errors='ignore').read()
    for m in PATH_RE.finditer(txt):
        rel = m.group(1)
        rel = rel[2:] if rel.startswith('./') else rel   # NB: lstrip('./') would eat '.auto' -> 'auto'
        if rel.startswith(SKIP_PREFIX) or '$' in rel or '{' in rel:
            continue
        key = (s, rel)
        if key in seen:
            continue
        seen.add(key)
        cands = [os.path.join(ROOT, rel), os.path.join(ROOT, '..', rel),
                 os.path.join(ROOT, '..', 'eval-bilingual', rel),
                 os.path.join(ROOT, 'eval-bilingual', rel)]
        if not any(os.path.exists(c) for c in cands):
            bad(f"missing path referenced by {s}: {rel}")
if not any('missing path' in f for f in fails):
    ok(f"all {len(seen)} host paths referenced by harness scripts exist")

# ---- 3. git tracking (an untracked harness file can be deleted by an auto-revert)
r = sh(f'cd "{ROOT}" && git ls-files .auto')
tracked = set(r.stdout.split())
for s in scripts:
    if f'.auto/{s}' not in tracked:
        bad(f"harness script is NOT tracked in git (an auto-revert will delete it): {s}")
if not any('NOT tracked' in f for f in fails):
    ok(f"all {len(scripts)} harness scripts are tracked in git")

# ---- 3b. tree cleanliness between runs (Exp759).
# pi-autoresearch reverts a discard with `git checkout -- .` executed at the SESSION
# workDir (extensions/pi-autoresearch/index.ts:2446, cwd=workDir). Here the session
# workDir is the PARENT of this repo and is not a git repository at all - which is why
# every log_experiment prints "git add failed (exit 128): not a git repository". The
# commit failure is loud, but the REVERT is not: pi.exec's exit code is never checked
# there, so a discard prints "Git: reverted changes" while nothing was reverted, and the
# discarded code would survive into every later measurement. Dirty tracked source between
# runs therefore means that bug has fired.
WD = os.path.dirname(ROOT)
inside_repo = sh(f'cd "{WD}" && git rev-parse --show-toplevel 2>/dev/null').stdout.strip()
r = sh(f'cd "{ROOT}" && git status --porcelain')
dirty = [ln[3:] for ln in r.stdout.splitlines()
         if ln.strip() and not ln[3:].lstrip('"').startswith('.auto/')]
if dirty:
    bad(f"tracked files are DIRTY between runs: {dirty[:6]} - auto-revert cannot protect "
        f"you (session workDir {WD} is "
        f"{'not a git repo' if not inside_repo else 'not the repo root'}); revert or "
        f"commit in {ROOT} before trusting any measurement")
elif not inside_repo:
    ok(f"tree clean; NOTE auto-commit/auto-revert are inert here (workDir {WD} is not a "
       f"git repo) - commit discards manually in {ROOT}")
else:
    ok("tree clean between runs")

# ---- 4. config-source consistency ---------------------------------------------
measure = open(os.path.join(HERE, 'measure.sh'), errors='ignore').read()
dev_m = re.search(r'^DEV=(\S+)', measure, re.M)
if not dev_m:
    bad("measure.sh no longer defines DEV - device id source of truth moved")
else:
    DEV = dev_m.group(1)
    ok(f"DEV={DEV} parsed from measure.sh (the single source)")
    for s in scripts:
        if s.endswith('.sh'):
            txt = open(os.path.join(HERE, s), errors='ignore').read()
            if 'DEV' in txt and 'config.json' in txt and not re.search(r'^DEV=', txt, re.M):
                warn(f"{s} mentions DEV and config.json - verify DEV is not parsed from a file that lacks it")
    if '--skip-device' not in sys.argv:
        r = sh(f'adb -s {DEV} get-state 2>&1', timeout=30)
        (ok if 'device' in r.stdout else bad)(f"device {DEV} state: {r.stdout.strip()[:40]}")

        # ---- 4b. shipped-tier spec vs every runner's defaults -------------------------
# Exp694: eval40.sh defaulted to VAE=vae-encoder-q8_0mixed, LM=streaming-lm-q4_k_m, PIECES=13 while the
# shipped tier was something else, so a gate with no explicit env measured a DIFFERENT SYSTEM and printed
# a plausible WER/RTF for it. One declaration, checked against every consumer.
TIER = {}
tp = os.path.join(HERE, 'tier.env')
if not os.path.exists(tp):
    bad("tier.env is missing - it must declare the shipped tier's VAE_FILE/LM_FILE/PIECES")
else:
    for line in open(tp, errors='ignore'):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            TIER[k.strip()] = v.strip()
    for k in ('VAE_FILE', 'LM_FILE', 'PIECES'):
        if k not in TIER:
            bad(f"tier.env does not declare {k}")
    for s_ in ('measure.sh', 'eval40.sh'):
        txt = open(os.path.join(HERE, s_), errors='ignore').read()
        for k, want in TIER.items():
            vals = re.findall(r'\{' + k + r':-([^}]*)\}', txt)
            if not vals:
                bad(f"{s_}: no default for {k} found - cannot verify it matches the shipped tier")
            elif want not in vals:
                bad(f"{s_}: {k} defaults to {sorted(set(vals))} but tier.env says '{want}' - "
                    f"a run without explicit env measures a DIFFERENT SYSTEM")
            elif len(set(vals)) > 1:
                warn(f"{s_}: {k} has several distinct defaults in the file {sorted(set(vals))} - "
                     f"make them consistent")
        if TIER.get('VAE_FILE'):
            ok(f"{s_} defaults match tier.env for {sorted(TIER)}")
    pm = os.path.join(HERE, 'prompt.md')
    if os.path.exists(pm) and TIER.get('VAE_FILE'):
        t = open(pm, errors='ignore').read()
        if TIER['VAE_FILE'] not in t:
            bad(f"prompt.md never names the shipped VAE file ({TIER['VAE_FILE']}) - the tier table is "
                f"stale prose that future sessions will follow")
        else:
            ok("prompt.md's tier table names the shipped VAE file")

# ---- 5. device inventory ----------------------------------------------
        # derive the pushed-artifact list from measure.sh (the harness's own definition),
        # so the audit cannot drift from what actually gets pushed
        push = []
        for m in re.finditer(r'(?:for \w+ in |adb -s \$DEV push )([^\n]*?)(?: do|\$RDIR)', measure):
            push += [t.strip(';,') for t in m.group(1).split() if '/' in t and '$' not in t]
        push += [t for t in re.findall(r'[\w/\.\-]+\.so', measure)]   # measure.sh line 36 style
        push += ['build-android/bin/asr_streaming', '.auto/bench_device.sh']
        if not any(t.endswith('.so') for t in push):
            bad("could not derive the pushed-library list from measure.sh - 'device in sync' "
                "would be an over-claim, so it is reported as a failure instead")
        small = []
        for rel in dict.fromkeys(push):
            lp = next((os.path.join(ROOT, pre, rel) for pre in ('', 'build-android/')
                       if os.path.exists(os.path.join(ROOT, pre, rel))), None)
            if lp is None:
                continue                      # not built in this tree - nothing to compare
            small.append((os.path.basename(rel), lp))
        for f, lp in small:
            d = sh(f'adb -s {DEV} shell md5sum {RDIR}/{f} 2>/dev/null', timeout=60).stdout
            dm = re.search(r'([0-9a-f]{32})', d)
            if not dm:
                bad(f"{f} missing on device")
                continue
            (ok if dm.group(1) == md5(lp) else bad)(
                f"{f}: host vs device md5 {'match' if dm.group(1) == md5(lp) else 'MISMATCH (' + dm.group(1) + ' != ' + md5(lp) + ')'}")
        for f in ['vae-encoder-q4x4ffn.gguf', 'lm-q8head.gguf', 'stream_10s_24k.wav']:
            d = sh(f'adb -s {DEV} shell "ls -l {RDIR}/{f} 2>/dev/null"', timeout=60).stdout
            size = re.search(r'\s(\d+)\s', d.split('\n')[-1]) if d.strip() else None
            host = os.path.join(ROOT, f)
            if not d.strip():
                bad(f"{f} missing on device")
            elif os.path.exists(host) and size and int(size.group(1)) != os.path.getsize(host):
                bad(f"{f}: device size {size.group(1)} != host {os.path.getsize(host)}")
            else:
                ok(f"{f} present on device" + ('' if os.path.exists(host) else ' (no host copy)'))

# ---- 7. device audio-asset integrity (Exp675) ----------------------------------
# Two clips backing documented cells turned out to be BYTE-IDENTICAL: chat.wav had been
# overwritten by the chat69.wav push, so the 17 s ladder cell was measuring a file that no
# longer existed. A device-side file with no recorded identity can be silently replaced by a
# later experiment and still print a plausible RTF, so every clip that backs a published number
# is hashed, and - the part that actually caught this one - collision-checked against the others.
# Manifest: .auto/device_assets.json (regenerate deliberately with --bless).
MANIFEST = os.environ.get('AUDIT_ASSET_MANIFEST') or os.path.join(HERE, 'device_assets.json')
if '--skip-device' not in sys.argv:
    if '--bless' in sys.argv:
        names = sorted(os.path.basename(f) for f in
                       sh(f'adb -s {DEV} shell "ls {RDIR}/*.wav"', timeout=60).stdout
                       .replace('\r', '').split() if f)
        h = sh('adb -s ' + DEV + ' shell "md5sum ' + ' '.join(f'{RDIR}/{n}' for n in names) + '"',
               timeout=300).stdout
        assets = {}
        for line in h.splitlines():
            m = re.match(r'([0-9a-f]{32})\s+\S*/(\S+)$', line.strip())
            if m:
                assets[m.group(2)] = {'md5': m.group(1), 'note': 'blessed from device'}
        with open(MANIFEST, 'w') as f:
            json.dump({'assets': assets}, f, indent=1, sort_keys=True)
        print(f"blessed {len(assets)} device clips into {MANIFEST}")
        sys.exit(0)
    if os.path.exists(MANIFEST):
        man = json.load(open(MANIFEST))['assets']
        # Keys prefixed 'HOST ' point at host-tracked copies (behavioral-watchdog clips that used to live
        # only on the device, where any later experiment could overwrite them - Exp800). Everything else is
        # a device-side name. Feed each class to its own hasher; a HOST key must never reach adb.
        host = {k: man.pop(k) for k in [x for x in man if x.startswith('HOST ')]}
        for k, v in host.items():
            rel = k[len('HOST '):]
            if not os.path.exists(rel):
                (bad if v.get('cell') else warn)(f"host asset {rel} is missing")
            elif hashlib.md5(open(rel, 'rb').read()).hexdigest() != v.get('md5'):
                bad(f"host asset {rel} md5 != blessed {v.get('md5')}"
                    + (f" (cell: {v['cell']})" if v.get('cell') else ''))
            else:
                ok(f"host asset {os.path.basename(rel)} hash matches manifest")
        names = list(man)
        got = sh('adb -s ' + DEV + ' shell "md5sum ' + ' '.join(f'{RDIR}/{n}' for n in names)
                 + ' 2>/dev/null"', timeout=300).stdout
        have = {os.path.basename(l.split()[-1]): l.split()[0]
                for l in got.splitlines() if re.match(r'[0-9a-f]{32}\s+\S+', l)}
        dev = {n: have[n] for n in names if n in have}
        missing = [n for n in names if n not in have]
        for n in missing:
            (bad if man[n].get('cell') else warn)(
                f"device clip {n} missing" + (f" - backs a documented cell ({man[n]['cell']})"
                                              if man[n].get('cell') else ' (probe only)'))
        for n, d in dev.items():
            if man[n].get('md5') and d != man[n]['md5']:
                bad(f"{n}: device md5 {d} != blessed {man[n]['md5']}" +
                    (f" (cell: {man[n]['cell']})" if man[n].get('cell') else ''))
            else:
                ok(f"{n} hash matches manifest")
        # the check that catches Exp675: two documented clips with the same bytes = one was
        # overwritten by the other, and any 'different condition' comparison is now a no-op.
        byhash = {}
        for n, d in dev.items():
            byhash.setdefault(d, []).append(n)
        for d, ns in sorted(byhash.items()):
            if len(ns) > 1:
                both_documented = all(not man[n].get('ignore_collision') for n in ns)
                msg = (f"asset collision: {' == '.join(sorted(ns))} are byte-identical "
                       f"({d[:12]}...) - any A/B between them is measuring one file twice")
                (bad if both_documented else warn)(
                    msg + ('' if both_documented else '  [one side is marked stale/unused, so not a failure]'))
        # derivation check: a clip documented as an excerpt must still be that excerpt
        for n, spec in man.items():
            der = spec.get('derived_from')
            if der and n in dev and der['file'] in dev:
                tmp = tempfile.mkdtemp()
                for f in (n, der['file']):
                    sh(f'adb -s {DEV} pull {RDIR}/{f} {tmp}/{f} > /dev/null 2>&1', timeout=180)
                if os.path.exists(f"{tmp}/{der['file']}") and os.path.exists(f"{tmp}/{n}"):
                    # compare AUDIO DATA, not file bytes: the excerpt has its own 44-byte header,
                    # so hashing the files would never match the source prefix.
                    wo, ws = wave.open(f"{tmp}/{n}"), wave.open(f"{tmp}/{der['file']}")
                    par = (wo.getframerate(), wo.getnchannels(), wo.getsampwidth()) == \
                          (ws.getframerate(), ws.getnchannels(), ws.getsampwidth())
                    same = par and wo.getnframes() == der['frames'] and \
                        wo.readframes(der['frames']) == ws.readframes(der['frames'])
                    detail = ('confirmed' if same else
                              f"NO (frames {wo.getnframes()} vs {der['frames']}, "
                              f"params {'match' if par else 'DIFFER'}" + 
                              (', data differs' if par and wo.getnframes() == der['frames'] else ')'))
                    wo.close(); ws.close()
                    (ok if same else bad)(
                        f"{n} == first {der['frames']} frames of {der['file']}: " + detail)
                shutil.rmtree(tmp, ignore_errors=True)
    else:
        warn(f"no asset manifest at {MANIFEST} - device clips are unverified (run --bless once)")

# ---- 7b. no co-runner on the device (Exp679) -----------------------------------
# A host-side `timeout` kills the adb client, not the device-side process. If the survivor is
# still RUNNING (State R, model loaded) it shares the two pinned cores and every timing in the
# session is ~2x inflated - numbers, not errors. Sleeping sub-MB survivors are clutter, not bias.
if '--skip-device' not in sys.argv:
    # NB this toybox has no STATE keyword in `ps -o` (it is S, and asking for STATE makes ps print
    # an error and NO rows, so a check built on it passes vacuously - the Exp660 sin). RSS alone is
    # the criterion we actually need: a co-runner that matters has a model resident (>200 MB).
    THRESH = int(os.environ.get('AUDIT_CO_RUNNER_KB', 200_000))
    ps = sh(f'adb -s {DEV} shell "ps -A -o PID,RSS,NAME"', timeout=60).stdout.replace('\r', '')
    if not [l for l in ps.splitlines() if l.strip()]:
        bad("ps returned no rows - the co-runner check cannot be trusted (bad -o keywords?)")
        live = []
    else:
        live = [(l.split()[0], l.split()[1]) for l in ps.splitlines()
                if 'asr_streaming' in l and len(l.split()) >= 3
                and l.split()[1].isdigit() and int(l.split()[1]) > THRESH]
    if live:
        bad(f"{len(live)} asr_streaming with a model resident on device (pids "
            f"{[p for p, _ in live]}) - timings are not comparable until they are killed")
    else:
        ok("no co-runner: device is idle for timing")

# ---- 8. frozen references used by A/B tooling ----------------------------------
for f in sorted(os.listdir(HERE)):
    if f.startswith('ref') and f.endswith('.txt'):
        if os.path.getsize(os.path.join(HERE, f)) < 20:
            bad(f"frozen reference {f} is empty/truncated")
refs_dir = os.path.join(ROOT, '..', 'eval-librispeech')
print("frozen references present: "
      + ', '.join(f for f in sorted(os.listdir(HERE)) if f.startswith('ref') and f.endswith('.txt'))
      + f"; gate refs.json: {'found' if os.path.exists(os.path.join(refs_dir, 'refs.json')) else 'MISSING'}")

# ---- 8. ledger coherence (Exp770: the ledger is split in two halves and the loop prompt points at
# the STALE archive half. A session that reads only the prompt path resurrects closed work - it cost
# runs 769-770, which re-ran a converter experiment the live half had already closed.)
live = os.path.join(HERE, 'ideas.md')
archive = os.path.join(ROOT, '..', '.auto', 'ideas.md')
log = os.path.join(ROOT, '..', '.auto', 'log.jsonl')
if os.path.exists(log) and os.path.exists(live):
    runs = [int(m) for m in re.findall(r'"run":\s*(\d+)', open(log, encoding='utf-8', errors='replace').read())]
    newest = max(runs) if runs else 0
    cited = [int(n) for n in re.findall(r'Exp(\d{3})', open(live, encoding='utf-8', errors='replace').read())]
    gap = newest - (max(cited) if cited else 0)
    if gap > 12:
        bad(f"LIVE ledger is {gap} runs behind the log (newest run {newest}, newest citation "
            f"Exp{max(cited) if cited else 0}) - closed axes will look open")
    else:
        ok(f"live ledger current (newest run {newest}, newest citation Exp{max(cited)})")
    if os.path.exists(archive):
        if 'POINTER (Exp770' not in open(archive, encoding='utf-8', errors='replace').read():
            bad("archive ledger half lacks the POINTER header - a session reading the prompt path "
                "alone will not know the live half exists")
        else:
            ok("archive ledger half carries the cross-reference pointer")
    else:
        warn("archive ledger half not found at ../.auto/ideas.md (fine if it was merged)")
else:
    warn("ledger coherence not checked (missing live ledger or loop log)")

# ---- report -------------------------------------------------------------------
print(f"harness audit: {len(oks)} checks passed, {len(warns)} warnings, {len(fails)} failures\n")
if '--verbose' in sys.argv:
    for o in oks:
        print('  ok   ', o[:110])
for w in warns:
    print("  WARN ", w)
for f in fails:
    print("  FAIL ", f)
if not fails:
    print("  all green - harness is self-consistent, scripts tracked, device in sync")
sys.exit(1 if fails else 0)
