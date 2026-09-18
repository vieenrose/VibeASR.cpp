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
import glob
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
DEVICE_PATHS = {}
_RDIR = '/data/local/tmp/vibeasr'
SKIP_PREFIX = ('/data', '/proc', '/sys', '/system', '/tmp', '$RDIR', '${RDIR}', '.auto/multi-', '.auto/last_')
seen = set()
for s in scripts:
    txt = open(os.path.join(HERE, s), errors='ignore').read()
    # Scripts that drive the PHONE binary reference files that live in $RDIR on the device, not in the repo.
    # Declared with a `# DEVICE_PATHS: name1 name2` marker so (a) the host-path check below does not cry wolf
    # (Exp815 printed 3 FAILs for exactly that) and (b) section 3c can assert they exist where they are
    # actually used - a driver whose model file disappeared would otherwise silently run the wrong tier.
    dcl = re.search(r'^\s*#\s*DEVICE_PATHS:\s*(.+)$', txt, re.M)
    if dcl:
        DEVICE_PATHS.setdefault(s, set()).update(dcl.group(1).split())
    for m in PATH_RE.finditer(txt):
        rel = m.group(1)
        rel = rel[2:] if rel.startswith('./') else rel   # NB: lstrip('./') would eat '.auto' -> 'auto'
        if rel in DEVICE_PATHS.get(s, set()):
            continue        # declared as a DEVICE-side artifact (see the DEVICE_PATHS marker) - checked in 3c
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

# ---- 3c. declared DEVICE-side artifacts exist on the phone (Exp815)
_d = re.search(r'^DEV=(\S+)', open(os.path.join(HERE, 'measure.sh'), errors='ignore').read(), re.M)
DEV3 = _d.group(1) if _d else None
_dev_paths = sorted({p for v in DEVICE_PATHS.values() for p in v})
if _dev_paths and DEV3 and '--skip-device' not in sys.argv:
    listing = sh(f'adb -s {DEV3} shell "ls {_RDIR}"').stdout.split()
    missing = [p for p in _dev_paths if p not in listing]
    for p in missing:
        bad(f"declared DEVICE_PATHS entry is absent on the phone: {p}")
    if not missing:
        ok(f"all {len(_dev_paths)} declared device-side artifacts present on {DEV3}")

# ---- 3. git tracking (an untracked harness file can be deleted by an auto-revert)
r = sh(f'cd "{ROOT}" && git ls-files .auto')
tracked = set(r.stdout.split())
for s in scripts:
    if f'.auto/{s}' not in tracked:
        bad(f"harness script is NOT tracked in git (an auto-revert will delete it): {s}")
if not any('NOT tracked' in f for f in fails):
    ok(f"all {len(scripts)} harness scripts are tracked in git")

# ---- 3c. executable bit (Exp809). A harness script can be tracked and syntactically fine and still be
# unrunnable: run_rtf_multi.sh came back as mode 100644 after a rewrite, so the paired-sweep path printed
# 'Permission denied' and, because most call sites piped output away, looked like a silent no-op run.
for s in scripts:
    if not s.endswith('.sh'):
        continue          # .py files are always invoked as `python3 x.py`; the exec bit is irrelevant
    p = os.path.join(HERE, s)
    if not os.access(p, os.X_OK):
        bad(f"harness script is not executable (chmod +x and commit the mode): .auto/{s}")
if not any('not executable' in f for f in fails):
    ok("all shell harness scripts are executable")

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
        # Stitched eval STREAMS must declare their phase policy (Exp866h/i). The windowed protocol advances
        # 70,400 samples per window, so a clip that does not start on that boundary is processed at an
        # arbitrary phase of the window grid. Measured on this stack: that does NOT bias WER (en +1.36 pp,
        # zh -1.50 pp, opposite signs, both inside their CIs) but it churns 11-15% of tokens, which is label
        # noise for training data and makes two builds of "the same" asset incomparable. So the property must
        # be stated, not assumed: phase_policy = "hop-aligned" (verified here) or "arbitrary" (accepted).
        HOP = 70400
        for mf in sorted(glob.glob(os.path.join(os.path.dirname(RDIR) if False else '.',
                                                '..', 'eval-bilingual', 'manifest_*.json'))):
            try:
                mj = json.load(open(mf))
            except Exception:
                continue
            tab = mj.get('table')
            if not tab or len(tab) < 2:
                continue                                     # single-clip assets have phase 0 by construction
            sr = mj.get('sr', 24000)
            # Prefer exact sample positions; fall back to the ms-rounded start_s with a +-32-sample tolerance,
            # because that timestamp is +-12 samples quantised and would otherwise call an aligned asset
            # misaligned (which is exactly what the first version of this check did to Exp866i's twins).
            devs = [min(t.get('start_samples', round(t['start_s'] * sr)) % HOP,
                        HOP - t.get('start_samples', round(t['start_s'] * sr)) % HOP) for t in tab]
            ph = [t.get('start_samples', round(t['start_s'] * sr)) % HOP for t in tab]
            nph = len(set(ph))
            aligned = max(devs) <= 32
            pol = mj.get('phase_policy')
            tag = os.path.basename(mf)
            if pol is None:
                warn(f"{tag}: {len(tab)} clips at {nph} distinct phases of the {HOP}-sample window grid and NO "
                     'phase_policy - run .auto/align_asset.py or declare "phase_policy": "arbitrary"')
            elif pol == 'hop-aligned':
                if aligned:
                    ok(f'{tag}: hop-aligned, {len(tab)} clips start a window (max deviation {max(devs)} samples)')
                else:
                    bad(f'{tag}: declares phase_policy=hop-aligned but clips deviate up to {max(devs)} samples '
                        'from the grid')
            elif pol == 'arbitrary':
                ok(f'{tag}: phases arbitrary ({nph} distinct over {len(tab)} clips) - declared, tokens churn '
                   'but WER is unbiased on this stack')
            else:
                bad(f'{tag}: unknown phase_policy {pol!r} (use "hop-aligned" or "arbitrary")')
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

# ---- 9. headline coherence (Exp857). A doc-drift class the OTHER 87 checks were blind to:
# the measured LADDER cells get refreshed on the doc-drift trigger, but the PROSE that a fresh
# session reads first (RESULTS.md's "Headline:", prompt.md's "Current best:") is prose, so nothing
# ties it to a measurement. It silently rotted two full eras: RESULTS.md still claimed
# "12.24 -> 2.70 (-78 %)" (last true at v3.5, Exp664) and prompt.md still claimed "Current best:
# 2.18 (MAX-SPEED v4.5)" after v4.6 (tail flush) and v4.7 (boundary batch) shipped - a 17 % error
# in the first two lines a new session reads. Check 4b could not see it: it greps prompt.md for the
# shipped VAE *filename*, which stayed correct while every number around it went stale.
# So: ONE machine-readable state file (.auto/headline.json) and this check. It FAILS when a line
# that asserts current state does not contain the current number.
SPEC = os.path.join(HERE, 'headline.json')
if not os.path.exists(SPEC):
    bad("headline.json is missing - the loop must declare its current era/cells in ONE machine-readable place")
else:
    try:
        spec = json.load(open(SPEC, encoding='utf-8'))
        cur = float(spec['shipped']['rtf10'])
        era = str(spec['era'])
        base = float(spec['baseline'])
        tol = max(0.01, 0.01 * cur)          # cells are quoted to 2 dp; 1 % covers rounding + state drift
        # A current-state claim is recognised by FORM, not by vocabulary. The first draft matched the
        # word 'headline' anywhere and produced 6 false positives ("headline speed claim for the
        # max-speed tier", "excluded from RTF", an Exp654 scorer row) - and the draft before that
        # anchored on 'headline:' with a colon, which a cosmetic rewrite to "Headline (era v4.7):"
        # silently broke, leaving the single most-read line unchecked. Both directions of the same
        # lesson: a guard that cries wolf gets ignored, and a guard whose pattern is brittle lies.
        # Three forms, each one an idiom the docs actually use:
        #   A. "Current best: 1.87"                  - prompt.md's briefing line
        #   B. "Headline ...: 12.24 -> 1.87"         - RESULTS.md's baseline -> current arrow
        #   C. a tier-table ROW labelled as the shipping tier - both ladder tables
        FORM_A = re.compile(r'current best[:\s]*\**(\d+\.\d+)', re.I)
        FORM_B = re.compile(r'headline\b[^\n]{0,48}?(\d+\.\d+)\s*(?:\u2192|->)\s*\**' r'(\d+\.\d+)', re.I)
        ROW_CI = [('max-speed-lean', 'lean'), ('max-speed (shipped', 'shipped')]
        ROW_CS = [('(DEFAULT', 'shipped')]
        # Explicitly-historical markers, tested against the MARKER LINE only (never the 3-line window:
        # the window version excused the CURRENT lean row because an adjacent row said "pre-flush
        # cells", and excused prompt.md's Current-best paragraph because a later line said "DEAD
        # tiers" - the guard then printed "claims checked" while skipping the two that matter).
        HIST = ('as of exp', 'superseded', 'snapshot', 'pre-flush cells', 'old-build',
                'previous default', 'predecessor')
        LOGROW = re.compile(r'^\s*[-*]?\s*Exp\d+')      # per-experiment log entry = historical by shape
        seen, skipped, checked = {}, 0, 0
        prev_d = False        # did FORM_D fire on the previous line? (its 2-line window double-hits
                              # a wrapped sentence - one real claim, two FAILs, which trains people
                              # to distrust the count)
        # A claim under a HISTORY heading is history, not a current-state claim. Structural, not
        # lexical: prompt.md's whole "What's Been Tried" half is per-experiment log prose, and its
        # lines quote whatever number was current THEN ("the headline is 12.24 -> 3.48 (-71.6 %)").
        # Without this rule the check reports 4 false alarms there and gets muted within a session.
        HISTHEAD = ("what's been tried", 'history', 'archive', 'superseded', 'log of', 'changelog')
        for name in ('RESULTS.md', 'STREAMING_1P5B.md', os.path.join('.auto', 'prompt.md')):
            p = os.path.join(ROOT, name)
            if not os.path.exists(p):
                continue
            lines = open(p, encoding='utf-8', errors='replace').read().splitlines()
            section = ''
            for i, ln in enumerate(lines):
                if ln.startswith('#'):
                    section = ln.lower()
                claim = None                              # (block, direct value or None)
                m = FORM_A.search(ln)
                if m:
                    claim = ('shipped', float(m.group(1)))
                else:
                    m = FORM_B.search(ln)
                    if m:
                        claim = ('shipped', float(m.group(2)))
                    else:
                        # tier-ROW claims are TABLE ROWS. Requiring the leading '|' keeps the rule
                        # structural: prompt.md's "MODEL ARTIFACTS: ... (DEFAULT since Exp690)" prose
                        # is a filename statement, not a ladder cell, and demanding an RTF on that
                        # line is the kind of false alarm that gets a guard muted.
                        tbl = ln.lstrip().startswith('|')
                        blk = next((b for mm, b in ROW_CI if tbl and mm in ln.lower()),
                                   next((b for mm, b in ROW_CS if tbl and mm in ln), None))
                        if blk:
                            claim = (blk, None)
                        else:
                            # D. PRESENT-TENSE config claims - "the shipped default is PIECES=1 at
                            # RTF ~2.43". Found while logging this check's own coverage gap: the
                            # sentence WRAPS ("the shipped\ndefault is ..."), so no line-anchored
                            # rule can see it; hence the 2-line window. Two guards against noise:
                            # the value must be an RTF-adjacent number (else "whose default is
                            # defer-OFF): 2.54 on the protocol clip" becomes a speed claim), and the
                            # historical-section / past-tense rules above still apply - which is why
                            # docs should use PAST tense for history and present tense only for now.
                            w2 = ' '.join(lines[i:i + 2]).lower()
                            hit_d = any(t in w2 for t in ('default is', 'shipped tier is',
                                                          'currently ships', 'shipping default is',
                                                          'default tier is', 'the default now'))
                            if hit_d and prev_d:
                                hit_d = False          # same sentence, second window line
                            mv = None
                            if hit_d:
                                mv = re.search(r'rtf\s*\**[~\u2248]?(\d+\.\d+)', w2)
                                if mv:
                                    claim = ('lean' if 'lean' in w2 else 'shipped', float(mv.group(1)))
                            prev_d = bool(hit_d and mv)
                if claim is None:
                    continue
                block, val = claim
                if any(h in ln.lower() for h in HIST) or LOGROW.match(ln) or any(h in section for h in HISTHEAD):
                    skipped += 1
                    continue
                want = cur if block == 'shipped' else float(spec.get(block, {}).get('rtf10', cur))
                win = ' '.join(lines[i:i + 3])            # a row/claim may wrap
                checked += 1
                seen[block] = seen.get(block, 0) + 1
                nums = [val] if val is not None else [float(x) for x in re.findall(r'\d+\.\d+', win)]
                if not any(abs(n - want) <= tol for n in nums):
                    bad(f"{name}:{i+1} asserts CURRENT state ({block}) but quotes "
                        f"{nums[0] if val is not None else 'no current number'}; "
                        f".auto/headline.json says {spec['era']} {block} 10 s = {want} "
                        f"(Exp{spec.get(block + '_exp', '?')}). {ln.strip()[:90]} - update the prose "
                        f"OR the spec if a cell really moved, never leave the two disagreeing")
                elif spec['era'] not in (ln if val is not None else win):
                    # Tested on the CLAIM LINE for direct-value claims (A/B). Reading the 3-line
                    # window here made the era clause vacuous in wrapped paragraphs: deleting 'v4.7'
                    # from prompt.md's Current-best line still passed, because the next sentence of
                    # the paragraph mentioned it. A guard whose negative control cannot fail is noise.
                    bad(f"{name}:{i+1} states current numbers but never names the current era "
                        f"{spec['era']} on its own line - it will be read as the latest word and is "
                        f"already stale prose")
        if checked:
            ok(f"headline prose agrees with headline.json ({era}, {cur}) across {checked} current-state "
               f"claim(s): " + ', '.join(f"{b}={n}" for b, n in sorted(seen.items()))
               + (f"; {skipped} explicitly-historical claim(s) excluded" if skipped else ""))
        else:
            warn("no current-state claim line found in the docs - the headline check has nothing to check")
        # ladder cells: formatting is free, staleness is not -> WARN if a cell vanished from RESULTS.md
        missing = [f"{spec['shipped'][k]}" for k in ('rtf10', 'rtf17', 'rtf69', 'rtf138')
                   if spec['shipped'].get(k) is not None
                   and str(spec['shipped'][k]) not in open(os.path.join(ROOT, 'RESULTS.md'),
                                                           encoding='utf-8', errors='replace').read()]
        if missing:
            warn(f"RESULTS.md no longer contains the shipped ladder cell(s) {missing} - if the row was "
                 f"re-measured, update headline.json too (it is the source this check reads)")
        elif not missing and spec['shipped'].get('rtf138'):
            ok("RESULTS.md carries every shipped ladder cell declared in headline.json")
    except (KeyError, ValueError) as e:
        bad(f"headline.json is unreadable/incomplete: {e!r}")

# ---- 10. documented COMMANDS must name the shipped tier (Exp859). Check 4b compares measure.sh and
# eval40.sh's DEFAULTS against tier.env; it says nothing about the commands the DOCS tell a human to
# run, and RESULTS.md's reproduce block had drifted to the F16-conv REFERENCE file. So a reader
# following "reproduce" under the 1.87 headline measured another system and got 1.9372 (+3.5 %) - a
# plausible, correct-looking number for the wrong tier, which is the Exp694 failure class wearing a
# different hat (there it was eval40.sh's defaults; here it is prose a human trusts).
cmd_claims = 0
for name in ('RESULTS.md', 'STREAMING_1P5B.md', os.path.join('.auto', 'prompt.md'), 'README.md'):
    p_ = os.path.join(ROOT, name)
    if not os.path.exists(p_) or not TIER:
        continue
    for i, ln in enumerate(open(p_, encoding='utf-8', errors='replace').read().splitlines()):
        if 'measure.sh' not in ln and 'eval40.sh' not in ln:
            continue                      # commands only: prose mentions of an alternate tier are fine
        if any(h in ln.lower() for h in ('as of exp', 'superseded', 'snapshot', 'pre-flush',
                                         'previous default', 'reference build')):
            continue
        for k in ('VAE_FILE', 'LM_FILE'):
            for v in re.findall(k + r'=([\w.\-]+\.gguf)', ln):
                cmd_claims += 1
                if v != TIER.get(k):
                    bad(f"{name}:{i+1} documents a command that runs {k}={v}, but the shipped tier is "
                        f"{k}={TIER.get(k)} (.auto/tier.env) - a reader following this reproduces a "
                        f"DIFFERENT SYSTEM and gets a different RTF (Exp859: +3.5 % for exactly this)")
if cmd_claims:
    ok(f"{cmd_claims} documented command file-override(s) all match the shipped tier")
else:
    warn("no file overrides found in any documented command - check 10 had nothing to compare")

# ---- 11. documented FLAGS must exist in the parser they are documented against (Exp860).
# Exp855 found --kv-type q8_0 recommended by a code comment and by two of my own next-hints, in a
# binary that parses 10 flags and exits 1 on anything else. That iteration added a fault probe for
# THAT flag. This closes the class: every flag the docs pass to asr_streaming, measure.sh or
# eval40.sh is checked against the option set the real parser accepts, derived from source.
# Scope is deliberately narrow and stated: flags belonging to OTHER programs on the same line
# (cmake --build, python tools, llama-quantize) are NOT checked, because their option sets are not
# derivable here and a deny-list of foreign flags is how a guard starts lying. A flag is treated as
# an argument only when it appears AFTER the program token, which is what keeps
# "cmake --build build --target asr_streaming" from being read as asr_streaming's own interface.
BIN_FLAGS, HARNESS_FLAGS = set(), {}
try:
    dsrc = open(os.path.join(ROOT, 'demo', 'asr_streaming.cpp'), encoding='utf-8', errors='ignore').read()
    BIN_FLAGS = set(re.findall(r'"(--[a-z][\w-]*)"', dsrc)) | {'-t', '-h'}
except OSError:
    bad("demo/asr_streaming.cpp not readable - the documented-flag check has no ground truth")
for sh in ('measure.sh', 'eval40.sh'):
    try:
        t = open(os.path.join(HERE, sh), encoding='utf-8', errors='ignore').read()
        HARNESS_FLAGS[sh] = set(re.findall(r'^\s{2,4}(--[a-z][\w-]+)\)', t, re.M))
    except OSError:
        HARNESS_FLAGS[sh] = set()
if not HARNESS_FLAGS.get('measure.sh'):
    warn("no case-label flags found in measure.sh - check 11 cannot verify its interface")
flag_claims, badflags = 0, []
for name in ('RESULTS.md', 'STREAMING_1P5B.md', os.path.join('.auto', 'prompt.md'), 'README.md'):
    p_ = os.path.join(ROOT, name)
    if not os.path.exists(p_):
        continue
    for i, ln in enumerate(open(p_, encoding='utf-8', errors='replace').read().splitlines()):
        if any(h in ln.lower() for h in ('as of exp', 'superseded', 'snapshot', 'previous default')):
            continue
        # An argument belongs to a program only where the program is invoked AS ONE: the token must
        # be preceded by '/' (./asr_streaming, ./.auto/measure.sh, $RDIR/asr_streaming). First draft
        # used rfind(prog), which read `cmake --build build --target asr_streaming -j` as an
        # invocation of the binary with a -j flag, and flagged a prose cell about two co-running
        # processes - 4 false alarms from 4 lines that mention a name rather than run it. Precision
        # is the whole value of this check: Exp660's rule is that an over-claiming guard gets muted.
        for prog, truth in ([('asr_streaming', BIN_FLAGS)] +
                            [(k, v) for k, v in HARNESS_FLAGS.items()]):
            m = None
            for m in re.finditer(r'/'+re.escape(prog)+r'\b', ln):
                pass                                    # last invocation on the line wins
            if m is None:
                continue
            tail = ln[m.end():]
            for f in re.findall(r'(?<![-\w])(--?[a-z][\w-]*)', tail):
                if f in ('-c', '-s', '-o'):      # launcher-side short opts (adb/taskset conventions)
                    continue
                flag_claims += 1
                if f not in truth:
                    badflags.append(f"{name}:{i+1} documents {prog} {f} - not in its parser "
                                    f"({prog == 'asr_streaming' and 'exits 1: Unknown arg' or 'ignored or error'})")
for b in sorted(set(badflags)):
    bad(b)
# Code COMMENTS too: the fictional --kv-type was recommended by a comment in demo/asr_streaming.cpp,
# not by the docs (Exp855). Lines that DISAVOW a flag are skipped, which is how Exp855's own
# corrective comment ("there is no --kv-type flag; passing it exits 1") can stay in the tree without
# tripping the check that exists to catch its resurrected cousin.
NEG = ('does not exist', "no --", 'there is no', 'never existed', 'no longer', 'not a flag',
       'exits 1', 'is not a', 'instead of a flag', 'would need', 'is not an option')
for f in sorted(glob.glob(os.path.join(ROOT, 'src', '*.cpp')) + glob.glob(os.path.join(ROOT, 'src', '*.h'))
                + glob.glob(os.path.join(ROOT, 'demo', '*.cpp'))):
    rel = os.path.relpath(f, ROOT)
    if rel == os.path.join('demo', 'asr_streaming.cpp'):
        continue                       # this file IS the parser; its string literals are the truth set
    for i, ln in enumerate(open(f, encoding='utf-8', errors='ignore').read().splitlines()):
        if not ln.strip().startswith(('//', '*', '/*')) or any(n in ln.lower() for n in NEG):
            continue
        for fl in sorted(set(re.findall(r'--([a-z][\w-]*)', ln))):
            flag_claims += 1
            if ('--' + fl) not in BIN_FLAGS:
                badflags.append(f"{rel}:{i+1} comment recommends {('--' + fl)}, which the binary does "
                                f"not parse (it exits 1 'Unknown arg') - the Exp855 failure class")
for b in sorted(set(badflags)):
    bad(b)
if flag_claims and not badflags:
    ok(f"{flag_claims} documented/commented flag use(s) all exist in the parsers they name")

# ---- 12. the SWEEP harness must resolve to the shipping tier (Exp865c). run_rtf_multi.sh used to expand
# empty threads/pieces fields into NOTHING, which SHIFTED bench_device.sh's positionals (the run tag landed
# in the thread-count argument) - so eight anti-overfit guard rotations measured a config that is not the
# shipping one, ~1.2 % slow, and every cross-session comparison inherited that offset. The tool now has a
# --dry mode; the assertion below is that its resolved argv equals the tier, plus a negative control proving
# the tool rejects a malformed arm instead of silently shifting arguments again.
try:
    tier_pieces = None
    for ln in open('.auto/tier.env', encoding='utf-8', errors='ignore'):
        if ln.startswith('PIECES='):
            tier_pieces = ln.strip().split('=', 1)[1]
    sh = os.path.join(ROOT, '.auto', 'run_rtf_multi.sh')
    r = subprocess.run([sh, '--dry', 'probe|clip.wav'], capture_output=True, text=True, timeout=60)
    line = next((l for l in (r.stdout + r.stderr).splitlines() if l.startswith('DRY probe')), '')
    if r.returncode != 0 or not line:
        bad(f"run_rtf_multi.sh --dry failed (exit {r.returncode}) - the sweep harness cannot be verified; "
            f"fix it rather than trusting any number it produced")
    else:
        want_p = (line.split('pieces=')[1].split()[0] if 'pieces=' in line else '?')
        want_t = (line.split('threads=')[1].split()[0] if 'threads=' in line else '?')
        if tier_pieces and want_p != tier_pieces:
            bad(f"sweep defaults to pieces={want_p} but tier.env says PIECES={tier_pieces} - the guard board "
                f"would measure a different configuration than the metric (Exp865c's bug class)")
        elif want_t != '2':
            bad(f"sweep defaults to threads={want_t}, not the shipped 2 threads on the A78 primes")
        else:
            ok(f"sweep harness resolves to the shipping tier (threads={want_t}, pieces={want_p}, from tier/measure defaults)")
        if 'vae-encoder-convint8.gguf' not in line or 'lm-q8head.gguf' not in line:
            bad(f"sweep probe does not name the shipped files: {line}")
    r2 = subprocess.run([sh, '--dry', 'probe|clip.wav|2|9|'], capture_output=True, text=True, timeout=60)
    if r2.returncode == 0:
        bad("run_rtf_multi.sh ACCEPTED pieces=9 - the argument validation is gone, so a malformed arm would "
            "again be measured with shifted positionals instead of failing (Exp660 rule: prove the guard fires)")
    else:
        ok("sweep harness rejects a malformed arm (exit != 0), so argument drift fails loudly")
except FileNotFoundError:
    bad("run_rtf_multi.sh missing - the guard rotation cannot be verified")
except Exception as e:
    bad(f"sweep-harness check raised {type(e).__name__}: {e}")

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
