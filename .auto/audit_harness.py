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
import os
import re
import subprocess
import sys

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

# ---- 6. frozen references used by A/B tooling ----------------------------------
for f in sorted(os.listdir(HERE)):
    if f.startswith('ref') and f.endswith('.txt'):
        if os.path.getsize(os.path.join(HERE, f)) < 20:
            bad(f"frozen reference {f} is empty/truncated")
refs_dir = os.path.join(ROOT, '..', 'eval-librispeech')
print("frozen references present: "
      + ', '.join(f for f in sorted(os.listdir(HERE)) if f.startswith('ref') and f.endswith('.txt'))
      + f"; gate refs.json: {'found' if os.path.exists(os.path.join(refs_dir, 'refs.json')) else 'MISSING'}")

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
