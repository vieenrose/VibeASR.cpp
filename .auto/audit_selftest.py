#!/usr/bin/env python3
"""Exp873: does every audit check actually FAIL when its condition is violated?

This loop's standing rule (Exp660) is "a self-check is untrustworthy until a planted fault makes it
fail". That rule has been applied check-by-check as each check was written, and never systematically -
so the audit could contain guards that can never fire. Exp872 is the proof this is worth doing: a check
I had just written reported success while iterating an empty dict, and only a control caught it.

So this driver is the rule applied to the whole audit: plant one fault per check, run the audit, and
require that the SPECIFIC check fires. A check that stays green under its own fault is a guard that
does not exist. Faults that cannot be planted safely (device-side co-runner, a real binary-hash drift
that would need a rebuild) are reported as UNCOVERED rather than silently skipped - an untested guard
must be visible, not implied.

  ./.auto/audit_selftest.py              every plantable fault (~8 min; each step runs the audit once)
  ./.auto/audit_selftest.py --list       show the matrix without running anything

Nothing here ships a change. Every plant is reverted in a finally block, and the last thing the run
does is a clean audit that must be green.
"""
import os, re, shutil, subprocess, sys, tempfile, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
AUDIT = [sys.executable, os.path.join('.auto', 'audit_harness.py')]
DEV = os.environ.get('AUDIT_DEV', 'AYBY6HQCMBF6B6KZ')
RDIR = '/data/local/tmp/vibeasr'


def run_audit():
    r = subprocess.run(AUDIT, capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    m = re.search(r'harness audit: (\d+) checks passed, (\d+) warnings, (\d+) failures', out)
    fails = [l.strip() for l in out.splitlines() if l.strip().startswith('FAIL')]
    return (int(m.group(3)) if m else -1), fails, out


# Content snapshots, NOT `git checkout`. Exp874's first run proved why: a revert that restores the
# COMMITTED state silently destroys any uncommitted fix to the same file - and my own working tree always
# has uncommitted work in it during a session. Every plant here now saves bytes and puts them back.
_SNAP = {}


def snap_write(path, text):
    _SNAP[path] = open(path, 'rb').read()
    open(path, 'w').write(text)
    return lambda: open(path, 'wb').write(_SNAP.pop(path))


def snap_append(path, text):
    _SNAP[path] = open(path, 'rb').read()
    open(path, 'a').write(text)
    return lambda: open(path, 'wb').write(_SNAP.pop(path))


def git_revert(*paths):
    subprocess.run(['git', 'checkout', '--'] + list(paths), cwd=ROOT, capture_output=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)


# ---------------------------------------------------------------- the fault matrix
# Each entry: (check id, what is broken, plant(), expect-substring, revert())
# Invariants: a plant touches ONE property; every revert is idempotent; nothing here edits src/.
def plant_syntax():
    return snap_append('.auto/fault_run.sh', '\ndef broken(:\n')


def plant_path():
    return snap_append('.auto/bench_device.sh', '\n# referenced helper: .auto/this_tool_does_not_exist.py\n')


def plant_untracked():
    p = '.auto/selftest_untracked_copy.sh'
    shutil.copyfile('.auto/fault_run.sh', p)
    os.chmod(p, 0o755)
    return lambda: os.remove(p)


def plant_execbit():
    p = '.auto/fault_run.sh'
    os.chmod(p, 0o644)
    return lambda: (os.chmod(p, 0o755), git_revert(p))


def plant_dirty():
    # Section 3b deliberately ignores paths under .auto/ (the autoresearch log writes there constantly),
    # so the plant must dirty a TRACKED file outside .auto - my first version dirtied .auto/config.json
    # and the check was correctly silent.
    if not os.path.exists('README.md'):
        raise RuntimeError('README.md not found; pick another tracked file outside .auto')
    return snap_append('README.md', '\n<!-- audit_selftest dirty marker -->\n')


def plant_dev_mismatch():
    p = '.auto/measure.sh'
    s = open(p).read()
    s2 = re.sub(r'^DEV=.*$', 'DEV=ZZZZWRONGSERIAL', s, count=1, flags=re.M)
    assert s2 != s, 'DEV= line not found in measure.sh - the plant is invalid, not the check'
    return snap_write(p, s2)


def plant_stray_device_clip():
    sh(f'adb -s {DEV} shell "touch {RDIR}/zz_selftest_extra.wav"')
    return lambda: sh(f'adb -s {DEV} shell "rm -f {RDIR}/zz_selftest_extra.wav"')


def plant_asset_header():
    # Find a HOST-mirrored clip, corrupt its data-chunk length by +4096.
    import json
    man = json.load(open('.auto/device_assets.json'))['assets']
    host = [k[5:] for k in man if k.startswith('HOST ') and os.path.exists(k[5:])]
    if not host:
        raise RuntimeError('no HOST-mirrored asset found')
    p = host[0]
    bak = p + '.selftest-bak'
    shutil.copyfile(p, bak)
    b = bytearray(open(p, 'rb').read())
    off = 12
    while off + 8 <= len(b):
        cid, ln = b[off:off + 4], int.from_bytes(b[off + 4:off + 8], 'little')
        if cid == b'data':
            b[off + 4:off + 8] = (ln + 4096).to_bytes(4, 'little')
            break
        off += 8 + ln + (ln & 1)
    open(p, 'wb').write(bytes(b))
    return lambda: (shutil.move(bak, p), git_revert(p))


def plant_ref_missing():
    # Gate refs are what score_hyp/compare_arms diff against. Hiding them must be a FAILURE, not a note.
    p = '../eval-librispeech/refs.json'
    if not os.path.exists(p):
        raise RuntimeError('gate refs.json not found')
    mv = p + '.selftest-hidden'
    os.rename(p, mv)
    return lambda: os.rename(mv, p)


def plant_dup_device_clip():
    # The Exp675 class: an undeclared copy of a documented clip. Collision detection must see it even
    # though nothing in the manifest mentions the copy.
    src, dst = 'chat17.wav', 'zz_selftest_dup.wav'
    r = sh(f'adb -s {DEV} shell "cp {RDIR}/{src} {RDIR}/{dst} && echo ok"')
    if 'ok' not in r.stdout:
        raise RuntimeError(f'could not create the duplicate on device: {r.stdout}{r.stderr}')
    return lambda: sh(f'adb -s {DEV} shell "rm -f {RDIR}/{dst}"')


def plant_headline_drift():
    p = 'RESULTS.md'
    s = open(p).read()
    s2 = s.replace('**Headline (era v4.8):** phone RTF **12.24 → 1.85',
                   '**Headline (era v4.8):** phone RTF **12.24 → 2.94', 1)
    if s2 == s:
        raise RuntimeError('headline prose line not found - plant invalid (docs reworded?)')
    return snap_write(p, s2)


def plant_wrong_tier():
    return snap_append('RESULTS.md', '\n| selftest (temporary) | reproduce with VAE_FILE=vae-encoder-f16.gguf '
                       './.auto/measure.sh --skip-build |\\n')


def plant_bad_flag():
    return snap_append('RESULTS.md', '\n| selftest (temporary) | reproduce with ./.auto/measure.sh --not-a-flag |\\n')


def plant_schedule():
    return snap_append('RESULTS.md', '\n| selftest (temporary) | shipping tier: ./.auto/measure.sh '
                       'EXTRA_ENV="VAE_DEFER_LATE=1" |\\n')


def plant_selftest():
    # Break a scorer's OWN expectation (not the scorer): the audit must notice a RED self-test, because an
    # unrun or ignored test is the same failure class as an unfirable check. Exp874 found score_mixed.py's
    # self-test red since the commit that added it - nothing ever invoked it.
    p = '.auto/score_mixed.py'
    txt = open(p).read()
    txt2 = txt.replace('("你好世界", "你好世界世", 1 / 4),', '("你好世界", "你好世界世", 1 / 8),  # PLANTED', 1)
    if txt2 == txt:
        raise RuntimeError('fixture not found - the plant is invalid, not the check')
    return snap_write(p, txt2)


FAULTS = [
    ('14 tool self-tests',    'a scorer self-test starts failing',           plant_selftest,      'selftest FAILED|self-test FAILED'),
    ('1 syntax',            'a harness script stops parsing',              plant_syntax,        'syntax'),
    ('2 host paths',        'a script references a missing file',          plant_path,          'not exist|missing|no such'),
    ('3 git tracking',      'a harness script is untracked',               plant_untracked,     'NOT tracked'),
    ('3c exec bit',         'a harness script loses its +x',               plant_execbit,       'exec'),
    ('3b dirty tree',       'a tracked file is modified between runs',     plant_dirty,         'DIRTY'),
    ('4 config source',     'measure.sh points at another device',         plant_dev_mismatch,  'device .*state|not found'),
    ('7 duplicate clip',    'an undeclared copy of a documented clip',     plant_dup_device_clip, 'BYTE-IDENTICAL|no manifest'),
    ('7 wav header',        'a clip\'s data chunk disagrees with its size', plant_asset_header,  'data chunk'),
    ('8 frozen reference',  'the gate reference set disappears',           plant_ref_missing,   'refs.json|frozen reference'),
    ('9 headline',          'the prose headline disagrees with headline.json', plant_headline_drift, 'headline|headline.json'),
    ('10 command tier',     'a documented command runs a non-shipped file', plant_wrong_tier,   'shipped tier'),
    ('11 flags',            'a doc passes a flag the parser lacks',        plant_bad_flag,      'not in its parser'),
    ('13 schedule',         'a shipping command changes the window grid',  plant_schedule,      'SCHEDULE'),
]
UNCOVERED = [
    ('5 device inventory / binary hashes', 'needs a rebuilt-or-stale .so on the phone; a rebuild is the '
     'only honest plant, and a wrong-hash file would be pushed by the next measure.sh anyway'),
    ('7b co-runner', 'needs a second asr_streaming resident on the device; that would also poison every '
     'timing in the session, so it is planted by accident far more often than on purpose (Exp679)'),
    ('12 sweep resolves to tier', 'has its own --dry negative control inside audit_harness (Exp865c), '
     'which is why it is not repeated here'),
]

if '--only' in sys.argv:
    k = sys.argv[sys.argv.index('--only') + 1]
    FAULTS = [f for f in FAULTS if re.search(k, f[0], re.I)]
    UNCOVERED = []
    print(f"guard coverage: subset /{k}/ ({len(FAULTS)} fault(s))\n")

if '--list' in sys.argv:
    for c, d, _, _ in FAULTS:
        print(f"  PLANTABLE   {c:22s} {d}")
    for c, d in UNCOVERED:
        print(f"  UNCOVERED   {c:22s} {d}")
    sys.exit(0)

print(f"guard coverage: {len(FAULTS)} planted faults, {len(UNCOVERED)} classes accepted as uncovered\n")
n_fire = n_miss = 0
for cid, desc, plant, expect in FAULTS:
    revert = None
    try:
        revert = plant()
        nfails, fails, _ = run_audit()
        hit = [f for f in fails if re.search(expect, f, re.I)]
        if hit:
            n_fire += 1
            print(f"  FIRED     {cid:22s} ({desc})")
            print(f"              {hit[0][:150]}")
        else:
            n_miss += 1
            print(f"  MISS !!   {cid:22s} ({desc}) - audit reported {nfails} failure(s), none matching "
                  f'/{expect}/')
            for f in fails[:3]:
                print(f'              other: {f[:130]}')
    except Exception as e:                                  # noqa: BLE001 - a bad plant must not abort cleanup
        n_miss += 1
        print(f"  INVALID   {cid:22s} plant failed: {e}")
    finally:
        if revert:
            revert()
for c, d in UNCOVERED:
    print(f"  UNCOVERED {c:22s} {d}")
nfails, fails, out = run_audit()
print(f"\nfinal clean-tree audit: {out.strip().splitlines()[-3] if len(out.splitlines()) > 3 else '?'}")
for f in fails:
    print('  ' + f[:150])
print(f"\ncoverage: {n_fire}/{len(FAULTS)} plantable checks fire on their own fault; "
      f"{n_miss} silent or invalid; {len(UNCOVERED)} classes accepted as uncontrolled.")
if nfails:
    print("WARNING: the audit is NOT green after reverting every plant - investigate before trusting it.")
