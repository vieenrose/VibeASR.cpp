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
does is a clean audit that must be green. Plants are host-side text edits only (fault 17 touches a
tracked source file, but only through a byte snapshot that is restored - the Exp874 rule).
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


def _snap(path):
    # A plant must leave NO trace, including mtime. Exp897: fault 17's byte-perfect revert still
    # updated demo/asr_streaming.cpp's mtime, which made run_rtf_multi's freshness guard (Exp662:
    # refuse unless the binary is newer than every source) reject the NEXT sweep with a stale-binary
    # error on a content-current tree. So the snapshot covers the stat too, not just the bytes.
    st = os.stat(path)
    _SNAP[path] = (open(path, 'rb').read(), st.st_atime_ns, st.st_mtime_ns)


def _restore(path):
    data, atime_ns, mtime_ns = _SNAP.pop(path)
    open(path, 'wb').write(data)
    os.utime(path, ns=(atime_ns, mtime_ns))


def snap_write(path, text):
    _snap(path)
    open(path, 'w').write(text)
    return lambda: _restore(path)


def snap_append(path, text):
    _snap(path)
    open(path, 'a').write(text)
    return lambda: _restore(path)


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
    # Exp946: this plant used to replace the LITERAL '**Headline (era v4.8):** phone RTF **12.24 -> 1.85',
    # so the Exp945 regime rewrite (1.85 -> 1.19) made it INVALID - the board reported that loudly, but
    # the consequence was that the headline CHECK went untested until someone read the INVALID line.
    # Match the FORM (a headline line's arrow) instead of the number, so a doc rewrite cannot disarm it.
    p = 'RESULTS.md'
    s = open(p).read()
    m = re.search(r'(\*\*Headline[^\n]*?\*\*\s*phone RTF\s*\*\*12\.24\s*\u2192\s*)(\d+\.\d+)', s)
    if not m:
        raise RuntimeError('headline prose line not found - plant invalid (docs reworded?)')
    s2 = s[:m.start(2)] + '2.94' + s[m.end(2):]
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


def plant_bad_silence():
    # Exp946: the derivation grammar grew synthetic SILENCE parts at Exp936 (long250.wav). The existing
    # 7e plant rewrites a whole concat; this one perturbs a SILENCE length by ONE SAMPLE, which is the
    # exact failure mode the arithmetic check exists for (a clip rebuilt with a different gap) and which
    # was demonstrated by hand at Exp936 but never entered the board.
    p = '.auto/device_assets.json'
    import json as _json
    man = _json.loads(open(p).read())
    hit = False
    for r in man.get('derives') or []:
        if r.get('file') == 'long250.wav':
            for part in r.get('concat') or []:
                if isinstance(part, dict) and 'silence_bytes' in part:
                    part['silence_bytes'] = int(part['silence_bytes']) + 2   # +1 s16 sample
                    hit = True
    if not hit:
        raise RuntimeError('long250 silence part not found - plant invalid (manifest reworded?)')
    return snap_write(p, _json.dumps(man, indent=1, sort_keys=True))


def plant_false_derivation():
    # The clip-composition check must fail when a note describes audio that was NOT used, WITHOUT touching
    # any clip's bytes (so the blessed-hash check stays green and this stays single-variable).
    p = '.auto/device_assets.json'
    import json as _json
    man = _json.loads(open(p).read())
    hit = False
    for r in man.get('derives') or []:
        if r.get('file') == 'chat155.wav':
            r['concat'] = ['chat69.wav', 'chat17.wav']     # wrong source, right shape
            hit = True
    if not hit:
        raise RuntimeError('derivation rule not found - plant invalid (manifest reworded?)')
    return snap_write(p, _json.dumps(man, indent=1, sort_keys=True))


def plant_help_default():
    # Check 17 compares the usage text against the struct in demo/asr_streaming.cpp, so corrupt the
    # STRUCT side: n_ctx 4096 -> 4444 must make check 17 fail naming -c. A byte snapshot (not a git
    # revert) restores it - the Exp874 rule.
    p = 'demo/asr_streaming.cpp'
    s = open(p, 'rb').read()
    s2 = s.replace(b'int n_ctx = 4096;', b'int n_ctx = 4444;', 1)
    if s2 == s:
        raise RuntimeError('struct-default anchor for -c not found; check 17 cannot be planted')
    return snap_write(p, s2.decode())


def plant_telemetry_gap():
    # Check 18 compares last_out.txt against device_state.tsv, so hide the TSV: with a capture
    # present and no TSV, the check must fail naming the telemetry gap. Rename (not delete), so
    # the revert is exact. If no TSV exists the plant is INVALID rather than silent - it means the
    # audit is already red on check 18 and there is nothing to prove.
    p, q = '.auto/device_state.tsv', '.auto/device_state.tsv.selftest_hidden'
    if not os.path.exists(p):
        raise RuntimeError('no device_state.tsv to hide; audit should already fail check 18')
    os.rename(p, q)
    return lambda: os.rename(q, p)


def plant_unguarded_grep():
    # Reproduce the Exp876 class in one line: an unguarded $(grep ...) assignment in a set -e script.
    return snap_append('.auto/checks.sh', '\nZZ_SELFTEST=$( grep -oE zzz .auto/config.json | head -n1 )\n')


def plant_stale_device_lib():
    # The device-binary-hash check (Exp660's "a stale lib silently measures the OLD kernels") had never
    # been controlled: it was the one coverage hole that mattered, because this loop has actually been
    # burned by stale objects (the Exp595-607 flag sweeps measured a binary that was never rebuilt).
    # Plant: append one byte to the device's libllama.so, so the bytes differ from the host's while the
    # file stays where the harness expects it. Restore is a device-side mv of the backup (no host path).
    lib, bak = f'{RDIR}/libllama.so', f'{RDIR}/libllama.so.selftest-bak'
    r = sh(f'adb -s {DEV} shell "cp {lib} {bak} && printf x >> {lib} && echo ok"')
    if 'ok' not in r.stdout:
        raise RuntimeError(f'could not mutate the device lib: {r.stdout}{r.stderr}')
    def restore():
        sh(f'adb -s {DEV} shell "mv {bak} {lib} && echo restored"')
    return restore


def plant_arm_guard():
    # Exp997: check 19 (the arm's <2000 MHz state guard must still exist) was added in Exp988b after that
    # guard was silently deleted by a refactor. A new guard needs a plant, or the coverage board reports
    # "all checks fire" while the newest one is untested - exactly the gap this board exists to close.
    # Plant: rename the guard's message inside measure.sh; the audit must FAIL naming check 19.
    p = '.auto/measure.sh'
    s = open(p, encoding='utf-8').read()
    if 'WARNING: arm ran but cpu7_deliv_mhz=' not in s:
        raise RuntimeError('arm-guard message not found in measure.sh; check 19 cannot be planted')
    return snap_write(p, s.replace('WARNING: arm ran but cpu7_deliv_mhz=',
                                   'PLANTED-ARM-GUARD-GONE:', 1))


FAULTS = [
    ('5 device binary hash',  'a lib on the phone is not the one on the host', plant_stale_device_lib,
     'MISMATCH|missing on device'),
    ('15 set -e grep',      'a grep in $( ) can abort a set -e script',   plant_unguarded_grep,   'unguarded command substitution'),
    ('17 usage defaults',   'a usage line promises a default the struct lacks', plant_help_default, 'check 17.*-c'),
    ('19 arm guard',        'the arm state-guard is deleted from measure.sh',  plant_arm_guard,
     'check 19.*MISSING'),
    ('18 telemetry gap',    'runs stop being recorded in device_state.tsv',   plant_telemetry_gap,  'not being recorded'),
    ('7e derivation',         'a clip note describes audio that was not used', plant_false_derivation,
     'derivation NOT proven|derivation .*payload lengths|is NOT a prefix'),
    ('7f silence part',       'a derivation gap is one sample too long',      plant_bad_silence,
     'payload lengths sum|NOT proven'),
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
    ('7b co-runner', 'needs a second asr_streaming resident on the device; that would also poison every '
     'timing in the session, so it is planted by accident far more often than on purpose (Exp679)'),
    ('12 sweep resolves to tier', 'has its own --dry negative control inside audit_harness (Exp865c), '
     'which is why it is not repeated here'),
    ('16 capture identity', 'WARN-only by design (a non-protocol capture is legitimate mid-session), and '
     'this driver matches FAIL lines - it cannot fire here by construction. Manual control instead: '
     'point .auto/last_out.txt at a long-clip capture and confirm the WARN names its window count.'),
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
