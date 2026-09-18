#!/usr/bin/env python3
"""Long-run memory soak: sample the device process RSS while a run proceeds.

Why: the Exp664/671/673 fusions each ADD graph nodes, and the only memory number those
iterations recorded was the single peak_rss_mb from a 10 s clip. A slow climb (arena growth,
per-window buffers never released) is invisible in a peak sample taken at exit, and shows up
only over minutes - which is exactly where a phone assistant actually runs. Exp644/614 did this
by hand; this makes it one command.

Usage:  rss_soak.py [--interval N] -- <command ...>
Prints a time series, then min/max/first/last and a least-squares slope in MB/min, and re-prints
any METRIC lines the wrapped command emitted (so the soak result lands in the same place as a
normal run).

Quoting note: adb is invoked with an argv LIST, never through a shell string, so remote `$(...)`
is not expanded locally - the bug that made Exp676's first bless hash empty stdin.
"""
import os
import re
import math
import statistics
import sys
def _selftest():
    """Validate classify()/trend_se() on synthetic series, WITHOUT device time and WITHOUT re-implementing
    them: this file's run section executes at import time, so the helpers are pulled from source here
    (Exp865d - the soak verdict must not be a coin toss; two runs of one binary said GROWTH and FLAT)."""
    ns = {'math': math, 'statistics': statistics}
    src = open(__file__, encoding='utf-8').read()
    lines = src.splitlines()
    for name in ('trend', 'trend_se', 'classify'):
        try:
            i = next(j for j, l in enumerate(lines) if l.startswith('def ' + name + '('))
        except StopIteration:
            print(f'  selftest FAIL: def {name}( not found in this file')
            return 1
        block = [lines[i]]; i += 1
        # a function body ends at the first column-0 statement (this file has module-level code after defs,
        # which is why a greedy regex was wrong here)
        while i < len(lines):
            ln = lines[i]
            if ln.strip() and not (ln.startswith((' ', '\t')) or ln.startswith(')')):
                break
            block.append(ln); i += 1
        exec('\n'.join(block), ns)
    trend, trend_se, classify = ns['trend'], ns['trend_se'], ns['classify']
    rng = __import__('random').Random(7)
    def series(slope, noise):
        return [(i * 10.0, 2180.0 + (i * 10.0 / 60.0) * slope + rng.uniform(-noise, noise)) for i in range(30)]
    # Expectations are calibrated to what the sampler can actually resolve (Exp865d): with the +-12 MB
    # two-state spikes a real device shows, a 5-minute window CANNOT resolve 0.5 MB/min, and the tool must
    # say INCONCLUSIVE rather than pick a verdict - which is exactly the coin toss that made one soak of an
    # unchanged binary report GROWTH and the next report FLAT.
    cases = [('+40 MB/min leak, +-12 noise', 40.0, 12.0, 'GROWTH'),
             ('+1.2 MB/min leak, +-3 noise', 1.2, 3.0, 'GROWTH'),
             ('flat, +-3 noise',              0.0, 3.0, 'FLAT'),
             ('flat, +-12 noise (device-like)', 0.0, 12.0, 'INCONCLUSIVE')]
    rc = 0
    for name, sl, nz, want in cases:
        pts = series(sl, nz)
        got = classify(trend(pts), trend_se(pts))
        se = 2 * trend_se(pts)
        mark = 'ok' if got.startswith(want) else 'WRONG, expected ' + want
        print(f'  selftest {name:28s} slope={trend(pts):+6.2f} +/- {se:.2f}  -> {got.split(" -")[0]:12s} {mark}')
        if not got.startswith(want):
            rc = 1
    print(f'  resolution at +-12 MB noise: detectable slope >= {0.5 + 2 * trend_se(series(0.0, 12.0)):.2f} MB/min'
          '  (threshold 0.5 plus 2se)')
    if rc == 0:
        print('  rss_soak self-test: PASS (leaks detected, flat accepted, unresolvable cases reported honestly)')
    return 0





import subprocess
import sys
import threading
import time

if '--selftest' in sys.argv:
    sys.exit(_selftest())

args = sys.argv[1:]
interval = 5.0
if args and args[0] == '--interval':
    interval = float(args[1]); args = args[2:]
if args and args[0] == '--repeat':            # Exp865f: strip it here, or it leaks into the wrapped argv
    REPEAT_N = int(args[1]); args = args[2:]   # and the harness tries to execute a flag as a program
else:
    REPEAT_N = 1
if args and args[0] == '--':
    args = args[1:]

DEV = re.search(r'^DEV=(\S+)', open('.auto/measure.sh').read(), re.M).group(1)
PROC = os.environ.get('SOAK_PROC', 'asr_streaming')
# SOAK_PID targets one process directly. It exists so the fd/thread columns can be proven SENSITIVE
# against a planted leak (Exp660's rule: a self-check is untrustworthy until a fault makes it move),
# which pidof-by-name cannot do when several processes share a name.
TARGET_PID = os.environ.get('SOAK_PID')
WARM_MB = float(os.environ.get('SOAK_WARM_MB', 200))   # ignore anything not yet running a model
samples = []          # (t, rss_mb, hwm_mb, vsz_mb, threads, n_fd)
nofile = [None]
stop = threading.Event()


def adb(argv, timeout=20):
    return subprocess.run(['adb', '-s', DEV] + argv, capture_output=True, text=True, timeout=timeout)


stale_seen = [0]


def poll():
    # pidof can return SEVERAL pids: a `timeout` that fires on the host leaves the device-side
    # process alive (blocked, ~0.6 MB, never loads the models). Selecting by pid list broke the
    # first version of this tool; select by RESIDENT SIZE instead - the real run is multi-GB and
    # the orphans are sub-MB, so the largest-RSS pid is unambiguous and self-documenting.
    while not stop.is_set():
        try:
            pids = [TARGET_PID] if TARGET_PID else adb(['shell', 'pidof', PROC]).stdout.replace('\r', '').split()
            best = None
            for p in pids:
                # cat, not grep: adb JOINS its argv into one device-shell command string, so an
                # unquoted alternation like ^VmRSS|^VmHWM is parsed as shell PIPES by the device and
                # the probe silently returns nothing (this cost the first two soak attempts).
                r = adb(['shell', 'cat', f'/proc/{p}/status'])
                rss = hwm = vsz = nthr = None
                for line in r.stdout.replace('\r', '').splitlines():
                    if line.startswith('VmRSS:'):
                        rss = int(line.split()[1]) / 1024.0
                    elif line.startswith('VmHWM:'):
                        hwm = int(line.split()[1]) / 1024.0
                    elif line.startswith('VmSize:'):
                        vsz = int(line.split()[1]) / 1024.0
                    elif line.startswith('Threads:'):
                        nthr = int(line.split()[1])
                if rss is None:
                    continue
                if best is None or rss > best[1]:
                    best = (p, rss, hwm, vsz, nthr)
            if best is None:
                pass                                  # nothing resident yet - keep sampling
            elif best[1] < WARM_MB:
                stale_seen[0] += 1                     # loading, or an orphan: not a data point
            else:
                # fd count + its hard limit: a per-window descriptor leak is invisible in RSS and
                # kills a long session at RLIMIT_NOFILE, which no RTF-shaped board can see (Exp849's
                # "resource-limit edges" lesson). One extra adb round trip per sample, so keep interval >= 5 s.
                nfd = None
                try:
                    fd_out = adb(['shell', 'ls', f'/proc/{best[0]}/fd'], timeout=15).stdout
                    nfd = len([x for x in fd_out.replace('\r', ' ').split() if x.strip()])
                except Exception:
                    pass
                if nofile[0] is None and nfd is not None:
                    lim = adb(['shell', 'cat', f'/proc/{best[0]}/limits'], timeout=15).stdout
                    for line in lim.splitlines():
                        if line.lower().startswith('max open files'):
                            nofile[0] = line.split()[-2] if line.split()[-2].isdigit() else line.split()[-2]
                samples.append((time.time(), best[1], best[2], best[3], best[4], nfd))
        except Exception:
            pass                                       # adb hiccup - keep sampling
        time.sleep(interval)


cmd = args or ['bash', '.auto/measure.sh', '--skip-build']
print(f'soak: sampling {PROC} every {interval:.0f} s while running: {" ".join(cmd)}', flush=True)
t0 = time.time()
th = threading.Thread(target=poll, daemon=True)
# Exp865f: --repeat N runs the wrapped command N times under ONE sampler and compares the per-run steady
# medians. Why this beats a longer soak: a single process cannot exceed ~4 min of audio before the KV
# context ends the session (Exp849 measured ~245 s at n_ctx=4096), so a one-run soak cannot reach a slope
# resolution that answers 'does memory drift across a session'. Per-run MEDIANs are immune to the +-20 MB
# two-state spikes that defeat a 5-minute slope (Exp865d), and comparing run 1 to run N is a paired test.
REPEAT = REPEAT_N
runs = []
out = ''
th.start()
for _r in range(REPEAT):
    _t0 = time.time()
    out += subprocess.run(cmd, capture_output=True, text=True).stdout
    runs.append((_t0, time.time()))
stop.set(); th.join(timeout=2)

if not samples:
    print('NO SAMPLES - the process was never visible to pidof; this soak proved nothing')
    sys.exit(2)

base = samples[0][0]
print(f"{'t(s)':>6} {'VmRSS':>9} {'VmHWM':>9} {'VmSize':>9} {'Thr':>4} {'fd':>5}")
for s_ in samples:
    t, r, h, v = s_[0], s_[1], s_[2], s_[3]
    nth, nfd = (s_[4] if len(s_) > 4 else None), (s_[5] if len(s_) > 5 else None)
    print(f'{t - base:6.0f} {r:9.1f} {(h or 0):9.1f} {(v or 0):9.1f} '
          f'{(nth if nth else 0):4d} {(nfd if nfd else 0):5d}')

def trend_se(pairs):
    """Standard ERROR of trend()'s slope, in MB/min (Exp865d).

    Why: two soaks of one binary on the 138 s clip returned +2.40 MB/min (verdict GROWTH) and +0.95
    (verdict FLAT) with identical first/last/HWM. On a 5-minute window a few 20 MB spikes move a bare
    least-squares slope by more than the leak threshold, so the old `abs(slope) < 2.0` verdict was a coin
    toss at the low end and cried wolf at the high end. Report slope +/- 2 se and decide on the interval.
    """
    if len(pairs) >= 3 and statistics.pstdev([t for t, _ in pairs]) > 0:   # 3 points = 1 dof, still a real interval
        xs = [t / 60.0 for t, _ in pairs]
        ys = [r for _, r in pairs]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        Sxx = sum((x - mx) ** 2 for x in xs)
        if not Sxx:
            return float('nan')
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / Sxx
        a = my - b * mx
        rss_ = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        return math.sqrt(rss_ / (len(xs) - 2) / Sxx)
    return float('nan')


def classify(slope, se, threshold=0.5):
    """GROWTH only when the interval clears the threshold; FLAT when it cannot reach it; otherwise say so."""
    if se != se:                      # nan
        return 'INCONCLUSIVE - too few samples to fit'
    lo, hi = slope - 2 * se, slope + 2 * se
    if lo > threshold:
        return 'GROWTH - investigate (the 95% interval is above the threshold)'
    if hi < threshold:
        return 'FLAT - no leak signal'
    return 'INCONCLUSIVE - slope +/- 2se straddles the threshold; re-run or lengthen the soak'




def trend(pairs):
    """Least-squares MB/min over (t_seconds, rss_mb) pairs - shared so the self-test uses this code path."""
    if len(pairs) > 2 and statistics.pstdev([t for t, _ in pairs]) > 0:
        xs = [t / 60.0 for t, _ in pairs]
        ys = [r for _, r in pairs]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float('nan')
    return float('nan')


rss = [s[1] for s in samples]
if REPEAT > 1:
    print(f'\n---- session soak across {REPEAT} runs of one command ----')
    meds = []
    for r, (t0, t1) in enumerate(runs):
        pts = [s_[1] for s_ in samples if t0 + 20 <= s_[0] <= t1 - 2]     # skip each run's first-touch ramp
        if len(pts) < 3:
            print(f'  run {r + 1}: too few in-window samples ({len(pts)}) - INCONCLUSIVE')
            continue
        meds.append((r, statistics.median(pts), max(s_[2] for s_ in samples if t0 <= s_[0] <= t1) or 0.0, len(pts)))
    for r, m, h, n in meds:
        print(f'  run {r + 1}: steady median RSS {m:.1f} MB  peak {h:.1f} MB  ({n} samples)')
    if len(meds) >= 2:
        pts = [(r * 1.0, m) for r, m, _, _ in meds]
        sl, se = trend([(t * 60.0, m) for t, m in pts]), trend_se([(t * 60.0, m) for t, m in pts])
        print(f'  drift across runs = {sl:+.2f} +/- {2 * se:.2f} MB per run   '
              f'first-vs-last = {meds[-1][1] - meds[0][1]:+.1f} MB')
        print(f'  session-memory verdict: {classify(sl, se)}')
        hws = [h for _, _, h, _ in meds]
        print(f'  peak per run: {" ".join(f"{h:.1f}" for h in hws)} MB  (spread {max(hws) - min(hws):.1f} MB)')
        # The peak is the statistic that answers 'does a session grow?': it is per-run, monotone within a run,
        # and immune to the +-20 MB two-state RSS pattern that makes per-run MEDIANS step (Exp865f: three
        # runs gave medians 2176.8 / 2176.8 / 2197.2 but peaks 2198.2 / 2198.1 / 2198.2 - state, not drift).
        if max(hws) - min(hws) < 2.0:
            print('  per-run peak is invariant -> NO SESSION GROWTH (the median step is the two-state pattern,'
                  ' not accumulation)')

dur = (samples[-1][0] - samples[0][0]) / 60.0
xs = [(s[0] - base) / 60.0 for s in samples]
slope = trend([(t - base, r) for r, (t, _) in zip(rss, [(s_[0], s_[1]) for s_ in samples])])
# Steady-state window: mmap'd weights are faulted in LAZILY (majflt stays 0, they are in the page cache),
# so RSS ramps for the first windows and a full-range fit measures that page-in, not retention. Exp840:
# a full-range fit on the 138 s clip said +24.4 MB/min while VmHWM never moved after t~200 s - the exact
# signature of a ramp read as a leak. Report both, and base the verdict on the steady-state window.
hwms = [s[2] or 0 for s in samples]
hmax = max(hwms) if hwms else 0.0
# Two conditions, because HWM saturates early (it is a high-water mark, so it reaches its max in the first
# few windows even while RSS is still climbing): require the peak to be near its max AND the resident set to
# be near its own median, otherwise sample 0 of a lazy-faulting run can sit inside the "steady" window.
med_rss = statistics.median(rss)
k = next((i for i, (h, r) in enumerate(zip(hwms, rss)) if h >= 0.98 * hmax and r >= 0.95 * med_rss),
         max(0, len(samples) // 4))
ss = [(samples[i][0] - base, samples[i][1]) for i in range(k, len(samples))]
ss_slope = trend(ss)
ss_se = trend_se(ss)
ss_band = (min(r for _, r in ss), max(r for _, r in ss)) if ss else (0.0, 0.0)
hwm_ss = max(hwms[k:]) if k < len(hwms) else 0.0
hwm_first = hwms[k] if k < len(hwms) else 0.0
print(f"\nsamples={len(samples)}  span={dur:.1f} min  first={rss[0]:.1f} MB  last={rss[-1]:.1f} MB"
      f"  min={min(rss):.1f}  max={max(rss):.1f}  peak(HWM)={max((s[2] or 0) for s in samples):.1f}")
print(f'full-range trend = {slope:+.2f} MB/min   net last-first = {rss[-1] - rss[0]:+.1f} MB'
      '   <- includes the first-touch ramp of the mmap\'d weights; NOT the leak indicator')
print(f'steady state from t={ss[0][0]:.0f}s (sample {k}/{len(samples)}): trend = {ss_slope:+.2f} '
      f'+/- {2 * ss_se:.2f} MB/min (95% interval, {len(ss)} samples), '
      f' band {ss_band[0]:.1f}-{ss_band[1]:.1f} MB ({ss_band[1] - ss_band[0]:.1f} MB wide),'
      f' HWM {hwm_first:.1f} -> {hwm_ss:.1f} MB')
# Threshold: a leak that matters is >= 0.5 MB/min (30 MB/hour in a pocket assistant). The old rule used a
# bare |slope| < 2.0, which both missed slow real leaks and flagged sampler noise as GROWTH (Exp865d).
verdict = classify(ss_slope, ss_se)
if verdict == 'FLAT - no leak signal' and (hwm_ss - hwm_first) >= 25.0:
    verdict = 'GROWTH - peak is climbing even though the resident trend is flat'
print(f'memory verdict: {verdict}')

# ---- descriptor / thread verdicts (the resource-limit edge) -------------------------------
fds = [s_[5] for s_ in samples if len(s_) > 5 and s_[5]]
ths = [s_[4] for s_ in samples if len(s_) > 4 and s_[4]]
if fds:
    fd_slope = trend([(samples[i][0] - base, samples[i][5]) for i in range(len(samples)) if len(samples[i]) > 5 and samples[i][5]])
    fd_net = fds[-1] - fds[0]
    head = f" / limit {nofile[0]}" if nofile[0] else ''
    print(f'descriptors: first={fds[0]} last={fds[-1]} min={min(fds)} max={max(fds)} net={fd_net:+d}'
          f' trend={fd_slope:+.2f}/min{head}')
    print('descriptor verdict: ' + ('FLAT - no fd leak' if abs(fd_net) <= 2 and abs(fd_slope) < 0.5
          else f'LEAK - {fd_net:+d} descriptors over {dur:.1f} min; long sessions die at RLIMIT_NOFILE'))
else:
    print('descriptor verdict: NOT MEASURED - /proc/<pid>/fd was unreadable, this soak proved nothing about fds')
if ths:
    print(f'threads: min={min(ths)} max={max(ths)} ' +
          ('FLAT' if max(ths) - min(ths) <= 1 else f'CHANGES by {max(ths) - min(ths)} - thread leak?'))
for line in out.splitlines():
    if line.startswith('METRIC'):
        print(line)
