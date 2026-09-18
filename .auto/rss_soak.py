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
import statistics
import subprocess
import sys
import threading
import time

args = sys.argv[1:]
interval = 5.0
if args and args[0] == '--interval':
    interval = float(args[1]); args = args[2:]
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
th.start()
out = subprocess.run(cmd, capture_output=True, text=True).stdout
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
ss_band = (min(r for _, r in ss), max(r for _, r in ss)) if ss else (0.0, 0.0)
hwm_ss = max(hwms[k:]) if k < len(hwms) else 0.0
hwm_first = hwms[k] if k < len(hwms) else 0.0
print(f"\nsamples={len(samples)}  span={dur:.1f} min  first={rss[0]:.1f} MB  last={rss[-1]:.1f} MB"
      f"  min={min(rss):.1f}  max={max(rss):.1f}  peak(HWM)={max((s[2] or 0) for s in samples):.1f}")
print(f'full-range trend = {slope:+.2f} MB/min   net last-first = {rss[-1] - rss[0]:+.1f} MB'
      '   <- includes the first-touch ramp of the mmap\'d weights; NOT the leak indicator')
print(f'steady state from t={ss[0][0]:.0f}s (sample {k}/{len(samples)}): trend = {ss_slope:+.2f} MB/min,'
      f' band {ss_band[0]:.1f}-{ss_band[1]:.1f} MB ({ss_band[1] - ss_band[0]:.1f} MB wide),'
      f' HWM {hwm_first:.1f} -> {hwm_ss:.1f} MB')
verdict = 'FLAT - no leak signal' if abs(ss_slope) < 2.0 and (hwm_ss - hwm_first) < 25.0 else 'GROWTH - investigate'
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
