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
PROC = 'asr_streaming'
WARM_MB = float(os.environ.get('SOAK_WARM_MB', 200))   # ignore anything not yet running a model
samples = []          # (t_seconds, rss_mb, hwm_mb, state)
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
            pids = adb(['shell', 'pidof', PROC]).stdout.replace('\r', '').split()
            best = None
            for p in pids:
                # cat, not grep: adb JOINS its argv into one device-shell command string, so an
                # unquoted alternation like ^VmRSS|^VmHWM is parsed as shell PIPES by the device and
                # the probe silently returns nothing (this cost the first two soak attempts).
                r = adb(['shell', 'cat', f'/proc/{p}/status'])
                rss = hwm = vsz = None
                for line in r.stdout.replace('\r', '').splitlines():
                    if line.startswith('VmRSS:'):
                        rss = int(line.split()[1]) / 1024.0
                    elif line.startswith('VmHWM:'):
                        hwm = int(line.split()[1]) / 1024.0
                    elif line.startswith('VmSize:'):
                        vsz = int(line.split()[1]) / 1024.0
                if rss is None:
                    continue
                if best is None or rss > best[1]:
                    best = (p, rss, hwm, vsz)
            if best is None:
                pass                                  # nothing resident yet - keep sampling
            elif best[1] < WARM_MB:
                stale_seen[0] += 1                     # loading, or an orphan: not a data point
            else:
                samples.append((time.time(), best[1], best[2], best[3]))
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
print(f"{'t(s)':>6} {'VmRSS':>9} {'VmHWM':>9} {'VmSize':>9}")
for t, r, h, v in samples:
    print(f'{t - base:6.0f} {r:9.1f} {(h or 0):9.1f} {(v or 0):9.1f}')

rss = [s[1] for s in samples]
dur = (samples[-1][0] - samples[0][0]) / 60.0
xs = [(s[0] - base) / 60.0 for s in samples]
if len(xs) > 2 and statistics.pstdev(xs) > 0:
    mx, my = statistics.mean(xs), statistics.mean(rss)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, rss)) / sum((x - mx) ** 2 for x in xs)
else:
    slope = float('nan')
print(f"\nsamples={len(samples)}  span={dur:.1f} min  first={rss[0]:.1f} MB  last={rss[-1]:.1f} MB"
      f"  min={min(rss):.1f}  max={max(rss):.1f}  peak(HWM)={max((s[2] or 0) for s in samples):.1f}")
print(f'growth trend = {slope:+.2f} MB/min   net last-first = {rss[-1] - rss[0]:+.1f} MB')
for line in out.splitlines():
    if line.startswith('METRIC'):
        print(line)
