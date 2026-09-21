#!/usr/bin/env python3
"""Delivered 2.4 GHz share from two cpufreq stats/time_in_state dumps (Exp942, extracted Exp947).

Why this exists as a file: measure.sh and run_rtf_multi.sh both need it, and duplicating the parse in
two harness scripts is how the loop's two copies of a rule drift apart (the Exp869/Exp816 class). The
unit is a 10 ms jiffy (calibrated: 100.3 units per second of wall, Exp942), so the share is
unit-independent; the 2.4 GHz share is the state indicator (96-98 % boost, 0 % settled, in between =
a half-flipped run) and it is REQUEST-independent, which is why it sees what scaling_cur_freq cannot.

Usage:  .auto/deliv_share.py <before_dump> <after_dump>
            -> "<deliv2400_pct> <mean_mhz> <ge2000_pct>", or "-1 -1 -1" if unreadable
        .auto/deliv_share.py --selftest                     -> no device needed

Exp987: the ONE implementation now returns all three witnesses (the 2.4 GHz share, the MEAN delivered
MHz, and the share at >= 2.0 GHz) because Exp974 showed the 2.4 share ALONE cannot tell "capped at
1.3 GHz" from "governed at 2.15 GHz" - the fastest long cell of that day read deliv2400 = 0 %. Before
this, measure.sh and run_rtf_multi.sh each had their own parse of the same histogram, which is exactly
the drift this file exists to prevent (Exp869/Exp816).
"""
import os
import sys
import tempfile


def _delta(before, after):
    """time_in_state deltas (keys are kHz), keeping only steps that gained time. None if unreadable."""
    def rd(p):
        return {int(l.split()[0]): int(l.split()[1]) for l in open(p) if len(l.split()) == 2}
    a, b = rd(before), rd(after)
    d = {k: b.get(k, 0) - a.get(k, 0) for k in set(a) | set(b)}
    return {k: v for k, v in d.items() if v > 0}


def share(before, after):
    d = _delta(before, after)
    tot = sum(d.values())
    return round(100 * d.get(2400000, 0) / tot) if tot > 0 else -1


def mean_mhz(before, after):
    """MEAN delivered big-core MHz - the one number that names the P-state. Units: time_in_state keys
    are kHz, so divide by 1000 (Exp975 shipped this after the first cut emitted kHz under an `mhz`
    header; the value was read back before the commit, which is the only reason it was caught)."""
    d = _delta(before, after)
    tot = sum(d.values())
    return round(sum(f * v for f, v in d.items()) / tot / 1000, 1) if tot > 0 else -1


def ge2000(before, after):
    d = _delta(before, after)
    tot = sum(d.values())
    return round(100 * sum(v for f, v in d.items() if f >= 2000000) / tot) if tot > 0 else -1


def _selftest():
    """Prove the parse fires on both states and refuses a no-change dump (Exp660 rule)."""
    d = tempfile.mkdtemp()
    a = os.path.join(d, 'a'); b = os.path.join(d, 'b'); c = os.path.join(d, 'c')
    open(a, 'w').write("2400000 100\n2000000 50\n1430000 7\n")
    open(b, 'w').write("2400000 196\n2000000 51\n1430000 7\n")      # +96 at 2.4, +1 at 2.0 -> 99 %
    open(c, 'w').write("2400000 100\n2000000 150\n1430000 7\n")     # +100 at 2.0 -> 0 %
    ok = True
    s1, s2 = share(a, b), share(a, c)
    if s1 != 99:
        print(f"FAIL: boost dump gave {s1}, expected 99"); ok = False
    if s2 != 0:
        print(f"FAIL: settled dump gave {s2}, expected 0"); ok = False
    if share(a, a) != -1:
        print("FAIL: an unchanged dump must give -1 (no counted time), not a fake share"); ok = False
    # Exp987: the mean and the >=2.0 GHz share are the numbers that disambiguate a 2.15 GHz run; check
    # them on the same planted dumps rather than trusting the new code by inspection (Exp660).
    m_boost = mean_mhz(a, b)      # +96 jiffies at 2400000 kHz, +1 at 2000000 kHz -> ~2396 MHz
    m_21 = mean_mhz(a, c)         # +100 jiffies at 2000000 kHz                  -> 2000 MHz
    g_21 = ge2000(a, c)
    if not (2390 <= m_boost <= 2400):
        print(f"FAIL: boost mean = {m_boost} MHz, expected ~2396"); ok = False
    if m_21 != 2000.0:
        print(f"FAIL: 2.15 GHz-style dump mean = {m_21}, expected 2000.0"); ok = False
    if g_21 != 100:
        print(f"FAIL: >=2.0 GHz share = {g_21}, expected 100"); ok = False
    if mean_mhz(a, a) != -1 or ge2000(a, a) != -1:
        print("FAIL: an unchanged dump must give -1 for every witness"); ok = False
    if not ok:
        print("deliv_share --selftest: FAILED")
        return 1
    print("deliv_share --selftest: OK (99 % boost / 0 % settled / -1 on no change; mean 2396 vs 2000 MHz; "
          "ge2000 100 % on the 2.0 GHz dump)")
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        sys.exit(_selftest())
    if len(sys.argv) != 3:
        sys.exit("usage: deliv_share.py <before_dump> <after_dump> | --selftest")
    print(f"{share(sys.argv[1], sys.argv[2])} {mean_mhz(sys.argv[1], sys.argv[2])} "
          f"{ge2000(sys.argv[1], sys.argv[2])}")
