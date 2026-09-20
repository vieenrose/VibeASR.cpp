#!/usr/bin/env python3
"""Delivered 2.4 GHz share from two cpufreq stats/time_in_state dumps (Exp942, extracted Exp947).

Why this exists as a file: measure.sh and run_rtf_multi.sh both need it, and duplicating the parse in
two harness scripts is how the loop's two copies of a rule drift apart (the Exp869/Exp816 class). The
unit is a 10 ms jiffy (calibrated: 100.3 units per second of wall, Exp942), so the share is
unit-independent; the 2.4 GHz share is the state indicator (96-98 % boost, 0 % settled, in between =
a half-flipped run) and it is REQUEST-independent, which is why it sees what scaling_cur_freq cannot.

Usage:  .auto/deliv_share.py <before_dump> <after_dump>     -> integer percent, or -1 if unreadable
        .auto/deliv_share.py --selftest                     -> no device needed
"""
import os
import sys
import tempfile


def share(before, after):
    def rd(p):
        return {l.split()[0]: int(l.split()[1]) for l in open(p) if len(l.split()) == 2}
    a, b = rd(before), rd(after)
    d = {k: b.get(k, 0) - a.get(k, 0) for k in set(a) | set(b)}
    tot = sum(v for v in d.values() if v > 0)
    return round(100 * d.get('2400000', 0) / tot) if tot > 0 else -1


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
    if not ok:
        print("deliv_share --selftest: FAILED")
        return 1
    print("deliv_share --selftest: OK (99 % boost / 0 % settled / -1 on no change)")
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        sys.exit(_selftest())
    if len(sys.argv) != 3:
        sys.exit("usage: deliv_share.py <before_dump> <after_dump> | --selftest")
    print(share(sys.argv[1], sys.argv[2]))
