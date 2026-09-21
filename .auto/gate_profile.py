#!/usr/bin/env python3
"""Paired per-clip speed profile of two gate runs (Exp1038).

The gate has always been the accuracy instrument. It is also a SPEED instrument: eval40.sh writes a per-clip
rtf column (hyp-<tag>/rtf.log), so two archived gate rotations can be compared clip-by-clip with NO extra
device time. What that comparison is good for is narrower than it looks, and this tool prints the caveat next
to the numbers:

  * WHAT IT ANSWERS  - is a difference UNIFORM across clips, or clip-selective? (a real speed change often hits
    only short clips, or only the long ones; a session state shift hits every clip equally)
  * WHAT IT DOES NOT ANSWER - "did the mean move?". The session/state term is COMMON to all 40 clips, so it
    cancels in no way; the error bar for one run per arm is the session spread (measured sd 2.26 % over four
    same-config armed gates: 1.3808 / 1.3180 / 1.3789 / 1.3795), NOT the 40-clip standard error (~0.5 %).
    A large t here means "uniform", not "real" - the same trap as pooling gate means across eras (Exp1027).

Usage: gate_profile.py <hyp-dir-A> <hyp-dir-B>     (paths or bare tags; needs rtf.log in each)
       gate_profile.py --selftest
"""
import math
import os
import re
import statistics as st
import sys


def load(path):
    """rtf.log rows are `<i> <clip> rtf=<x> tok=<n> t=<ts>`; return {clip: (rtf, tokens)}."""
    d = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        for ln in f:
            m = re.match(r'\s*\d+\s+(\S+)\s+rtf=([0-9.]+)\s+tok=(\d+)', ln)
            if m:
                d[m.group(1)] = (float(m.group(2)), int(m.group(3)))
    if not d:
        sys.exit(f'ERROR: no rtf rows in {path} - a gate that never ran cannot be profiled')
    return d


def sign_p(b, c):
    """Exact two-sided sign/McNemar p = 2 * P(X <= min(b,c)), X ~ Bin(n, 0.5), capped at 1.

    Exp1038: the first version of this counted `range(min, 21)`, which printed p=1.1 for b=1,c=39 - an
    impossible value, and the fourth statistic-implementation bug in this loop (Exp655 used |b-c| instead of
    |b-n/2|). The self-tests at the bottom are what make the number trustworthy.
    """
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def profile(a, b):
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        sys.exit('ERROR: the two runs share fewer than 2 clips - nothing to pair')
    d = [a[k][0] / b[k][0] - 1 for k in keys]
    mean, sd = st.mean(d), st.stdev(d)
    se = sd / math.sqrt(len(d))
    faster = sum(1 for x in d if x < 0)
    # length halves by token count (a duration proxy) - where a clip-selective effect would show up
    tok = sorted(keys, key=lambda k: a[k][1])
    half = len(tok) // 2
    lo = [a[k][0] / b[k][0] - 1 for k in tok[:half]]
    hi = [a[k][0] / b[k][0] - 1 for k in tok[half:]]
    return dict(n=len(keys), mean=mean, sd=sd, se=se, faster=faster,
                p=sign_p(faster, len(keys) - faster), lo=st.mean(lo), hi=st.mean(hi),
                mA=st.mean([a[k][0] for k in keys]), mB=st.mean([b[k][0] for k in keys]))


def resolve(arg):
    for p in (arg, f'../eval-librispeech/hyp-{arg}'):
        f = os.path.join(p, 'rtf.log')
        if os.path.exists(f):
            return f
    sys.exit(f'ERROR: {arg} has no rtf.log (pass a hyp dir or a bare tag)')


def selftest():
    ok = True
    for b, c, want in [(0, 40, 'lt1e-11'), (1, 39, 'lt1e-9'), (20, 20, 'eq1'), (6, 34, 'lt1e-4'), (7, 6, 'eq1')]:
        p = sign_p(b, c)
        good = (p < 1e-11 if want == 'lt1e-11' else p < 1e-9 if want == 'lt1e-9'
                else p < 1e-4 if want == 'lt1e-4' else abs(p - 1.0) < 1e-12)
        ok &= good
        print(f'  sign_p({b},{c}) = {p:.3g}  {"OK" if good else "FAIL"}')
    if not 1.0 <= 1.0:
        ok = False
    # a p-value above 1 is impossible by construction; assert it can never be reported
    bad = max(sign_p(x, 40 - x) for x in range(41))
    print(f'  max p over all splits of n=40: {bad:.6f}  {"OK" if bad <= 1.0 else "FAIL"}')
    ok &= bad <= 1.0
    # synthetic: a uniform 5% shift must read mean +5%, sd ~0, all 40 clips one-sided
    a = {f'c{i}': (1.05, 100) for i in range(40)}
    b = {f'c{i}': (1.00, 100) for i in range(40)}
    r = profile(a, b)
    good = abs(r['mean'] - 0.05) < 1e-9 and r['faster'] == 0 and r['p'] < 1e-11
    print(f"  uniform +5% shift -> mean {r['mean']*100:+.2f}%, {r['faster']}/40 faster, p={r['p']:.2g}  {'OK' if good else 'FAIL'}")
    ok &= good
    # synthetic: a short-clip-only effect must NOT look uniform (halves must differ)
    a = {f'c{i}': (1.10 if i < 20 else 1.00, 50 + i) for i in range(40)}
    b = {f'c{i}': (1.00, 50 + i) for i in range(40)}
    r = profile(a, b)
    good = r['lo'] > 0.04 and abs(r['hi']) < 1e-9
    print(f"  clip-selective check -> short half {r['lo']*100:+.2f}%, long half {r['hi']*100:+.2f}%  {'OK' if good else 'FAIL'}")
    ok &= good
    print('selftest:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args or any(a.startswith('-') and a != '--selftest' for a in args):
        if args and args[0] == '--selftest':
            sys.exit(selftest())
        print(__doc__)
        sys.exit(2 if args else 0)
    if args[0] == '--selftest':
        sys.exit(selftest())
    if len(args) != 2:
        print('usage: gate_profile.py <hyp-A> <hyp-B>', file=sys.stderr)
        sys.exit(2)
    A, B = load(resolve(args[0])), load(resolve(args[1]))
    r = profile(A, B)
    print(f'{args[0]} vs {args[1]}: n={r["n"]} clips | run means {r["mA"]:.4f} vs {r["mB"]:.4f}')
    print(f'  paired per-clip: mean {r["mean"]*100:+.2f}%  sd {r["sd"]*100:.2f}%  se(mean) {r["se"]*100:.2f}%')
    print(f'  {r["faster"]}/{r["n"]} clips faster  (corrected sign test p={r["p"]:.3g})')
    print(f'  by duration half: short {r["lo"]*100:+.2f}%  long {r["hi"]*100:+.2f}%   <- a CLIP-SELECTIVE test')
    print('  CAVEAT: se(mean) is the clip-level error only. The session/state term is common to all clips, so a')
    print('  one-run-per-arm comparison carries the SESSION spread (~2.3% sd over same-config gates), not this se.')
