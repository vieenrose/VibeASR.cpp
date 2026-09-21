#!/usr/bin/env python3
"""Session-level uptime drift: the only form of the drift claim this loop can support.

Founding problem (Exp1047): Exp1002 published a two-parameter law ("1.1939 at ~16 h, +0.041 per 10 h").
Re-tested as a PREDICTOR against 35 clean armed sessions it had never seen, that law is biased **-3.03 %**
(sd 1.78 %, worst -5.58 %), so quoting it to normalize a number across epochs is wrong even though its
direction is right. What survives is the SLOPE, fitted here at session level: +0.26 +/- 0.10 %/h.

Method rules encoded here (each one is a bug this loop already paid for):
  * Rows are grouped into SESSIONS (>300 s gap = new session). Exp1038: the state term is common to every
    clip in a session, so per-row error bars understate it. Session means are the unit of analysis.
  * A session counts as CLEAN only if its MEDIAN cpu7_deliv_mhz is at/above the knee (Exp1025: the witness
    is a step, not a line). Sub-floor sessions are excluded and reported, never averaged in (Exp995's
    "retry, never average").
  * Only protocol-clip rows are used (tokens == 39); boards and probes emit other counts and their rtf is
    not the metric.
  * The INTERCEPT is reported as not identifiable across eras: Exp1003 showed the arm recipe is collinear
    with uptime, so a level correction across epochs cannot be fitted from this data. Hence this tool prints
    a slope and a bias TEST, never a "corrected rtf".

Usage: uptime_law.py [--tsv PATH] [--min-uptime H] [--selftest]
"""
import csv, os, statistics as st, sys

TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'device_state.tsv')
SESSION_GAP_S = 300.0
KNEE_MHZ = 2000.0
PROTOCOL_TOKENS = '39'


def sessions(rows, min_uptime_h=31.0, tokens=PROTOCOL_TOKENS):
    """Group telemetry rows into sessions. Returns (clean, subfloor, skipped) where each session is a dict."""
    out, cur, skipped = [], None, 0
    for r in rows:
        try:
            u = float(r['uptime_s']); rt = float(r['rtf']); ts = float(r['ts'])
        except (KeyError, ValueError, TypeError):
            skipped += 1
            continue
        if u < min_uptime_h * 3600 or str(r.get('tokens', '')).strip() != tokens:
            skipped += 1
            continue
        if cur is None or ts - cur['t1'] > SESSION_GAP_S:
            cur = {'v': [], 'm': [], 't1': ts, 'u': u}
            out.append(cur)
        cur['v'].append(rt); cur['t1'] = ts; cur['u'] = u
        try:
            cur['m'].append(float(r['cpu7_deliv_mhz']))
        except (KeyError, ValueError, TypeError):
            pass
    clean = [s for s in out if s['m'] and st.median(s['m']) >= KNEE_MHZ]
    sub = [s for s in out if s['m'] and st.median(s['m']) < KNEE_MHZ]
    nowit = [s for s in out if not s['m']]
    return clean, sub, nowit, skipped


def fit(pts):
    """OLS of session-mean rtf on uptime_h. Returns dict or None if underdetermined."""
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    n = len(pts); mx = st.mean(xs); my = st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in pts]
    s2 = sum(r * r for r in res) / (n - 2)
    # se_b = sqrt(s2 / Sxx). My first version wrote `**5` instead of `**0.5`, which printed se = 0.000 and
    # t = 1.9e27 on the very first real run - absurd on its face, which is why selftest case (6) now refuses
    # a collapsed standard error instead of letting it reach a ledger.
    se_b = (s2 / sxx) ** 0.5
    return {'n': n, 'a': a, 'b': b, 'se_b': se_b, 't': b / se_b if se_b else float('nan'),
            'slope_pct_h': 100 * b / my, 'resid_sd': st.stdev(res), 'mean': my,
            'span_h': (min(xs), max(xs)), 'resid': res}


def bias_vs_law(pts, base=1.1939, per10h=0.041, at_h=16.0):
    """Test a published level+slope law as a predictor. Returns (mean bias %, sd, worst, best)."""
    d = [100 * (y / (base + per10h * (x - at_h) / 10) - 1) for x, y in pts]
    return st.mean(d), (st.stdev(d) if len(d) > 1 else float('nan')), min(d), max(d)


def _selftest():
    ok = True
    # (1) IDENTITY CONTROL: a constant series must give slope 0 (a fitter that returns nonzero here is lying).
    flat = [(h + 31, 1.20) for h in range(0, 12)]
    f0 = fit(flat)
    if f0 is None or abs(f0['slope_pct_h']) > 1e-9:
        print(f"FAIL: a flat series must give slope 0, got {f0 and f0['slope_pct_h']}"); ok = False
    # (2) PLANTED SLOPE: +0.5 %/h must be recovered within noise.
    pts = [(31 + h, 1.20 * (1 + 0.005 * h)) for h in range(0, 13)]
    f1 = fit(pts)
    if f1 is None or not (0.45 < f1['slope_pct_h'] < 0.55):
        print(f"FAIL: planted +0.5 %/h read {f1 and f1['slope_pct_h']:.3f}"); ok = False
    # (3) Session grouping: a >300 s gap splits, a burst inside 300 s stays one session.
    def row(u, rt, ts, mhz='2300.0', tok='39'):
        return {'uptime_s': str(u), 'rtf': str(rt), 'ts': str(ts), 'cpu7_deliv_mhz': mhz, 'tokens': tok}
    rws = [row(32 * 3600, 1.20, 1000), row(32 * 3600, 1.21, 1030),      # same session
           row(33 * 3600, 1.40, 2000, mhz='1800.0'),                     # sub-floor -> excluded
           row(40 * 3600, 1.25, 99999),                                  # new session (gap)
           row(41 * 3600, 1.25, 100030, tok='446')]                      # non-protocol -> skipped
    cl, sub, nowit, sk = sessions(rws)
    if len(cl) != 2:
        print(f"FAIL: expected 2 clean sessions, got {len(cl)}"); ok = False
    if len(sub) != 1:
        print(f"FAIL: the 1800 MHz session must land in the sub-floor list, got {len(sub)}"); ok = False
    if sk < 1:
        print("FAIL: a non-protocol row must be counted as skipped"); ok = False
    # (4) The law test must fire on a law that is wrong by construction.
    b = bias_vs_law([(31 + h, 1.20 * (1 + 0.005 * h)) for h in range(13)])
    if not (b[0] < -1.0 or b[0] > 1.0):
        print(f"FAIL: bias_vs_law against a mismatched law read {b[0]:.2f}% - it should be far from 0"); ok = False
    # (5) Underdetermined input must return None, not a confident 0.
    if fit([(31.0, 1.20), (31.0, 1.21)]) is not None:
        print("FAIL: a zero-variance-in-x fit must refuse (None), not invent a slope"); ok = False
    # (6) A noisy series must yield a NONZERO, sane standard error (this is the case that would have caught the
    # `**5` exponent typo: se collapsed to 0 and t printed as 1.9e27).
    noisy = [(31 + h, 1.20 * (1 + 0.005 * h) * (1 + (0.004 if h % 2 else -0.004))) for h in range(14)]
    f6 = fit(noisy)
    if f6 is None or not (f6['se_b'] > 0) or not (0.5 < f6['t'] < 1e3):
        print(f"FAIL: noisy fit gave se {f6 and f6['se_b']} t {f6 and f6['t']} - a collapsed or absurd standard "
              'error means the variance code is wrong'); ok = False
    if not ok:
        print("uptime_law --selftest: FAILED"); return 1
    print("uptime_law --selftest: OK (flat -> slope 0; planted +0.5 %/h recovered; sessions split at 300 s; "
          "sub-floor session excluded and reported; mismatched law flagged; degenerate x refused; "
          "noisy fit has a sane nonzero se)")
    return 0


if __name__ == '__main__':
    args = sys.argv[1:]
    for a in args:
        if a.startswith('-') and a not in ('--selftest', '--tsv', '--min-uptime'):
            sys.exit(f"ERROR: unknown argument {a!r} - usage: uptime_law.py [--tsv PATH] [--min-uptime H] [--selftest]")
    if '--selftest' in args:
        sys.exit(_selftest())
    tsv = TSV
    if '--tsv' in args:
        tsv = args[args.index('--tsv') + 1]
    minu = 31.0
    if '--min-uptime' in args:
        minu = float(args[args.index('--min-uptime') + 1])
    with open(tsv, encoding='utf-8', errors='ignore') as fh:
        rows = list(csv.DictReader(fh, delimiter='\t'))
    clean, sub, nowit, skipped = sessions(rows, min_uptime_h=minu)
    print(f"rows {len(rows)} | used {sum(len(s['v']) for s in clean)} rows in {len(clean)} clean sessions "
          f"| excluded sub-floor {len(sub)} | no-witness {len(nowit)} | not-applicable {skipped}")
    if sub:
        print(f"  sub-floor session mean {st.mean([st.mean(s['v']) for s in sub]):.4f} "
              f"(excluded per 'retry, never average')")
    pts = sorted((s['u'] / 3600.0, st.mean(s['v'])) for s in clean)
    f_ = fit(pts)
    if f_ is None:
        sys.exit("NOT ENOUGH DATA for a slope (need >=3 sessions with distinct uptimes) - printed nothing to quote")
    print(f"uptime span {f_['span_h'][0]:.1f}-{f_['span_h'][1]:.1f} h, {f_['n']} sessions")
    print(f"  slope {f_['slope_pct_h']:+.3f} +/- {100 * f_['se_b'] / f_['mean']:.3f} %/h   t={f_['t']:.2f}   "
          f"(total {f_['slope_pct_h'] * (f_['span_h'][1] - f_['span_h'][0]):+.2f}% over the span)")
    print(f"  session residual sd {f_['resid_sd']:.4f} ({100 * f_['resid_sd'] / f_['mean']:.2f}%) "
          "- any curvature or epoch claim must beat this")
    half = len(pts) // 2
    e, l = st.mean(f_['resid'][:half]), st.mean(f_['resid'][half:])
    print(f"  curvature: early residual {e:+.4f} late {l:+.4f} -> "
          + ('CONCAVE, drift slowing' if e > 2 * abs(l) and e > f_['resid_sd'] / 4 else 'no resolved curvature'))
    mb = bias_vs_law(pts)
    verdict = 'REJECT as a predictor (quote the slope instead)' if abs(mb[0]) > 1.5 else 'usable'
    print(f"  Exp1002's law (1.1939 @16h, +0.041/10h) as a predictor: mean bias {mb[0]:+.2f}% sd {mb[1]:.2f}% "
          f"worst {mb[2]:+.2f}%/{mb[3]:+.2f}% -> {verdict}")
    print("  NOTE: the intercept is NOT identifiable across eras (Exp1003: the arm recipe is collinear with "
          "uptime), so do not apply a level correction - quote numbers with a DERIVED uptime instead.")
