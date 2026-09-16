#!/usr/bin/env python3
"""Output-equivalence map: paired token distance of every archived 40-utt gate set from a reference.

WHY (Exp810). The loop has 66 usable archived hyp-* transcript sets. A claim like "config X is
output-equivalent to what ships" must be a PAIRED token distance (Exp655 rule), because the gate's WER
resolution is ~2 tokens of 731 (Exp657) - WER comparisons are blind to exactly the differences that
decide keep/discard on accuracy-neutral changes. All the data already exists on disk, so the whole map
costs zero device time and is immune to thermal drift.

FAST PATH MATTERS. compare_arms.compare_gate() runs a 20 000-sample bootstrap, so calling it 66 times
took >300 s and timed out. Here we reuse its OWN tokeniser and stream builder (gate_streams) and do one
LCS diff per set - ~30 s for the whole map. Never reimplement the tokeniser: an ad-hoc version gave
distances 2-4x too large and disagreed with the ledger (caught before it was written down).

Read this with two caveats:
  * sets recorded before Exp617 have no run-info.log stamp ('unstamped' below) - the distance is real,
    the recorded configuration is weaker evidence (Exp657).
  * the reference is whatever set you pass; distances are not comparable across different references.
    The shipped reference today is hyp-gate791 (v4.4, 40/40 identical to hyp-norm784 and hyp-gatem2).

Usage:  python3 .auto/equiv_map.py [REF_DIR] [-o out.md]     # REF_DIR default: eval-librispeech/hyp-gate791
"""
import os, sys, glob, json, difflib, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'eval-librispeech')
sys.path.insert(0, HERE)
from compare_arms import gate_streams                       # the loop's single tokeniser implementation


def stream(hyp_dir, refs, names):
    return [t['t'] for t in gate_streams(hyp_dir, refs, names)[1]]


def dist(A, B):
    """Raw token-presence difference: (b, c) = tokens present only in B / only in A.
    NOT the same quantity as compare_arms' McNemar b/c, which are CORRECTNESS discordants (a token that
    A got right and B got wrong). A config can move tokens without changing accuracy, so use this column
    to ask 'same system?', and compare_arms --gate to ask 'better?'. Exp810: conflating the two would
    make this map look like it contradicts the ledger when it is answering a different question."""
    eq = sum(bl.size for bl in difflib.SequenceMatcher(None, A, B, autojunk=False).get_matching_blocks())
    return len(B) - eq, len(A) - eq


def main():
    args = [a for a in sys.argv[1:]]
    out = None
    if '-o' in args:
        i = args.index('-o'); out = args[i + 1]; del args[i:i + 2]
    ref = args[0] if args else os.path.join(EVAL, 'hyp-gate791')
    refs = json.load(open(os.path.join(EVAL, 'refs.json'), encoding='utf-8'))
    names = sorted(refs)
    A = stream(ref, refs, names)
    stampf = os.path.join(ref, 'run-info.log')
    stamp = open(stampf).read().strip().splitlines()[0] if os.path.exists(stampf) else 'UNSTAMPED'

    rows = []
    for d in sorted(glob.glob(os.path.join(EVAL, 'hyp-*'))):
        name = os.path.basename(d)
        if d == ref or not os.path.isdir(d):
            continue
        if len([f for f in glob.glob(d + '/*.txt')]) != 40:
            continue                                       # partial gate runs cannot be paired
        b, c = dist(A, stream(d, refs, names))
        rows.append((name, b + c, b, c, os.path.exists(os.path.join(d, 'run-info.log'))))
    rows.sort(key=lambda r: (r[1], r[0]))
    same = [r[0] for r in rows if r[1] == 0]

    L = [f"# output-equivalence map vs {os.path.basename(ref)}",
         f"generated {datetime.date.today()} by .auto/equiv_map.py - paired token distance, tool tokeniser",
         f"reference: {len(A)} tokens / 40 utterances; stamp: `{stamp}`",
         "",
         f"**{len(same)} of {len(rows)} archived sets are byte-identical to the reference:** "
         f"{', '.join(same)}",
         "",
         "Everything else differs by >=1 token. These are RAW TOKEN DIFFERENCES (presence), not McNemar",
         "correctness discordants - for accuracy claims use `compare_arms.py --gate A B refs.json`. Per",
         "Exp657's resolution limit, 4-8 differing tokens is two to four words out of 731 and is NOT",
         "evidence of an accuracy difference: read this table as 'same system?', not 'which is better'.",
         "Distances are relative to THIS reference only; re-run with a different REF_DIR to re-anchor.", ""]
    L.append("| set | tokens differ | only in set | only in ref | provenance |")
    L.append("|---|---|---|---|---|")
    for n, x, b, c, st in rows:
        if x:
            L.append(f"| {n} | {x} | {b} | {c} | {'stamped' if st else 'unstamped (pre-Exp617)'} |")
    txt = "\n".join(L) + "\n"
    if out:
        open(out, 'w').write(txt)
        print(f"wrote {out}")
    print(txt)


if __name__ == '__main__':
    main()
