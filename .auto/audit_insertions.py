#!/usr/bin/env python3
"""Hallucination / runaway audit on existing artifacts (Exp658).

WER treats a substitution, a deletion and an insertion as one error each, but they are not
the same product failure. Insertions are words the speaker never said - hallucination - and
they are the ones that destroy user trust, plus they can compound into a repetition loop.
Deletions just lose content. So the gate's headline says nothing about this failure mode,
and the held-out sets showed an insertion RATE ~10x the clean gate.

Per set we report:
  * insertions per 1000 reference tokens (rate, so sets of different length compare),
  * how concentrated they are (how many turns hold all of them, and the longest spurious
    run in one place),
  * a runaway check: any repeated 8-gram inside a single turn, and the longest repeated run.

Everything is computed from artifacts already on disk (hyps + manifests), so this cannot
become a benchmark to optimize against.

Usage: .auto/audit_insertions.py <hyp> <manifest> [label]   (repeatable)
"""
import json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_stream import ref_stream, hyp_stream, align   # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def audit(hyp_path, manifest, label):
    man = json.load(open(manifest, encoding='utf-8'))
    ref, hyp = ref_stream(man), hyp_stream(hyp_path)
    ops, _ = align(ref, hyp)

    # Bucket insertions by the turn of the last reference token seen so far.
    turn_of_ref = [t['turn'] for t in ref]
    cur_turn = None
    ins_turn = Counter()
    runs = []
    run = 0
    for op, i, j in ops:
        if op == 'I':
            run += 1
            ins_turn[cur_turn] += 1
        else:
            if run:
                runs.append(run)
                run = 0
            if i is not None:
                cur_turn = turn_of_ref[i]
    if run:
        runs.append(run)
    n_ins = sum(ins_turn.values())
    per_turn = Counter()
    for t in man['table']:
        per_turn[t['turn']] = ins_turn.get(t['turn'], 0)
    hot = [(t, c) for t, c in per_turn.items() if c > 0]
    hot.sort(key=lambda x: -x[1])

    # runaway / repetition check on the system output itself
    worst_rep, rep_example = 0, ''
    by_turn = {}
    for t in man['table']:
        by_turn[t['turn']] = t['text']
    hyp_text_by_turn = {}
    for op, i, j in ops:                      # crude: use reference turn as the bucket
        turn = turn_of_ref[i] if i is not None else cur_turn
        hyp_text_by_turn.setdefault(turn, []).append(hyp[j]['t'] if j is not None else '')
    for turn, toks in hyp_text_by_turn.items():
        toks = [t for t in toks if t]
        if len(toks) < 8:
            continue
        grams = Counter(tuple(toks[k:k + 8]) for k in range(len(toks) - 7))
        rep = max(grams.values())
        if rep > worst_rep:
            worst_rep = rep
            rep_example = ' '.join(max(grams, key=lambda g: grams[g]))
    return {'label': label, 'ref_tokens': len(ref), 'hyp_tokens': len(hyp), 'ins': n_ins,
            'rate_per_1k': 1000.0 * n_ins / max(1, len(ref)), 'max_run': max(runs) if runs else 0,
            'turns_with_ins': len(hot), 'turns': len(man['table']), 'hot': hot[:4],
            'max_rep8': worst_rep, 'rep_example': rep_example}


if __name__ == '__main__':
    print(f"{'set':34s} {'ins':>4} {'per 1k ref':>10} {'max run':>7} {'turns w/ ins':>12} {'max rep-8gram':>13}")
    args = sys.argv[1:]
    for a in args:
        parts = a.split(':')
        hyp, man = (parts[0], parts[1]) if len(parts) > 2 else (parts[0], parts[1])
        label = parts[2] if len(parts) > 2 else os.path.basename(man)
        r = audit(hyp, man, label)
        print(f"{label:34s} {r['ins']:4d} {r['rate_per_1k']:10.1f} {r['max_run']:7d} "
              f"{r['turns_with_ins']:5d}/{r['turns']:<6d} {r['max_rep8']:13d}")
        if r['hot']:
            print(f"     concentrated in turns: {r['hot']}"
                  + (f" | longest spurious run {r['max_run']} words" if r['max_run'] > 2 else ""))
        if r['max_rep8'] > 1:
            print(f"     repeated 8-gram x{r['max_rep8']}: {r['rep_example'][:90]}")
