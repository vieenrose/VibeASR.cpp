#!/usr/bin/env python3
"""Paired comparison of two systems on the same reference (Exp655).

Unpaired CIs have been misleading this session: with ~500-700 tokens the absolute WER CI
is +/-2 pp, but two systems run on the SAME audio mostly err on the SAME tokens, so the
DIFFERENCE is far better determined than either rate. This is the analysis that should
have been done before the Exp652/653 flip-flop.

Per reference token we record, for each system, 0 = correct, 1 = wrong (substitution or
deletion from the WER alignment). Insertions carry no reference token, so they are added
to the totals as a separate, small term and reported - they cannot bias the paired test
but they do count toward WER.

Statistics:
  * McNemar's exact test on the discordant pairs (b = wrong-A/correct-B, c = correct-A/
    wrong-B) - the standard paired test for two classifiers on the same samples.
  * A paired bootstrap over reference tokens (resample token indices, recompute the WER
    difference), which does include the insertion term.

Usage: .auto/compare_arms.py <hypA> <hypB> <manifest> [label]
       .auto/compare_arms.py --selftest
"""
import json, math, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_stream import ref_stream, hyp_stream, align, CJK   # noqa: E402
from score_mixed import normalize, tokenize                    # noqa: E402
import re                                                      # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def per_token(hyp_path, ref):
    """0/1 correctness per reference token, plus the insertion count."""
    hyp = hyp_stream(hyp_path)
    ops, _ = align(ref, hyp)
    ok = [None] * len(ref)
    ins = 0
    for op, i, j in ops:
        if op == 'I':
            ins += 1
        elif i is not None:
            ok[i] = 0 if (op == 'S' and ref[i]['t'] == hyp[j]['t']) else 1
    for i in range(len(ref)):
        if ok[i] is None:
            ok[i] = 1
    return ok, ins


def mcnemar_exact(b, c):
    """Exact two-sided McNemar. With X ~ Bin(n=b+c, 0.5), the observed split is X=b, so
    extremeness is |X - n/2| >= |b - n/2| (note: |b - n/2| == |b - c| / 2 - using |b - c|
    instead, as an earlier revision did, halves the tail and overstates significance; the
    self-test's analytic b=0,c=3 -> p=0.25 case is what catches it)."""
    n = b + c
    if n == 0:
        return 1.0
    d = abs(b - n / 2.0)
    tail = sum(math.comb(n, k) for k in range(n + 1) if abs(k - n / 2.0) >= d - 1e-12)
    return min(1.0, tail / (2 ** n))


def compare(hA, hB, man, nboot=20000, seed=7):
    ref = ref_stream(man)
    a, ia = per_token(hA, ref)
    b, ib = per_token(hB, ref)
    n = len(ref)
    disc_a = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)   # A wrong, B right
    disc_b = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)   # A right, B wrong
    wa, wb = (sum(a) + ia) / n, (sum(b) + ib) / n
    p = mcnemar_exact(disc_a, disc_b)
    rng = random.Random(seed)
    diffs = []
    for _ in range(nboot):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = sum(a[i] for i in idx) + ia * len(idx) / n
        sb = sum(b[i] for i in idx) + ib * len(idx) / n
        diffs.append((sa - sb) / len(idx))
    diffs.sort()
    lo, hi = diffs[int(0.025 * nboot)], diffs[int(0.975 * nboot)]
    frac_below = sum(1 for d in diffs if d < 0) / nboot
    return {'n': n, 'wer_a': wa, 'wer_b': wb, 'delta': wa - wb, 'ci': (lo, hi),
            'disc_a': disc_a, 'disc_b': disc_b, 'p': p, 'ins_a': ia, 'ins_b': ib,
            'p_boot_gt0': 1.0 - frac_below}


def gate_streams(hyp_dir, refs, names):
    """Paired token streams for the 40-utt gate: one file per utterance under hyp_dir,
    references from refs.json. Utterances are a fixed, sorted list, so the two systems
    are compared token-for-token over the same reference (Exp655 rule: between-system
    claims need the paired test, not two independent WERs)."""
    ref, hyp = [], []
    for name in names:
        for tok in tokenize(normalize(refs[name])):
            ref.append({'t': tok, 'gold': name, 'lang': 'en', 'turn': 0, 'start_s': 0.0,
                        'voice': name, 'end_s': 0.0, 'dur_s': 0.0, 'sha': ''})
        p = os.path.join(hyp_dir, name + '.txt')
        body = open(p, encoding='utf-8').read() if os.path.exists(p) else ''
        for tok in tokenize(normalize(body)):
            hyp.append({'t': tok, 'spk': 'T0', 'win': 0})
    return ref, hyp


def compare_gate(hyp_dir_a, hyp_dir_b, refs_json, nboot=20000, seed=11):
    """Paired comparison of two gate hyp directories (same utterance set)."""
    refs = json.load(open(refs_json, encoding='utf-8'))
    names = sorted(x for x in refs if os.path.exists(os.path.join(hyp_dir_a, x + '.txt'))
                   and os.path.exists(os.path.join(hyp_dir_b, x + '.txt')))
    ref, hA = gate_streams(hyp_dir_a, refs, names)
    _r, hB = gate_streams(hyp_dir_b, refs, names)
    ok_a, ia = _mark(ref, hA)
    ok_b, ib = _mark(ref, hB)
    n = len(ref)
    b = sum(1 for x, y in zip(ok_a, ok_b) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(ok_a, ok_b) if x == 0 and y == 1)
    wa, wb = (sum(ok_a) + ia) / n, (sum(ok_b) + ib) / n
    rng = random.Random(seed)
    diffs = []
    for _ in range(nboot):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = sum(ok_a[i] for i in idx) + ia * len(idx) / n
        sb = sum(ok_b[i] for i in idx) + ib * len(idx) / n
        diffs.append((sa - sb) / len(idx))
    diffs.sort()
    return {'n': n, 'utts': len(names), 'wer_a': wa, 'wer_b': wb, 'delta': wa - wb,
            'ci': (diffs[int(0.025 * nboot)], diffs[int(0.975 * nboot)]),
            'disc_a': b, 'disc_b': c, 'p': mcnemar_exact(b, c), 'ins_a': ia, 'ins_b': ib}


def _mark(ref, hyp):
    ops, _ = align(ref, hyp)
    ok = [None] * len(ref)
    ins = 0
    for op, i, j in ops:
        if op == 'I':
            ins += 1
        elif i is not None:
            ok[i] = 0 if (op == 'S' and ref[i]['t'] == hyp[j]['t']) else 1
    return [x if x is not None else 1 for x in ok], ins


def selftest():
    man = {'table': [{'turn': i + 1, 'gold': f'S{1 + i // 8}', 'lang': 'en',
                      'voice': 'v', 'start_s': i * 2.0, 'end_s': i * 2.0 + 1.9, 'dur_s': 1.9,
                      'sha': '', 'text': 'the quick brown fox jumps over lazy dogs'}
                     for i in range(8)],
           'roles': [{'gold': f'S{i}'} for i in range(1, 2)], 'turns': 8, 'total_s': 16.0,
           'wav_sha256': '0' * 16}
    good = True
    with open('/tmp/ca_a.txt', 'w', encoding='utf-8') as f:
        f.write(''.join(f"[{i+1}/9]  Speaker 0:{t['text']}\n" for i, t in enumerate(man['table'])))
    r = compare('/tmp/ca_a.txt', '/tmp/ca_a.txt', man)
    ok = abs(r['delta']) < 1e-12 and r['p'] > 0.99 and r['disc_a'] == 0 and r['disc_b'] == 0
    good &= ok
    print(f"  {'ok  ' if ok else 'FAIL'} identical systems : delta {r['delta']:+.4f} p {r['p']:.3f} (expect 0, 1.0)")
    # B makes errors on 6 of 64 tokens that A gets right -> McNemar must see it
    with open('/tmp/ca_b.txt', 'w', encoding='utf-8') as f:
        for i, t in enumerate(man['table']):
            w = t['text'].split()
            if i < 3:
                w[0] = 'WRONG'                       # 3 turns x 1 token = 3 discordant
            f.write(f"[{i+1}/9]  Speaker 0:{' '.join(w)}\n")
    r = compare('/tmp/ca_a.txt', '/tmp/ca_b.txt', man)
    ok = r['disc_b'] == 3 and r['disc_a'] == 0 and abs(r['p'] - 0.25) < 1e-9
    good &= ok
    print(f"  {'ok  ' if ok else 'FAIL'} 3 tokens differ    : b={r['disc_a']} c={r['disc_b']} p {r['p']:.4f} delta {r['delta']:+.4f}")
    print("self-test:", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(selftest())
    if sys.argv[1] == '--gate':
        # compare_arms.py --gate <dirA> <dirB> <refs.json> [label]
        r = compare_gate(sys.argv[2], sys.argv[3], sys.argv[4])
        lbl = sys.argv[5] if len(sys.argv) > 5 else 'gate'
        print(f"{lbl}: n={r['n']} tokens / {r['utts']} utts | A {r['wer_a']:.4f} vs B {r['wer_b']:.4f} "
              f"| A-B {r['delta']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
        print(f"    McNemar exact p={r['p']:.4g} from discordants b={r['disc_a']} c={r['disc_b']} "
              f"| insertions A={r['ins_a']} B={r['ins_b']}")
        sys.exit(0)
    hA, hB, manifest = sys.argv[1], sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(manifest)
    r = compare(hA, hB, json.load(open(manifest, encoding='utf-8')))
    print(f"{label}: n={r['n']} tokens | A {r['wer_a']:.4f} vs B {r['wer_b']:.4f} "
          f"| A-B {r['delta']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] (paired bootstrap)")
    print(f"    McNemar exact p={r['p']:.4g} from discordants b={r['disc_a']} c={r['disc_b']} "
          f"| insertions A={r['ins_a']} B={r['ins_b']}")
