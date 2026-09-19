#!/usr/bin/env python3
"""Hybrid zh/en scorer for the bilingual long-form gate (and any mixed-language text).

Why not jiwer's word WER: with Mandarin, a whole sentence becomes a handful of
"words" only if it is space-less (it is), so word-level WER degenerates; and in a
code-switched transcript one English word would count the same as one Chinese
CHARACTER. The convention used in code-switch ASR work is therefore:

    per CJK character = 1 token, per Latin/digit word = 1 token

Normalization, in order:
  1. drop speaker tags ("Speaker 0:") and bracketed events ("[Silence]" ...) - same
     as score_hyp.py, so tags never count as errors;
  2. traditional -> simplified (zhconv) - the model emits simplified glyphs while
     the audio is zh-TW, so script difference must not be scored as error;
  3. lowercase, strip punctuation of both scripts, collapse whitespace;
  4. tokenize: CJK run -> one token per char; anything else -> whitespace words.

Usage:  .auto/score_mixed.py [hyp-file]        (default: .auto/last_out.txt)
Self-test: .auto/score_mixed.py --selftest
"""
import os, re, sys, unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REF = os.path.join(ROOT, 'eval-bilingual', 'gate_bilingual.ref.txt')

CJK = r'\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff'
strip_spk = re.compile(r'speaker\s*\d+\s*:', re.I)
strip_brk = re.compile(r'\[[^\]]*\]')

try:
    from zhconv import convert as _zh
    def fold_script(s): return _zh(s, 'zh-hans')
except ImportError:                                   # pragma: no cover
    def fold_script(s): return s

def _is_punct(ch):
    return unicodedata.category(ch).startswith('P') or unicodedata.category(ch).startswith('S')

def normalize(s):
    s = strip_brk.sub(' ', strip_spk.sub(' ', s))
    s = fold_script(s)
    s = ''.join(' ' if _is_punct(c) else c for c in s)
    return re.sub(r'\s+', ' ', s).strip().lower()

def tokenize(s):
    out, buf = [], []
    for ch in s:
        if re.match('[' + CJK + ']', ch):
            if buf: out.append(''.join(buf)); buf = []
            out.append(ch)
        elif ch.isspace():
            if buf: out.append(''.join(buf)); buf = []
        else:
            buf.append(ch)
    if buf: out.append(''.join(buf))
    return out

def score(ref_txt, hyp_txt):
    r, h = tokenize(normalize(ref_txt)), tokenize(normalize(hyp_txt))
    n, m = len(r), len(h)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    P = [[(0, 0)] * (m + 1) for _ in range(n + 1)]        # (from, op)
    for i in range(1, n + 1):
        D[i][0], P[i][0] = i, ((i - 1, 0), 'D')
    for j in range(1, m + 1):
        D[0][j], P[0][j] = j, ((0, j - 1), 'I')
    D[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = D[i - 1][j - 1] + (0 if r[i - 1] == h[j - 1] else 1)
            dele = D[i - 1][j] + 1
            ins = D[i][j - 1] + 1
            best = min((sub, 'S'), (dele, 'D'), (ins, 'I'), key=lambda t: t[0])
            D[i][j] = best[0]
            P[i][j] = ((i - 1, j - 1) if best[1] == 'S' else
                       (i - 1, j) if best[1] == 'D' else (i, j - 1), best[1])
    # backtrace for the S/D/I split
    i, j, S, Dl, I, H = n, m, 0, 0, 0, 0
    while (i, j) != (0, 0):
        (pi, pj), op = P[i][j]
        if op == 'S':
            (S, Dl, I, H) = (S, Dl, I, H + 1) if r[pi] == h[pj] else (S + 1, Dl, I, H)
        elif op == 'D': Dl += 1
        else: I += 1
        i, j = pi, pj
    wer = (S + Dl + I) / n if n else 0.0
    return {'wer': wer, 'S': S, 'D': Dl, 'I': I, 'H': H, 'ref_tokens': n, 'hyp_tokens': m}

def selftest():
    cases = [
        ("你好世界", "你好世界", 0.0),
        ("你好世界", "你好事界", 1 / 4),                     # 1 char substituted (世->事)
        ("你好世界 hello", "你好 hello", 2 / 5),              # 2 chars deleted
                # Exp874: this fixture read "1 char inserted" but the string duplicated TWO characters (世界), so
        # the correct WER was 2/4 = 0.50 and this self-test has been RED since the commit that added it
        # (869e73e) - nothing ran it. The scorer was right; the fixture contradicted its own comment.
        ("你好世界", "你好世界世", 1 / 4),                     # 1 char inserted -> 1 error / 4 ref tokens
        ("你好世界 hello", "你好世界 hello world", 1 / 5),   # 1 insertion (a word)
        ("你好世界", "你 好 世 界", 0.0),                     # whitespace is irrelevant
        ("傳統漢字", "传统汉字", 0.0),                        # hant/hans folding
        ("你好, 世界!", "你好世界", 0.0),                     # punctuation ignored
        ("你好世界 hello", "你好世界 HELLO", 0.0),            # case folded
        ("Speaker 0: [Silence] 你好世界", "你好世界", 0.0),   # tags/events stripped
        ("你好", "", 1.0),                                    # all deleted
    ]
    ok = True
    for ref, hyp, want in cases:
        got = score(ref, hyp)['wer']
        flag = 'ok' if abs(got - want) < 1e-9 else 'FAIL'
        ok &= (flag == 'ok')
        print(f"  {flag:4s} ref={ref!r:28s} hyp={hyp!r:24s} wer={got:.4f} want={want:.4f}")
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        sys.exit(selftest())
    hyp_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'VibeASR.cpp', '.auto', 'last_out.txt')
    ref_txt = open(REF, encoding='utf-8').read()
    hyp_txt = open(hyp_path, encoding='utf-8').read()
    r = score(ref_txt, hyp_txt)
    print(f"bilingual gate: WER {r['wer']:.4f} (S={r['S']} D={r['D']} I={r['I']} H={r['H']}) "
          f"over {r['ref_tokens']} ref tokens ({r['hyp_tokens']} hyp tokens)")
    tok = tokenize(normalize(ref_txt))
    zh = sum(1 for t in tok if len(t) == 1 and re.match('[' + CJK + ']', t))
    print(f"reference mix: {zh} CJK chars + {len(tok) - zh} latin/digit words")
