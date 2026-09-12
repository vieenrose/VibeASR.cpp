#!/usr/bin/env python3
"""Score the multi-speaker bilingual gate on the axes this build can actually measure:

  1. WER          - hybrid tokenization (CJK per character, Latin per word), folded
                    for case/punctuation and traditional<->simplified. Reported overall
                    and split by language (a zh error and an English word error each
                    cost 1, so neither language can hide inside the other).
  2. Diarization  - two numbers, both derived from the WER alignment so no timestamps
                    are needed (this build emits none):
                      * ATTRIBUTION  - % of aligned reference tokens whose gold voice
                                       does not match the Speaker tag the system used
                                       at that point in the stream.
                      * CONSISTENCY  - per gold voice with >=2 turns, the share of its
                                       attributed tokens carrying its majority tag
                                       (does one voice keep one label?).
                    plus distinct tags used vs gold voices (collapse / over-split).
  3. Emission timing (DIAGNOSTIC ONLY, not gated) - window index of each turn's first
                    emitted token, converted to seconds via HOP. Reported because the
                    engine has no token timestamps; window resolution is ~2.9 s.

Usage: .auto/score_stream.py [hyp-file] [manifest.json]   (default VibeASR.cpp/.auto/last_out.txt)
       .auto/score_stream.py --selftest
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_mixed import normalize, tokenize, CJK   # noqa: E402
import unicodedata                                  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MAN = os.path.join(ROOT, 'eval-bilingual', 'manifest_ms.json')
HOP_S = 70400 / 24000.0          # window k starts at k * HOP_S seconds
win_re = re.compile(r'^\[(\d+)/(\d+)\]\s?(.*)$')
spk_re = re.compile(r'speaker\s*(\d+)\s*:', re.I)
brk_re = re.compile(r'\[[^\]]*\]')

def _is_punct(ch):
    return unicodedata.category(ch).startswith('P') or unicodedata.category(ch).startswith('S')

def ref_stream(man):
    """Reference token stream, built turn by turn so every token keeps its gold
    speaker/turn/language. Concatenating per-turn tokens is what the .ref file is."""
    toks = []
    for t in man['table']:
        cleaned = ''.join(' ' if _is_punct(c) else c for c in t['text']).strip()
        for tok in tokenize(cleaned.lower()):
            toks.append({'t': tok, 'gold': t['gold'], 'turn': t['turn'],
                         'lang': 'zh' if re.match('[' + CJK + ']', tok) else 'en',
                         'start_s': t['start_s']})
    return toks

def hyp_stream(path):
    """Parse the streaming CLI output: [k/N] window markers plus 'Speaker N:' tags.
    The CLI also prints a final '--- Transcription ---' summary that repeats every
    window line; parsing it would double-count the whole transcript, so stop there."""
    runs = []
    win = 0
    for line in open(path, encoding='utf-8'):
        if line.startswith('---') and 'Transcription' in line:
            break
        m = win_re.match(line.strip())
        if m:
            win = int(m.group(1)) - 1
            body = m.group(3)
        else:
            body = line.rstrip('\n')
        if not body.strip():
            continue
        body = brk_re.sub(' ', body)
        pos, cur = 0, None
        for mm in spk_re.finditer(body):
            seg = body[pos:mm.start()]
            if seg.strip():
                runs.append((win, cur, seg))
            cur = 'T' + mm.group(1)          # 'T' = system tag namespace, distinct from gold 'S#' ids
            pos = mm.end()
        seg = body[pos:]
        if seg.strip():
            runs.append((win, cur, seg))
    toks = []
    for win_i, spk, seg in runs:
        cleaned = ''.join(' ' if _is_punct(c) else c for c in spk_re.sub(' ', seg)).strip()
        for tok in tokenize(cleaned.lower()):
            toks.append({'t': tok, 'spk': spk, 'win': win_i})
    return toks

def align(ref, hyp):
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    P = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): D[i][0], P[i][0] = i, (i - 1, 0, 'D')
    for j in range(1, m + 1): D[0][j], P[0][j] = j, (0, j - 1, 'I')
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = D[i-1][j-1] + (0 if ref[i-1]['t'] == hyp[j-1]['t'] else 1)
            best = min(((sub, 'S'), (D[i-1][j] + 1, 'D'), (D[i][j-1] + 1, 'I')), key=lambda x: x[0])
            D[i][j], P[i][j] = best[0], ((i-1, j-1, best[1]) if best[1] == 'S' else
                                         (i-1, j, 'D') if best[1] == 'D' else (i, j-1, 'I'))
    ops, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj, op = P[i][j]
        ops.append((op, pi if op != 'I' else None, pj if op != 'D' else None))
        i, j = pi, pj
    return list(reversed(ops)), D[n][m]

def score(hyp_path, man=None):
    man = man or json.load(open(MAN, encoding='utf-8'))
    ref, hyp = ref_stream(man), hyp_stream(hyp_path)
    ops, dist = align(ref, hyp)
    S = sum(1 for o, i, j in ops if o == 'S' and ref[i]['t'] != hyp[j]['t'])
    H = sum(1 for o, i, j in ops if o == 'S' and ref[i]['t'] == hyp[j]['t'])
    D = sum(1 for o, *_ in ops if o == 'D')
    I = sum(1 for o, *_ in ops if o == 'I')
    wer = (S + D + I) / len(ref) if ref else 0.0

    per = {}
    for op, i, j in ops:
        lang = ref[i]['lang'] if op != 'I' else ('zh' if re.match('[' + CJK + ']', hyp[j]['t']) else 'en')
        d = per.setdefault(lang, {'n': 0, 'err': 0})
        d['n'] += 1
        d['err'] += 0 if op == 'S' and ref[i]['t'] == hyp[j]['t'] else 1

    # Diarization attribution. System tags are 0-based and arbitrary in ORDER, so the
    # comparison must be over a mapping, not literal ids (this is what DER/cpWER do):
    # find the one-to-one tag<->gold assignment that maximizes correctly attributed
    # tokens, then report the residual error.
    conf = {}
    tag_by_voice = {}
    turn_first_win = {}
    attributed = 0
    for op, i, j in ops:
        if op in ('D', 'I'):
            continue          # deletions have no system tag, insertions have no gold speaker
        r, h = ref[i], hyp[j]
        attributed += 1
        tag = h['spk'] or '?'
        conf[(r['gold'], tag)] = conf.get((r['gold'], tag), 0) + 1
        tag_by_voice.setdefault(r['gold'], {}).setdefault(tag, 0)
        tag_by_voice[r['gold']][tag] += 1
        turn_first_win.setdefault((r['turn'], r['gold']), h['win'])
    golds = sorted({g for g, _ in conf})
    tags = sorted({t for _, t in conf})
    best = 0
    def search(k, used, gain):
        nonlocal best
        if k == len(golds):
            best = max(best, gain); return
        search(k + 1, used, gain)              # leave this voice unmapped (system used fewer tags)
        for t in tags:
            if t in used: continue
            used.add(t)
            search(k + 1, used, gain + conf.get((golds[k], t), 0))
            used.discard(t)
    if len(tags) <= 8 and len(golds) <= 8:
        search(0, set(), 0)
    else:                                     # safety valve for very many speakers
        best = sum(max(conf.get((g, t), 0) for t in tags) for g in golds)
    attribution = 1.0 - best / attributed if attributed else 0.0
    consist = {}
    for g, tags in tag_by_voice.items():
        tot = sum(tags.values())
        consist[g] = max(tags.values()) / tot if tot else 0.0
    consistency = sum(consist[g] for g in consist if len(tag_by_voice[g]) >= 1) / len(consist) if consist else 0.0
    tags_used = len({h['spk'] for h in hyp if h['spk']})
    gold_n = man.get('gold_voices') or man.get('gold_speakers') or len(man['roles'])
    onset_err = [abs(t['start_s'] - (turn_first_win[(t['turn'], t['gold'])] * HOP_S))
                 for t in man['table'] if (t['turn'], t['gold']) in turn_first_win]
    onset_err.sort()
    return {'wer': wer, 'S': S, 'D': D, 'I': I, 'H': H, 'ref_tokens': len(ref), 'hyp_tokens': len(hyp),
            'per_lang': {k: (v['err'] / v['n'] if v['n'] else 0.0, v['n']) for k, v in per.items()},
            'attribution': attribution, 'attributed': attributed, 'coverage': attributed / len(ref) if ref else 0,
            'consistency': consistency, 'per_voice': consist, 'tags_used': tags_used,
            'gold_voices': len(man['roles']), 'conf': conf,
            'onset_med_s': (onset_err[len(onset_err)//2] if onset_err else float('nan')),
            'onset_n': len(onset_err)}

def _synthetic_hyp(man, mode, path):
    """Build a known-good (or deliberately wrong) hypothesis for self-testing.
    System tags are emitted 0-based (as the engine does); gold ids are 1-based."""
    lines = []
    for t in man['table']:
        tag = str(int(t['gold'][1:]) - 1)      # gold S1 -> system Speaker 0
        if mode == 'one_tag': tag = '0'                       # collapse: one bucket
        if mode == 'swap' and t['gold'] in ('S2', 'S3'): tag = '2' if t['gold'] == 'S2' else '1'
        if mode == 'drop' and t['turn'] == man['table'][1]['turn']: continue
        win = int(t['start_s'] // HOP_S) + 1
        lines.append(f"[{win}/{len(man['table'])+1}]  Speaker {tag}:{t['text']}")
    open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

def selftest():
    man = json.load(open(MAN, encoding='utf-8'))
    ok = True
    tmp = '/tmp/score_stream_selftest.txt'
    _synthetic_hyp(man, 'perfect', tmp)
    r = score(tmp, man)
    good = r['wer'] < 1e-9 and r['attribution'] < 1e-9 and abs(r['consistency'] - 1.0) < 1e-9
    print(f"  {'ok  ' if good else 'FAIL'} perfect copy      : WER {r['wer']:.4f} attrib {r['attribution']:.4f} consist {r['consistency']:.3f}")
    ok &= good
    _synthetic_hyp(man, 'one_tag', tmp)
    r = score(tmp, man)
    toks = ref_stream(man)
    byvoice = {}
    for x in toks: byvoice[x['gold']] = byvoice.get(x['gold'], 0) + 1
    want = 1.0 - max(byvoice.values()) / len(toks)   # best mapping: bucket -> biggest voice
    good = abs(r['attribution'] - want) < 0.02 and r['tags_used'] == 1
    print(f"  {'ok  ' if good else 'FAIL'} collapsed to 1 tag  : attrib {r['attribution']:.4f} (expect {want:.4f}) consist {r['consistency']:.3f} tags {r['tags_used']}")
    ok &= good
    _synthetic_hyp(man, 'swap', tmp)
    r = score(tmp, man)
    # A permutation-invariant metric CANNOT see a pure relabeling - that is the point of
    # the mapping. The confusion matrix is where the swap must show up.
    swapped_visibly = (r['conf'].get(('S2', 'T2'), 0) > 0 and r['conf'].get(('S3', 'T1'), 0) > 0)
    good = r['attribution'] < 1e-9 and r['tags_used'] >= 3 and swapped_visibly
    print(f"  {'ok  ' if good else 'FAIL'} S2<->S3 swapped    : attrib {r['attribution']:.4f} (0: permutation-invariant) tags {r['tags_used']} conf-reveals-swap {swapped_visibly}")
    ok &= good
    _synthetic_hyp(man, 'drop', tmp)
    r = score(tmp, man)
    good = r['D'] > 0 and r['wer'] > 0
    print(f"  {'ok  ' if good else 'FAIL'} one turn deleted   : WER {r['wer']:.4f} D={r['D']} coverage {r['coverage']:.3f}")
    ok &= good
    # regression: the CLI's trailing '--- Transcription ---' summary must not be counted
    _synthetic_hyp(man, 'perfect', tmp)
    body = open(tmp, encoding='utf-8').read()
    open(tmp, 'w', encoding='utf-8').write(body + '\n--- Transcription ---\n' +
                                           ''.join(' Speaker %d:%s\n' % (int(t['gold'][1:]) - 1, t['text'])
                                                   for t in man['table']))
    r2 = score(tmp, man)
    good = r2['wer'] < 1e-9 and r2['hyp_tokens'] == len(ref_stream(man))
    print(f"  {'ok  ' if good else 'FAIL'} + summary block     : WER {r2['wer']:.4f} hyp_tokens {r2['hyp_tokens']} (no double count)")
    ok &= good
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        sys.exit(selftest())
    hyp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'VibeASR.cpp', '.auto', 'last_out.txt')
    man_path = sys.argv[2] if len(sys.argv) > 2 else MAN
    man = json.load(open(man_path, encoding='utf-8'))
    r = score(hyp, man)
    print(f"GATE {man['total_s']}s, {man['turns']} turns, {r['gold_voices']} gold voices, sha {man['wav_sha256'][:12]}")
    print(f"  WER          {r['wer']:.4f} (S={r['S']} D={r['D']} I={r['I']} H={r['H']}) over {r['ref_tokens']} ref tokens")
    for lang, (w, n) in sorted(r['per_lang'].items()):
        print(f"    {lang:3s}      {w:.4f}  ({n} ref tokens)")
    print(f"  DIARIZATION  attribution {r['attribution']:.4f} of {r['attributed']} attributed "
          f"(coverage {r['coverage']:.3f}) | consistency {r['consistency']:.3f} | tags used {r['tags_used']}/{r['gold_voices']}")
    print(f"    per-voice consistency: " + ", ".join(f"{k}:{v:.2f}" for k, v in sorted(r['per_voice'].items())))
    print(f"  (diagnostic) turn-onset |err| median {r['onset_med_s']:.2f}s over {r['onset_n']} turns "
          f"- window resolution floor {HOP_S:.2f}s, not gated")
