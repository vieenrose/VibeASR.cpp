#!/usr/bin/env python3
"""Build gate_ms_v2: a diarization-CAPABLE multi-speaker probe (v1 measured the protocol blind spot).

Root cause (Exp696, user-requested debug): chunk speaker labels are numbered window-locally
because nothing carries speaker identity across windows (encoder cache reset per window,
text-only KV history, no speaker-embedding machinery anywhere). A second tag needs two
voices' contrast inside ONE window's acoustic embeddings (window 3.4667 s, hop 2.9333 s).
v1's sequential 300 ms-gap turns align to voice-pure windows -> all-Speaker-0, which
misreads as "the model hears one speaker".

v2 therefore contains segments where separation is achievable, plus sequential controls:
  * 3 OVERLAP pairs (2x zh-TW S1xS2/S2xS1, 1x en S3xS4): B starts 2.0 s before A ends, mixed at
    0.5 gain each. The pre-pair gap is chosen (deterministic 10 ms search) so the first
    1.5 s of the overlap provably sits inside one window; the builder ASSERTS the shared
    window index and records it (grid-alignment proof, not luck).
  * 4 SEQUENTIAL controls (S1,S3,S2,S4, 300 ms gaps): reproduce v1 behavior in-file.

Same voices as v1 (S1..S4), fresh clips (different sentences; falls back to reuse only if
a voice has none left, recorded in the manifest). Same corpora/provenance (CV17 test, CC0),
same format (24 kHz mono s16le), fixed seed, metadata-only selection. v1 files untouched.
Reference/scorer contract unchanged: turns ordered by start time, one gold voice per turn,
so score_stream.py and compare_arms.py work unmodified (turn times feed diagnostics only).
"""
import csv, hashlib, json, os, random, subprocess, sys, wave
from array import array

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'eval-bilingual', 'raw')
OUT = os.path.join(ROOT, 'eval-bilingual')
SR, SEED = 24000, 650
HOP_S, WIN_S = 70400 / SR, 83200 / SR
OVERLAP_S, ALIGN_NEED_S, SEQ_GAP_S = 2.0, 1.5, 0.3

def decode_pcm(path):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR),
                        '-f', 's16le', '-'], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[:160])
    return p.stdout

def to_samples(pcm):
    a = array('h'); a.frombytes(pcm)
    return a

def to_pcm(a):
    return a.tobytes()

def mix(a_pcm, b_pcm):
    """B mixed over the tail region starting at byte offset; 0.5 gain each, hard clip guard."""
    return a_pcm, b_pcm  # placeholder replaced below (explicit two-buffer mix in main)

def main():
    rng = random.Random(SEED)
    v1 = json.load(open(os.path.join(OUT, 'manifest_ms.json'), encoding='utf-8'))
    v1_roles = {r['voice']: r for r in v1['roles']}
    v1_texts = {t['text'] for t in v1['table']}
    assert len(v1_roles) == 4, "v1 must have 4 gold voices"

    clip_index, en_clip_index = {}, {}
    for dirpath, _d, files in os.walk(os.path.join(RAW, 'clips')):
        for fn in files:
            if fn.endswith('.mp3'):
                clip_index[fn] = os.path.join(dirpath, fn)
    for dirpath, _d, files in os.walk(os.path.join(RAW, 'clips_en')):
        for fn in files:
            if fn.endswith('.mp3'):
                en_clip_index[fn] = os.path.join(dirpath, fn)
    zh_rows = list(csv.DictReader(open(os.path.join(RAW, 'cv17_zhtw_test.tsv'), encoding='utf-8'),
                                  delimiter='\t'))
    en_rows = list(csv.DictReader(open(os.path.join(RAW, 'cv17_en_test.tsv'), encoding='utf-8'),
                                  delimiter='\t'))

    # same 4 voices as v1, matched by 8-char prefix (assert unique); fresh clips only
    # (sentences not used in v1), falling back to reuse per voice if fewer than 3 remain.
    voices = {}  # gold -> {'lang', 'prefix', 'kind', 'pool', 'fallback'}
    for r in v1_roles.values():
        pre = r['voice'].rsplit('-', 1)[-1]
        lang = r['lang']
        rows = zh_rows if lang == 'zh-TW' else en_rows
        lo, hi = (8, 40) if lang == 'zh-TW' else (20, 60)
        idx = clip_index if lang == 'zh-TW' else en_clip_index
        cands = [x for x in rows if x['client_id'][:8] == pre]
        assert len({x['client_id'] for x in cands}) == 1, f"prefix {pre} not unique"
        cands = [x for x in cands
                 if int(x['up_votes'] or 0) >= 2 and int(x['down_votes'] or 0) == 0
                 and lo <= len(x['sentence']) <= hi and x['path'] in idx]
        fresh = [x for x in cands if x['sentence'] not in v1_texts]
        fallback = False
        if len(fresh) < 3:
            fresh, fallback = cands, True
        fresh.sort(key=lambda x: x['sentence_id'])
        rng.shuffle(fresh)
        voices[r['gold']] = {'lang': lang, 'prefix': pre,
                             'kind': 'mp3' if lang == 'zh-TW' else 'mp3en',
                             'pool': fresh, 'fallback': fallback}

    # take pool as-is (2-12 clips/voice); assignment below consumes per-appearance.
    # Layout (10 turns): 4 sequential controls first (reproduce v1 per voice), then
    # 3 overlap pairs. Needs per voice: S1 3 (pool 4), S2 3 (pool 12, all fresh),
    # S3 2 (pool 2), S4 2 (pool 2). Overlaps are novel audio in all cases (v1 has no
    # mixed audio); per-turn fresh_clip flags record which clips are unseen.
    take = {gold: v['pool'] for gold, v in voices.items()}
    need = {'S1': 3, 'S2': 3, 'S3': 2, 'S4': 2}
    for gold, n in need.items():
        assert len(take[gold]) >= n, f"voice {gold} has {len(take[gold])} clips, needs {n}"
    plan = [('seq', 'S1'), ('seq', 'S3'), ('seq', 'S2'), ('seq', 'S4'),
            ('pair', ('S1', 'S2')), ('pair', ('S3', 'S4')), ('pair', ('S2', 'S1'))]
    # controls take index 0 of each voice; pairs consume from index 1 per appearance.
    use_count = {g: 1 for g in take}

    # pair (A,B): B starts OV before A ends; shared_window proves [tB, tB+1.5]
    # lies inside one window (grid alignment by pre-gap search, not luck).
    def shared_window(t0, need=ALIGN_NEED_S):
        frac = (t0 % HOP_S)
        if frac <= WIN_S - need:
            k = int(t0 // HOP_S)
            assert k * HOP_S <= t0 and t0 + need <= k * HOP_S + WIN_S
            return k
        return None
    table, segments, pos = [], [], 0.0  # pos in seconds; segments: (start_s, pcm_bytes, kind)
    pair_id = 0
    for kind, who in plan:
        if kind == 'seq':
            r = take[who][0]
            v = voices[who]
            idx = clip_index if v['kind'] == 'mp3' else en_clip_index
            pcm = decode_pcm(idx[r['path']])
            dur = len(pcm) / 2 / SR
            if segments:
                pos += SEQ_GAP_S
            table.append(mkturn(len(table) + 1, who, v, r, idx[r['path']], pos, dur, None,
                                r['sentence'] not in v1_texts, gain_db=0.0))
            segments.append((pos, pcm))
            pos += dur
        else:
            gA, gB = who
            rA = take[gA][use_count[gA]]; use_count[gA] += 1
            rB = take[gB][use_count[gB]]; use_count[gB] += 1
            vA, vB = voices[gA], voices[gB]
            idxA = clip_index if vA['kind'] == 'mp3' else en_clip_index
            idxB = clip_index if vB['kind'] == 'mp3' else en_clip_index
            a = to_samples(decode_pcm(idxA[rA['path']]))
            b = to_samples(decode_pcm(idxB[rB['path']]))
            # level-match pair clips: a voice far below its partner is a loudness
            # test, not a separation test. If either pair clip is under RMS 0.03,
            # scale BOTH to 0.05 (+12 dB cap each, peak-guarded); controls stay
            # byte-identical to source. Gains recorded per turn.
            import math as _m
            def _rms(x):
                return _m.sqrt(sum(v * v for v in x) / max(1, len(x))) / 32768
            def _scale(x, g):
                peak = max(abs(v) for v in x) or 1
                g = min(g, 0.95 * 32768 / peak)
                return array('h', (max(-32768, min(32767, int(round(v * g)))) for v in x)), g
            rA0, rB0 = _rms(a), _rms(b)
            if min(rA0, rB0) < 0.03:
                a, sA = _scale(a, min(0.05 / max(rA0, 1e-4), 3.98))
                b, sB = _scale(b, min(0.05 / max(rB0, 1e-4), 3.98))
                gA_db = round(20 * _m.log10(sA), 2)
                gB_db = round(20 * _m.log10(sB), 2)
            else:
                gA_db = gB_db = 0.0
            dA, dB = len(a) / SR, len(b) / SR
            assert dA > OVERLAP_S + 0.5 and dB > 1.0, "clips too short for a 2 s overlap"
            # deterministic gap search: pre-gap g in [0.3, 0.3+HOP) step 10 ms
            tA, chosen, kshare = None, None, None
            g = 0.3
            while g < 0.3 + HOP_S + 1e-9:
                candA = (segments[-1][0] + len(segments[-1][1]) / 2 / SR + g) if segments else g
                candB = candA + dA - OVERLAP_S
                k = shared_window(candB)
                if k is not None:
                    tA, chosen, kshare = candA, g, k
                    break
                g = round(g + 0.01, 3)
            assert chosen is not None, "grid alignment failed"
            tB = tA + dA - OVERLAP_S
            # mix: A full, B added over [tB, tB+dB) at 0.5/0.5
            total_len = int(round((tB + dB - tA) * SR))
            mixbuf = array('h', [0]) * total_len
            for i, s in enumerate(a):
                mixbuf[i] = s // 2
            off = int(round((tB - tA) * SR))
            peak = 0
            for i, s in enumerate(b):
                v_ = mixbuf[off + i] + s // 2
                v_ = max(-32768, min(32767, v_))
                mixbuf[off + i] = v_
                if abs(v_) > peak:
                    peak = v_
            pcm = to_pcm(mixbuf)
            pair_id += 1
            table.append(mkturn(len(table) + 1, gA, vA, rA, idxA[rA['path']], tA, dA, pair_id,
                                rA['sentence'] not in v1_texts, gain_db=gA_db,
                                overlap_s=round(min(dB, tA + dA - tB), 3), shared_window=kshare))
            table.append(mkturn(len(table) + 1, gB, vB, rB, idxB[rB['path']], tB, dB, pair_id,
                                rB['sentence'] not in v1_texts, gain_db=gB_db,
                                overlap_s=round(min(dB, tA + dA - tB), 3), shared_window=kshare))
            segments.append((tA, pcm))
            pos = tA + len(pcm) / 2 / SR
            print(f"  pair{pair_id} {gA}x{gB}: pre-gap {chosen:.2f}s, overlap {OVERLAP_S}s, "
                  f"shared window {kshare}, mix peak {peak/32768:.2f}")

    # render: segments may overlap in time (pairs were placed absolutely) -> lay onto one track
    end_s = max(s + len(p) / 2 / SR for s, p in segments)
    track = array('h', [0]) * int(round(end_s * SR) + SR)
    for s, p in segments:
        a = to_samples(p)
        off = int(round(s * SR))
        for i, sm in enumerate(a):
            v_ = track[off + i] + sm
            # sequential segments never overlap by construction; pairs are pre-mixed
            track[off + i] = max(-32768, min(32767, v_))
    out_pcm = to_pcm(track[:int(round(end_s * SR))])

    wav_path = os.path.join(OUT, 'gate_ms_v2.wav')
    with wave.open(wav_path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(out_pcm)
    ref_path = os.path.join(OUT, 'gate_ms_v2.ref.txt')
    with open(ref_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(t['text'] for t in table) + '\n')
    man = {'builder': '.auto/build_gate_ms_v2.py', 'seed': SEED, 'v1_manifest': 'manifest_ms.json',
           'window_samples': 83200, 'hop_samples': 70400,
           'design': 'overlap pairs (grid-aligned shared window, proof in table.shared_window) + '
                     'sequential controls; v1 files untouched',
           'sources': {'zh-TW': 'Mozilla Common Voice 17.0 zh-TW test split (CC0), up_votes>=2 & down_votes==0',
                       'en': 'Mozilla Common Voice 17.0 en test split (CC0), up_votes>=2 & down_votes==0'},
           'gold_speakers': 4, 'turns': len(table), 'pairs': pair_id,
           'total_s': round(end_s, 2),
           'wav_sha256': hashlib.sha256(open(wav_path, 'rb').read()).hexdigest(),
           'ref_sha256': hashlib.sha256(open(ref_path, 'rb').read()).hexdigest(),
           'roles': [{'gold': g, 'lang': voices[g]['lang'], 'fresh_clips_only': not voices[g]['fallback']}
                     for g in sorted(voices)],
           'table': table}
    json.dump(man, open(os.path.join(OUT, 'manifest_ms_v2.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f"built {os.path.relpath(wav_path, ROOT)}: {end_s:.1f}s, {len(table)} turns "
          f"({pair_id} overlap pairs + 4 controls), sha256 {man['wav_sha256'][:16]}")

def mkturn(n, gold, v, r, srcpath, start, dur, pair, fresh_clip, gain_db=0.0, overlap_s=None,
           shared_window=None):
    t = {'turn': n, 'gold': gold, 'lang': v['lang'],
         'voice': f"commonvoice-{'zhTW' if v['lang'] == 'zh-TW' else 'en'}-{v['prefix']}",
         'start_s': round(start, 3), 'end_s': round(start + dur, 3), 'dur_s': round(dur, 3),
         'src': os.path.relpath(srcpath, ROOT),
         'sha': hashlib.sha256(open(srcpath, 'rb').read()).hexdigest()[:16],
         'text': r['sentence'], 'pair': pair, 'fresh_clip': fresh_clip, 'gain_db': gain_db}
    if overlap_s is not None:
        t['overlap_s'] = overlap_s
        t['shared_window'] = shared_window
    return t

if __name__ == '__main__':
    main()
