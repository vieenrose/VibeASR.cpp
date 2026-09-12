#!/usr/bin/env python3
"""Build the multi-speaker bilingual accuracy gate (supersedes gate_bilingual v1).

What one sample must exercise, all at once (user requirement, Exp648-649):
  * zh-TW speech            - Mozilla Common Voice 17.0 zh-TW `test` split (CC0)
  * English speech          - LibriSpeech test-clean (public domain), local copies
  * zh-TW <-> en switching  - across turns (the languages come from different
                              corpora, so a single voice cannot speak both), AND
                              inside a turn: CV17 zh-TW test contains exactly one
                              naturally code-switched sentence ("google下去"), and
                              its speaker has 4 qualifying clips, so that voice is
                              used as speaker S1 and gets both a mixed and plain
                              turns. No synthetic audio, no splicing inside a turn.
  * speaker diarization     - 6 distinct gold voices with alternating turns.
  * timestamps              - NOT gated: this build emits no timestamps (checked
                              Exp649: no timestamp path in src/ or upstream; the
                              "timestamp" string in the LM gguf is a BPE vocab
                              entry). Turn times are recorded in the manifest and
                              printed as emission-latency diagnostics only.

Determinism: fixed seed, selection depends only on metadata (votes, length,
speaker), never on model output. The manifest records every turn and a sha256 of
the audio, so any change to the gate is visible as a hash change.
"""
import csv, hashlib, json, os, random, subprocess, sys, wave

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'eval-bilingual', 'raw')
OUT = os.path.join(ROOT, 'eval-bilingual')
SR, GAP_MS, SEED = 24000, int(os.environ.get('GAP_MS', 300)), 649
SFX = '' if GAP_MS == 300 else f'_g{GAP_MS}'   # keep the canonical probe's files untouched
ZH_TURNS_PER_SPEAKER = 4    # 2 zh-TW voices x 4 turns (~31 s), incl. the code-switched one
S1_TURNS = 4                # the code-switched voice only has 4 qualifying clips
ZH_SPEAKERS = 2
EN_VOICES = 2               # Common Voice en voices (CC0), 2 turns each (~24 s)
EN_TURNS_PER_VOICE = 2
EN_DUR_RANGE = (5.0, 12.0)  # keep the two languages duration-balanced
MIN_ZH_CHARS = 8          # prefer substantial turns (~3+ s)

def decode_pcm(path):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR),
                        '-f', 's16le', '-'], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[:160])
    return p.stdout

def main():
    rng = random.Random(SEED)
    gap = b'\x00\x00' * int(SR * GAP_MS / 1000)

    clip_index = {}
    for dirpath, _d, files in os.walk(os.path.join(RAW, 'clips')):
        for fn in files:
            if fn.endswith('.mp3'):
                clip_index[fn] = os.path.join(dirpath, fn)

    rows = list(csv.DictReader(open(os.path.join(RAW, 'cv17_zhtw_test.tsv'), encoding='utf-8'),
                               delimiter='\t'))
    qual = [r for r in rows if int(r['up_votes'] or 0) >= 2 and int(r['down_votes'] or 0) == 0
            and r['path'] in clip_index]

    # S1 = the voice that has the one naturally code-switched sentence
    cs = [r for r in qual if any(c.isascii() and c.isalpha() for c in r['sentence'])
          and len(r['sentence']) <= 40]
    assert cs, "expected a naturally code-switched zh-TW/en sentence in CV17 test"
    s1_id = max(cs, key=lambda r: sum(1 for x in qual if x['client_id'] == r['client_id']))['client_id']

    def pool(spk, n_turns, extra_ids=None, min_chars=MIN_ZH_CHARS):
        cand = [r for r in qual if r['client_id'] == spk and min_chars <= len(r['sentence']) <= 40]
        cand = [r for r in cand if not (extra_ids and r['sentence_id'] in extra_ids)]
        cand.sort(key=lambda r: r['sentence_id'])          # metadata only -> deterministic
        rng.shuffle(cand)
        out, seen = [], set()
        for r in cand:
            if r['sentence'] in seen: continue
            seen.add(r['sentence']); out.append(r)
            if len(out) >= n_turns: break
        return out

    zh_ids = [s1_id]
    by_spk = {}
    for r in qual:
        by_spk.setdefault(r['client_id'], []).append(r)
    others = []
    for cid, rs in sorted(by_spk.items()):
        if cid == s1_id:
            continue
        n = sum(1 for r in rs if MIN_ZH_CHARS <= len(r['sentence']) <= 40)
        if n:
            others.append((n, cid))
    others.sort(reverse=True)          # richest voices first -> full turn counts
    zh_ids += [cid for _n, cid in others[:ZH_SPEAKERS - 1]]
    roles = []
    for i, cid in enumerate(zh_ids):
        n_turns = S1_TURNS if cid == s1_id else ZH_TURNS_PER_SPEAKER
        turns = ([r for r in cs if r['client_id'] == cid][:1] if cid == s1_id else [])
        need = n_turns - len(turns)
        used = {t['sentence_id'] for t in turns}
        # S1 has few clips, so do not impose the length floor on it
        turns += pool(cid, need, extra_ids=used, min_chars=4 if cid == s1_id else MIN_ZH_CHARS)
        if cid == s1_id:
            turns.insert(len(turns) // 2, turns.pop(0))   # code-switch mid-stream, not first
        roles.append({'gold': f'S{i+1}', 'lang': 'zh-TW', 'voice': f'commonvoice-zhTW-{cid[:8]}',
                      'kind': 'zh', 'clips': [('mp3', t['path'], t['sentence']) for t in turns]})

    # English side: Common Voice 17.0 en test split (CC0) - same corpus family as the
    # zh-TW side, so the published artifact has a single provenance and every voice is
    # multi-turn (needed for a speaker-consistency signal).
    en_rows = list(csv.DictReader(open(os.path.join(RAW, 'cv17_en_test.tsv'), encoding='utf-8'),
                                  delimiter='\t'))
    en_clip_index = {}
    for dirpath, _d, files in os.walk(os.path.join(RAW, 'clips_en')):
        for fn in files:
            if fn.endswith('.mp3'):
                en_clip_index[fn] = os.path.join(dirpath, fn)
    en_qual = [r for r in en_rows if int(r['up_votes'] or 0) >= 2 and int(r['down_votes'] or 0) == 0
               and 20 <= len(r['sentence']) <= 60 and r['path'] in en_clip_index]
    en_by_spk = {}
    for r in sorted(en_qual, key=lambda r: (r['client_id'], r['sentence_id'])):
        en_by_spk.setdefault(r['client_id'], []).append(r)
    voices = [cid for cid, rs in sorted(en_by_spk.items()) if len(rs) >= EN_TURNS_PER_VOICE][:EN_VOICES]
    for j, cid in enumerate(voices):
        turns = en_by_spk[cid][:EN_TURNS_PER_VOICE]
        roles.append({'gold': f'S{len(roles)+1}', 'lang': 'en', 'voice': f'commonvoice-en-{cid[:8]}',
                      'kind': 'en', 'clips': [('mp3en', t['path'], t['sentence']) for t in turns]})

    # ---- schedule: seeded-shuffled rounds over the roles, so speaker AND language
    # change turn to turn instead of running one language in a block
    table, parts, pos, qi = [], [], 0, {id(r): 0 for r in roles}
    while any(qi[id(r)] < len(r['clips']) for r in roles):
        order = [r for r in roles if qi[id(r)] < len(r['clips'])]
        rng.shuffle(order)
        for r in order:
            k = qi[id(r)]
            if k >= len(r['clips']):
                continue
            kind, path, text = r['clips'][k]
            src = clip_index[path] if kind == 'mp3' else (en_clip_index[path] if kind == 'mp3en' else path)
            pcm = decode_pcm(src)
            dur = len(pcm) / 2 / SR
            parts.append(pcm); parts.append(gap)
            table.append({'turn': len(table) + 1, 'gold': r['gold'], 'lang': r['lang'],
                          'voice': r['voice'], 'start_s': round(pos / 2 / SR, 3), 'end_s': round((pos + len(pcm)) / 2 / SR, 3),
                          'dur_s': round(dur, 3), 'src': os.path.relpath(src, ROOT),
                          'sha': hashlib.sha256(pcm).hexdigest()[:16], 'text': text})
            pos += len(pcm) + len(gap)
            qi[id(r)] += 1

    wav_path = os.path.join(OUT, 'gate_ms' + SFX + '.wav')
    with wave.open(wav_path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(b''.join(parts))
    total = pos / 2 / SR          # pos is BYTES (s16le)
    ref_path = os.path.join(OUT, 'gate_ms' + SFX + '.ref.txt')
    with open(ref_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(t['text'] for t in table) + '\n')

    man = {'builder': '.auto/build_gate_multispk.py', 'seed': SEED, 'gap_ms': GAP_MS,
           'window_samples': 83200, 'hop_samples': 70400,
           'sources': {'zh-TW': 'Mozilla Common Voice 17.0 zh-TW test split (CC0), up_votes>=2 & down_votes==0',
                       'en': 'Mozilla Common Voice 17.0 en test split (CC0), up_votes>=2 & down_votes==0'},
           'gold_speakers': len(roles), 'turns': len(table), 'total_s': round(total, 2),
           'gaps_s': round(GAP_MS * len(table) / 1000.0, 2),
           'wav_sha256': hashlib.sha256(open(wav_path, 'rb').read()).hexdigest(),
           'ref_sha256': hashlib.sha256(open(ref_path, 'rb').read()).hexdigest(),
           'roles': [{'gold': r['gold'], 'lang': r['lang'], 'voice': r['voice'], 'turns': len(r['clips'])}
                     for r in roles],
           'table': table}
    json.dump(man, open(os.path.join(OUT, 'manifest_ms' + SFX + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    zs = sum(t['dur_s'] for t in table if t['lang'] == 'zh-TW')
    es = sum(t['dur_s'] for t in table if t['lang'] == 'en')
    print(f"built {os.path.relpath(wav_path, ROOT)}: {total:.1f}s, {len(table)} turns, "
          f"{len(roles)} gold voices (zh-TW {zs:.1f}s / en {es:.1f}s), sha256 {man['wav_sha256'][:16]}")
    mixed = [t['turn'] for t in table if any(c.isascii() and c.isalpha() for c in t['text'])
             and any('\u4e00' <= c <= '\u9fff' for c in t['text'])]
    print(f"  intra-turn code-switched turns: {mixed}")

if __name__ == '__main__':
    main()
