#!/usr/bin/env python3
"""Build the bilingual (zh-TW / en) long-form accuracy gate.

Design decisions (user-approved, Exp648):
  * source      Common Voice 17.0 zh-TW `test` split (CC0) for the Mandarin side,
                LibriSpeech test-clean (public domain, already in eval-librispeech/)
                for the English side. No synthetic audio, no model-generated refs.
  * shape       ONE continuous stream (~TARGET_MIN minutes), so the streaming
                pipeline is exercised across many windows, both languages, and
                many speakers - the current 10 s protocol clip and the 40 short
                gate utterances are both far shorter/simpler.
  * determinism fixed seed + an explicit per-clip manifest (offsets, durations,
                texts, sha256 of the wav). Re-running must reproduce byte-identical
                audio, and any change to the selection shows up as a hash change.
  * no tuning   the selection is made once from the metadata only (votes, length,
                speaker diversity). Never re-select to make a WER number look better.

Outputs (repo root /eval-bilingual):
  gate_bilingual.wav        24 kHz mono s16le - the gate audio
  gate_bilingual.ref.txt    gold transcript (clips joined by a space, in order)
  manifest.json             provenance + per-clip table + sha256
"""
import csv, hashlib, io, json, os, random, subprocess, sys, wave

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'eval-bilingual', 'raw')
OUT = os.path.join(ROOT, 'eval-bilingual')
SR = 24000
GAP_MS = 300            # silence between clips
TARGET_MIN = 3.2        # minutes of audio
MAX_ZH_CHARS, MIN_ZH_CHARS = 40, 4
SEED = 20260912

def decode_pcm(path):
    """Decode any audio file to 24 kHz mono s16le PCM bytes."""
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR),
                        '-f', 's16le', '-'], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[:200])
    return p.stdout

def main():
    rng = random.Random(SEED)
    gap = b'\x00\x00' * int(SR * GAP_MS / 1000)

    # the release tar keeps a wrapper directory, so index the mp3s by basename
    clip_index = {}
    for dirpath, _dirs, files in os.walk(os.path.join(RAW, 'clips')):
        for fn in files:
            if fn.endswith('.mp3'):
                clip_index[fn] = os.path.join(dirpath, fn)
    print(f"indexed clips: {len(clip_index)}", file=sys.stderr)

    # ---- zh-TW candidates: CV17 test split, upvoted, single-speaker-per-clip
    tsv = os.path.join(RAW, 'cv17_zhtw_test.tsv')
    rows = list(csv.DictReader(open(tsv, encoding='utf-8'), delimiter='\t'))
    cand = [r for r in rows
            if int(r['up_votes'] or 0) >= 2 and int(r['down_votes'] or 0) == 0
            and MIN_ZH_CHARS <= len(r['sentence']) <= MAX_ZH_CHARS]
    # a few naturally code-switched lines (Latin words inside a Chinese sentence)
    cs = [r for r in cand if any(c.isascii() and c.isalpha() for c in r['sentence'])]
    rest = [r for r in cand if r not in cs]
    rng.shuffle(rest); rng.shuffle(cs)
    zh_pool, seen_spk = [], set()
    for r in cs + rest:                      # code-switched lines get priority
        if r['client_id'] in seen_spk:
            continue
        seen_spk.add(r['client_id'])
        zh_pool.append(r)
    print(f"zh-TW candidates: {len(cand)} (naturally code-switched: {len(cs)}), "
          f"diverse-speaker pool: {len(zh_pool)}", file=sys.stderr)

    # ---- en candidates: LibriSpeech test-clean, already 24 kHz mono, public domain
    en_dir = os.path.join(ROOT, 'eval-librispeech', 'wav24k')
    en_refs = json.load(open(os.path.join(ROOT, 'eval-librispeech', 'refs.json')))
    en_all = sorted(f for f in os.listdir(en_dir) if f.endswith('.wav')
                    and os.path.basename(f)[:-4] in en_refs)
    en_pool = [os.path.join(en_dir, f) for f in en_all[::2]]
    rng.shuffle(en_pool)
    print(f"en candidates: {len(en_pool)}", file=sys.stderr)

    # ---- interleave: zh leads (it is the bulk), en every 2nd/3rd slot
    target_bytes = int(TARGET_MIN * 60 * SR * 2)
    parts, table = [], []
    zi = ei = 0
    zh_spent = en_spent = 0
    pos = 0
    n_clips = 0
    while len(b''.join(parts)) < target_bytes and zi < len(zh_pool) and ei < len(en_pool):
        take_en = (n_clips % 3 == 2) and ei < len(en_pool)     # zh, zh, en, zh, zh, en, ...
        if take_en:
            src = en_pool[ei]; ei += 1; lang = 'en'; text = None
        else:
            r = zh_pool[zi]; zi += 1; lang = 'zh'
            src = clip_index.get(r['path'])
            if src is None:
                continue
            text = r['sentence']
        if not os.path.exists(src):
            continue
        try:
            pcm = decode_pcm(src)
        except Exception as e:
            print(f"skip {src}: {e}", file=sys.stderr); continue
        if lang == 'en':
            text = ' '.join(en_refs[os.path.basename(src)[:-4]].split())
        if text is None:
            print(f"no gold text for {src}", file=sys.stderr); continue
        dur = len(pcm) / 2 / SR
        parts.append(pcm); parts.append(gap)
        table.append({'lang': lang, 'src': os.path.relpath(src, ROOT), 'start_s': round(pos / SR, 3),
                      'dur_s': round(dur, 3), 'sha': hashlib.sha256(pcm).hexdigest()[:16],
                      'text': text})
        pos += len(pcm) + len(gap)
        n_clips += 1
        if lang == 'zh': zh_spent += dur
        else: en_spent += dur

    wav_body = b''.join(parts)
    wav_path = os.path.join(OUT, 'gate_bilingual.wav')
    with wave.open(wav_path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(wav_body)
    total = len(wav_body) / 2 / SR

    ref_path = os.path.join(OUT, 'gate_bilingual.ref.txt')
    with open(ref_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(t['text'] for t in table) + '\n')

    man = {'builder': '.auto/build_gate_bilingual.py', 'seed': SEED,
           'sources': {'zh': 'Mozilla Common Voice 17.0, zh-TW, test split (CC0), '
                             'up_votes>=2 & down_votes==0, <=1 clip per speaker',
                       'en': 'LibriSpeech test-clean (public domain), local eval-librispeech/wav24k'},
           'gap_ms': GAP_MS, 'target_min': TARGET_MIN,
           'clips': len(table), 'zh_clips': sum(1 for t in table if t['lang'] == 'zh'),
           'en_clips': sum(1 for t in table if t['lang'] == 'en'),
           'total_s': round(total, 2), 'zh_s': round(zh_spent, 2), 'en_s': round(en_spent, 2),
           'wav_sha256': hashlib.sha256(open(wav_path, 'rb').read()).hexdigest(),
           'ref_sha256': hashlib.sha256(open(ref_path, 'rb').read()).hexdigest(),
           'table': table}
    json.dump(man, open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f"built {os.path.relpath(wav_path, ROOT)}: {total:.1f}s "
          f"({man['zh_clips']} zh-TW clips / {man['zh_s']}s + {man['en_clips']} en clips / "
          f"{man['en_s']}s), sha256 {man['wav_sha256'][:16]}")

if __name__ == '__main__':
    main()
