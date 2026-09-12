#!/usr/bin/env python3
"""Generic held-out SET builder (generalizes build_holdout_en.py to other languages).

Usage:  PICKS=raw/holdout_zh_picks.json LANG=zh-TW NAME=holdout_zh python .auto/build_holdout_any.py

Selection is already fixed in the picks file (built from corpus metadata only:
up_votes >= 2, down_votes == 0, length band, one clip per voice, sorted by
client_id/sentence_id, and sentences already used by the bilingual probe excluded),
so this script never looks at a model. Output manifest uses the same schema as
score_stream.py, so one scorer serves every set.
"""
import hashlib, json, os, subprocess, wave

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'eval-bilingual', 'raw')
OUT = os.path.join(ROOT, 'eval-bilingual')
SR = 24000
GAP_MS = int(os.environ.get('GAP_MS', 300))
NAME = os.environ.get('NAME', 'holdout_any')
LANG = os.environ.get('LANG', 'en')
PICKS = os.environ['PICKS']
CLIPS_DIR = os.environ.get('CLIPS_DIR', 'clips' if LANG == 'zh-TW' else 'clips_holdout')


def decode_pcm(path):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR),
                        '-f', 's16le', '-'], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[:200])
    return p.stdout


def main():
    index = {}
    for dirpath, _d, files in os.walk(os.path.join(RAW, CLIPS_DIR)):
        for fn in files:
            if fn.endswith('.mp3'):
                index[fn] = os.path.join(dirpath, fn)
    picks = json.load(open(PICKS if os.path.isabs(PICKS) else os.path.join(RAW, PICKS),
                            encoding='utf-8'))
    gap = b'\x00\x00' * int(SR * GAP_MS / 1000)
    parts, table, pos = [], [], 0
    for i, p in enumerate(picks, 1):
        src = index.get(p['path'])
        if not src:
            raise SystemExit(f"missing clip {p['path']}")
        pcm = decode_pcm(src)
        parts += [pcm, gap]
        table.append({'turn': i, 'gold': f'S{i}', 'lang': 'zh' if LANG == 'zh-TW' else 'en',
                      'voice': f"commonvoice-{LANG.lower()}-{p['client_id'][:8]}",
                      'start_s': round(pos / 2 / SR, 3), 'end_s': round((pos + len(pcm)) / 2 / SR, 3),
                      'dur_s': round(len(pcm) / 2 / SR, 3),
                      'sha': hashlib.sha256(pcm).hexdigest(), 'text': p['sentence']})
        pos += len(pcm) + len(gap)

    wav = os.path.join(OUT, NAME + '.wav')
    with wave.open(wav, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(b''.join(parts))
    body = open(wav, 'rb').read()[44:]
    ref_path = os.path.join(OUT, NAME + '.ref.txt')
    with open(ref_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(t['text'] for t in table) + '\n')

    man = {'builder': '.auto/build_holdout_any.py', 'gap_ms': GAP_MS, 'sr': SR,
           'sources': {LANG: 'Mozilla Common Voice 17.0 test split (CC0), up_votes>=2, '
                             'down_votes==0, length band, 1 clip/voice, probe sentences excluded'},
           'gold_speakers': len(table), 'turns': len(table),
           'total_s': round(len(body) / 2 / SR, 2),
           'gaps_s': round(GAP_MS * len(table) / 1000.0, 2),
           'wav_sha256': hashlib.sha256(body).hexdigest(),
           'ref_sha256': hashlib.sha256(open(ref_path, 'rb').read()).hexdigest(),
           'roles': [{'gold': t['gold'], 'lang': t['lang'], 'voice': t['voice'], 'turns': 1,
                      'kind': t['lang']} for t in table],
           'table': table}
    json.dump(man, open(os.path.join(OUT, 'manifest_' + NAME + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f"built eval-bilingual/{NAME}.wav: {man['total_s']}s, {len(table)} clips, "
          f"sha256 {man['wav_sha256'][:16]}")


if __name__ == '__main__':
    main()
