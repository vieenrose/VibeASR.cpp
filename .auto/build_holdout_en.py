#!/usr/bin/env python3
"""Held-out ENGLISH accuracy set - an anti-overfit check on the loop's WER claim.

Motivation (Exp651): the 40-utterance LibriSpeech gate has been consulted on every
keep decision for 650 runs. Even without anyone tuning weights against it, repeatedly
selecting changes that do not hurt one fixed sample gives mild selection pressure.
This set is English-only, from a corpus the loop has never scored against (Common
Voice 17.0 `en` test split, CC0), selected purely by metadata (up-votes >= 2, no
down-votes, 25-70 characters, one clip per voice, sorted order) and with the probe's
sentences explicitly excluded, so nothing here was chosen with a model in the loop.

It is one concatenated clip so it costs ONE device run; scoring uses score_stream.py
(hybrid tokenizer = word-level for Latin, so it is comparable to the eval40 scorer).
"""
import hashlib, json, os, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'eval-bilingual', 'raw')
OUT = os.path.join(ROOT, 'eval-bilingual')
SR, GAP_MS = 24000, 300


def decode_pcm(path):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR),
                        '-f', 's16le', '-'], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[:200])
    return p.stdout


def main():
    index = {}
    for dirpath, _d, files in os.walk(os.path.join(RAW, 'clips_holdout')):
        for fn in files:
            if fn.endswith('.mp3'):
                index[fn] = os.path.join(dirpath, fn)
    picks = json.load(open(os.path.join(RAW, 'holdout_en_picks.json'), encoding='utf-8'))

    gap = b'\x00\x00' * int(SR * GAP_MS / 1000)
    parts, table, pos = [], [], 0
    for i, p in enumerate(picks, 1):
        src = index.get(p['path'])
        if not src:
            raise SystemExit(f"missing clip {p['path']} (extract it from the en tar first)")
        pcm = decode_pcm(src)
        parts += [pcm, gap]
        table.append({'turn': i, 'gold': f'S{i}', 'lang': 'en',
                      'voice': f"commonvoice-en-{p['client_id'][:8]}",
                      'start_s': round(pos / 2 / SR, 3), 'end_s': round((pos + len(pcm)) / 2 / SR, 3),
                      'dur_s': round(len(pcm) / 2 / SR, 3),
                      'sha': hashlib.sha256(pcm).hexdigest(), 'text': p['sentence']})
        pos += len(pcm) + len(gap)

    wav = os.path.join(OUT, 'holdout_en.wav')
    import wave
    with wave.open(wav, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(b''.join(parts))
    body = open(wav, 'rb').read()[44:]

    ref_path = os.path.join(OUT, 'holdout_en.ref.txt')
    with open(ref_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(t['text'] for t in table) + '\n')

    man = {'builder': '.auto/build_holdout_en.py', 'gap_ms': GAP_MS, 'sr': SR,
           'sources': {'en': 'Mozilla Common Voice 17.0 en test split (CC0), '
                             'up_votes>=2, down_votes==0, 25-70 chars, 1 clip/voice'},
           'gold_speakers': len(table), 'turns': len(table),
           'total_s': round(len(body) / 2 / SR, 2),
           'gaps_s': round(GAP_MS * len(table) / 1000.0, 2),
           'wav_sha256': hashlib.sha256(body).hexdigest(),
           'ref_sha256': hashlib.sha256(open(ref_path, 'rb').read()).hexdigest(),
           'roles': [{'gold': t['gold'], 'lang': 'en', 'voice': t['voice'], 'turns': 1, 'kind': 'en'}
                     for t in table],
           'table': table}
    json.dump(man, open(os.path.join(OUT, 'manifest_holdout_en.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f"built eval-bilingual/holdout_en.wav: {man['total_s']}s, {len(table)} clips from "
          f"{len(table)} voices, sha256 {man['wav_sha256'][:16]}")


if __name__ == '__main__':
    main()
