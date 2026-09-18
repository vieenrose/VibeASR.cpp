#!/usr/bin/env python3
"""Re-phase a stitched eval asset onto the windowed protocol's grid, WITHOUT changing its audio content.

Why (Exp865h, corroborating the Jetson loop's splice-alignment law on the A78 stack): the windowed
protocol advances HOP = 22 frames x 3200 samples = 70,400 samples per window (2.933 s at 24 kHz), while
every stitched asset this loop built inserts a fixed 300 ms = 7,200-sample silence between clips. A
non-multiple-of-hop insertion shifts the phase of ALL later audio relative to the window grid, and phase
is load-bearing here: Exp833 (shortening the LM's per-chunk input cost 14 pp WER via deletions) and
Exp650 (English WER moved 0.107 -> 0.233 purely by changing silence-gap length) both say so.

Consequence for our accuracy claims: in holdout_en / holdout_zh / gate_ms / control_ls, clip i>1 sits at
phase (sum of previous clip lengths + 7,200 each) mod 70,400 - an arbitrary value that differs per clip.
The 40-utterance gate has phase 0 by construction (one clip per run), so a stitched-stream WER is NOT
comparable to the gate's.

This tool emits a variant in which every clip starts exactly on a hop boundary, keeping each clip's PCM
byte-identical - proven by recomputing every turn's sha against the manifest's own per-clip sha.

Usage: align_asset.py --wav IN --manifest MAN --out OUT [--min-gap-s 0.3] [--hop 70400] [--dry]
"""
import argparse
import hashlib
import json
import wave

SR_DEFAULT = 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--manifest-out')
    ap.add_argument('--min-gap-s', type=float, default=0.3)
    ap.add_argument('--hop', type=int, default=70400, help='samples per window hop (22 frames x 3200)')
    ap.add_argument('--phase', type=int, default=0,
                    help='target phase for EVERY clip (0 = starts a window). Exp868: a non-zero target puts all '
                         'clips at the SAME phase, which is the systematic case the many-splice assets cannot '
                         'produce - there, per-clip phases are i.i.d. and any effect averages out in the mean.')
    ap.add_argument('--dry', action='store_true', help='report phases, write nothing')
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    sr = man.get('sr', SR_DEFAULT)
    with wave.open(a.wav) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, 'expected 16-bit mono PCM'
        assert w.getframerate() == sr, f'rate {w.getframerate()} != manifest sr {sr}'
        pcm = w.readframes(w.getnframes())

    # Locate every clip EXACTLY, using two facts about the builder: gaps are exactly gap_samples of zeros,
    # so position_{i+1} = position_i + len_i + gap, and the manifest stores sha256(clip pcm) while dur_s is
    # only millisecond-rounded. So try the +-WIN candidate lengths and let the sha pick the true one. This is
    # exact and silent-content-proof; gap-run scanning is neither (a clip with an internal pause is a lie).
    gap_samples = int(round(man.get('gap_ms', 300) * sr / 1000))
    WIN = 40
    bounds, pos = [], 0          # positions are SAMPLES; slice with 2* to index BYTES (16-bit mono)
    for t in man['table']:
        d = round(t['dur_s'] * sr)
        want = t.get('sha')
        if want:
            hit = next((L for L in range(max(1, d - WIN), d + WIN + 1)
                        if hashlib.sha256(pcm[2 * pos:2 * (pos + L)]).hexdigest() == want), None)
            if hit is None:
                raise SystemExit(f'FAIL turn {t["turn"]}: no clip length within +-{WIN} samples of {d} '
                                 f'reproduces the manifest sha -> the tool model of this asset is wrong')
            bounds.append((pos, pos + hit))
            pos += hit + gap_samples
        else:
            n = round(t['dur_s'] * sr)
            bounds.append((pos, pos + n))
            pos += n + gap_samples
    if pos != len(pcm) + gap_samples and abs(pos - len(pcm)) > gap_samples:
        print(f'  note: reconstructed length {pos} vs file {len(pcm)} (builder appends a trailing gap)')

    min_gap = int(a.min_gap_s * sr)
    out, table, phases_before, phases_after = b'', [], [], []
    # A leading pad is required for a non-zero target: padding only BETWEEN clips leaves clip 1 at phase 0,
    # which would put one clip on a different phase from every other clip and quietly break the systematic
    # condition this option exists to create.
    pos = a.phase % a.hop
    out = b'\x00\x00' * pos
    for t, (s, e) in zip(man['table'], bounds):
        seg = pcm[2 * s:2 * e]
        got = hashlib.sha256(seg).hexdigest()
        if t.get('sha') and got != t['sha']:
            # Content identity is the whole point; a mismatch means the manifest does not describe this
            # file (or the asset drifted), and a re-phased result would be an unattributable mix.
            raise SystemExit(f'FAIL turn {t.get("turn")}: segment sha {got[:12]} != manifest {t["sha"][:12]}')
        phases_before.append(s % a.hop)
        phases_after.append(pos % a.hop)
        table.append({**t, 'start_s': round(pos / sr, 3), 'end_s': round((pos + len(seg)) / sr, 3),
                      # start_s is millisecond-rounded (+-12 samples at 24 kHz), which is enough to make an
                      # aligned asset look misaligned to any tool that re-derives positions from it. Keep the
                      # exact sample positions so 'starts a window' is checkable, not approximate.
                      'start_samples': pos, 'end_samples': pos + len(seg) // 2, 'sha': got})
        out += seg
        pos += len(seg) // 2
        pad = (a.phase - pos) % a.hop             # land every clip on the requested phase of the grid
        if pad < min_gap:
            pad += a.hop                          # keep a natural pause; a whole hop does not change alignment
        out += b'\x00\x00' * pad
        pos += pad

    uniq_b = len(set(phases_before))
    print(f'clips={len(table)} hop={a.hop} sr={sr}')
    print(f'  phase before: {len(phases_before)} clips, {uniq_b} distinct values'
          f' (0 = starts a window as in a standalone run)  first 6: {phases_before[:6]}')
    print(f'  phase after : {sorted(set(phases_after))}  <- must be [0]')
    print(f'  length: {len(pcm) / 2 / sr:.2f} s -> {len(out) / 2 / sr:.2f} s '
          f'(+{(len(out) - len(pcm)) / 2 / sr:.2f} s of silence; bytes/samples differ by 2x at 16-bit)')
    print(f'  content: every clip sha verified against the manifest'
          f' -> {"PROVEN identical" if all(t.get("sha") for t in man["table"]) else "no per-clip sha available"}')
    if set(phases_after) != {a.phase % a.hop}:
        raise SystemExit(f'FAIL: alignment landed on phases {sorted(set(phases_after))}, target {a.phase}')
    if a.dry:
        print('  (dry run, nothing written)')
        return

    with wave.open(a.out, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(out)
    if a.manifest_out:
        m2 = dict(man)
        m2['table'] = table
        m2['total_s'] = round(len(out) / sr, 2)
        m2['gaps_s'] = round(sum(t2['start_s'] - t1['end_s'] for t1, t2 in zip(table, table[1:])) / 1.0, 2)
        m2['rephased_from'] = {'wav_sha256': man.get('wav_sha256'), 'gap_ms': man.get('gap_ms'),
                               'hop_samples': a.hop, 'phase_target': a.phase, 'builder': '.auto/align_asset.py'}
        m2['wav_sha256'] = hashlib.sha256(open(a.out, 'rb').read()).hexdigest()
        json.dump(m2, open(a.manifest_out, 'w'), indent=1)
        print(f'  wrote {a.out} and {a.manifest_out}')


if __name__ == '__main__':
    main()
