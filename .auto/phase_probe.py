#!/usr/bin/env python3
"""Phase-alignment probe: does WHERE a clip lands on the windowed protocol's 70,400-sample grid change
what the model writes? (Exp866h; corroborating the Jetson loop's splice-alignment law on the A78 stack.)

Method, deliberately independent of any assumption about how an asset was stitched: insert silence into an
existing asset at one offset and nothing else. The audio CONTENT is byte-identical - only the phase of
everything after the insertion relative to the 2.93 s window grid changes. So a reference transcript is
shared by all arms and the comparison is exactly paired.

  --shift 3600   non-aligned (half a hop)      -> prediction: WER moves (phase is load-bearing)
  --shift 70400  aligned (exactly one hop)     -> prediction: WER unchanged (a control, not a variant)

Usage: phase_probe.py --in IN.wav --out OUT.wav --shift SAMPLES [--at SEC]
"""
import argparse
import hashlib
import wave

HOP = 70400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shift', type=int, required=True, help='samples of silence to insert')
    ap.add_argument('--at', type=float, default=0.0, help='insertion point in seconds (default: file start)')
    ap.add_argument('--delete', type=int, default=0,
                    help='ALSO remove this many samples at --at: a real content discontinuity, as opposed to '
                         'silence insertion. Exp868: silence insertion at ANY phase measured WER-neutral here, '
                         'so the "splice costs 15 pp" symptom must come from the cut, not the phase.')
    a = ap.parse_args()

    with wave.open(a.src) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, 'expected 16-bit mono'
        sr, pcm = w.getframerate(), w.readframes(w.getnframes())
    # 16-bit mono: a SAMPLE index is 2 BYTES. Confusing the two inserts at the wrong place and prints a
    # nonsense duration, so do the arithmetic in samples and convert once, explicitly.
    n = len(pcm) // 2
    at = min(int(a.at * sr), n)
    if a.delete:
        # Cut INSIDE the audio (the caller picks --at inside a clip). Deleting from a silence gap would only
        # shorten the gap, which is the insertion case in disguise.
        out = pcm[:2 * at] + b'\x00\x00' * a.shift + pcm[2 * (at + a.delete):]
    else:
        out = pcm[:2 * at] + b'\x00\x00' * a.shift + pcm[2 * at:]
    with wave.open(a.out, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(out)
    body = hashlib.sha256(pcm).hexdigest()
    print(f'inserted {a.shift} samples at {a.at:.3f}s (deleted {a.delete})  ({a.shift % HOP} mod hop -> '
          f'{"ALIGNED, expect no change" if a.shift % HOP == 0 else "NON-ALIGNED, phase shift " + str(a.shift % HOP)})')
    print(f'  source pcm sha {body[:12]}  kept content: {len(pcm)} samples before, {len(pcm)} after '
          f'({a.shift} silence added)  duration {n / sr:.2f} s -> {(n + a.shift) / sr:.2f} s')


if __name__ == '__main__':
    main()
