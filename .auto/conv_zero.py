#!/usr/bin/env python3
"""Measure-only: zero the VAE conv weights (Exp682) - PROVEN VOID FOR TIMING, keep as a probe.

Original aim: price the conv path before building conv-int8 quantization (Exp681 showed the VAE is
kernel-RATE-bound, removing the "weight bloat" objection that parked Exp489).

RESULT - THIS CANNOT PRICE ANYTHING, and the reason is principled: zeroing WEIGHTS does not remove
MAC work. The im2col, the matmul and every launch still execute; only the values change. Fixed-shape
GEMM timing is data-independent, so the arm measured VAE 16.9 s vs 17.1/17.0 reference = unchanged.
Measured, not assumed (Exp682).

What the tool IS good for:
  1. A data-independence control: if some future arm's VAE time changes under this file, something
     value-dependent has entered the kernel path and that is a bug worth chasing.
  2. A degenerate-input robustness probe: with every conv weight zeroed the network runs on biases
     alone, and the pipeline stays graceful (no NaN crash, no runaway: 9 tokens, RTF 2.23, VAE time
     identical to the reference).
  3. Confirming that vae_s is LM-independent: this arm emits 9 tokens vs 39 and vae_s is unchanged,
     a much larger LM delta than Exp674's garbage arms.

The correct way to price the conv path is an OP-LEVEL view substitution in vae.cpp (the VAE_ABL_TAPS
precedent: return a view/slice with the same output shape instead of running the matmul), NOT a data
change. Queued in .auto/ideas.md.

The gguf parser is still sound: it reports data_start plus every tensor's dims/dtype/offset, and
`inventory` validates itself by checking data_start + max(off + nbytes) == the real file size.

Usage:
  conv_zero.py inventory <file.gguf> [real_size_if_parsing_a_header_slice]
  conv_zero.py ranges    <file.gguf>
  conv_zero.py make      <in.gguf> <out.gguf>

Never gate its output, never recommend it, never ship it: it yields garbage text by design.
"""
import struct
import sys

TYPES = {0: ('f32', 4), 1: ('f16', 2), 2: ('q4_0', 18), 3: ('q4_1', 20), 8: ('q8_0', 34),
         12: ('q4_K', 144), 13: ('q5_K', 176), 14: ('q6_K', 208), 31: ('q4_0_4x4', 18),
         32: ('q4_0_4x8', 18), 33: ('q4_0_8x8', 18), 37: ('i8_s', 17)}
BLK = {2: 32, 3: 32, 8: 32, 12: 256, 13: 256, 14: 256, 31: 32, 32: 32, 33: 32, 37: 16}
SCAL = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<B',
        10: '<Q', 11: '<q', 12: '<d'}


def _rd(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))[0]


def _str(f):
    ln = _rd(f, '<Q')
    return f.read(ln).decode('utf-8', 'replace')


def _val(f, t):
    if t == 8:
        return _str(f)
    if t in SCAL:
        return _rd(f, SCAL[t])
    if t == 9:
        et = _rd(f, '<I')
        n = _rd(f, '<Q')
        if et == 8:
            return [_str(f) for _ in range(n)]
        return [_rd(f, SCAL[et]) for _ in range(n)]
    raise ValueError(f'gguf kv type {t}')


def parse(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'GGUF', 'not a gguf'
        ver = _rd(f, '<I')
        n_tensors = _rd(f, '<Q')
        n_kv = _rd(f, '<Q')
        align = 32
        for _ in range(n_kv):
            key = _str(f)
            t = _rd(f, '<I')
            v = _val(f, t)
            if key == 'general.alignment':
                align = int(v)
        tensors = []
        for _ in range(n_tensors):
            name = _str(f)
            nd = _rd(f, '<I')
            dims = [_rd(f, '<Q') for _ in range(nd)]
            dt = _rd(f, '<I')
            off = _rd(f, '<Q')
            tensors.append(dict(name=name, dims=dims, dt=dt, off=off))
        end = f.tell()
    return dict(ver=ver, align=align, tensors=tensors,
                data_start=(end + align - 1) // align * align)


def nbytes(dims, dt):
    ne = 1
    for d in dims:
        ne *= d
    if dt in BLK:
        full, sz = ne // BLK[dt], TYPES[dt][1]
        rem = ne - full * BLK[dt]
        return full * sz + (sz if rem else 0) if dt != 37 else full * sz
    return ne * TYPES[dt][1]


def convs(info):
    # conv weights are the 3-D tensors [K, IC, OC]; depthwise ones have IC == 1. Linears are 2-D.
    return [t for t in info['tensors'] if len(t['dims']) == 3]


def main():
    mode, path = sys.argv[1], sys.argv[2]
    info = parse(path)
    cs = convs(info)
    if mode == 'inventory':
        import os
        # when parsing a pulled HEADER SLICE (adb exec-out head -c N), the file's own size is wrong;
        # pass the real device-side size as the third argument so the end-of-data check stays valid.
        size = int(sys.argv[3]) if len(sys.argv) > 3 else os.path.getsize(path)
        end = max(info['data_start'] + t['off'] + nbytes(t['dims'], t['dt']) for t in info['tensors'])
        print(f"tensors={len(info['tensors'])} conv3d={len(cs)} "
              f"depthwise={sum(1 for t in cs if t['dims'][1] == 1)} align={info['align']}")
        print(f"data_start={info['data_start']}  predicted_end={end}  actual={size}  "
              + ('OK' if abs(end - size) < info['align'] * len(info['tensors']) else 'MISMATCH - do not trust ranges'))
        for t in cs[:8]:
            print(f"  {t['name'][:44]:44s} {str(t['dims']):20s} dt={TYPES.get(t['dt'], (t['dt'],))[0]}"
                  f" {nbytes(t['dims'], t['dt'])/1e6:6.2f} MB")
        print(f"conv bytes total = {sum(nbytes(t['dims'], t['dt']) for t in cs)/1e6:.1f} MB")
        return 0
    if mode == 'ranges':
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else None   # header-slice mode: skip out-of-range tensors
        for t in cs:
            s, nb = info['data_start'] + t['off'], nbytes(t['dims'], t['dt'])
            if limit is not None and s + nb > limit:
                print(f"# SKIP {t['name']} range {s}+{nb} beyond pulled slice", file=sys.stderr)
                continue
            print(f"{s} {nb} {t['name']}")
        return 0
    if mode == 'make':
        out = sys.argv[3]
        with open(path, 'rb') as fh:
            buf = bytearray(fh.read())
        for t in cs:
            s, nb = info['data_start'] + t['off'], nbytes(t['dims'], t['dt'])
            buf[s:s + nb] = bytes(nb)
        with open(out, 'wb') as fh:
            fh.write(buf)
        print(f'wrote {out}: {len(cs)} conv tensors zeroed')
        return 0
    raise SystemExit('mode?')


if __name__ == '__main__':
    sys.exit(main())
