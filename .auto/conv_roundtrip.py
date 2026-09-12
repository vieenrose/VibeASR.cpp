#!/usr/bin/env python3
"""Round-trip check for conv_int8.py output (Exp685): dequantize the blocked 4x4 conv weights with
the layout quantize_q4_0_nr_bl writes, and compare against the original F16 values.

Why this exists: the device run of the converted file produced garbage from window 1, and the host
cannot run the blocked gemm, so the byte-level question - "do my bytes represent the matrix I think
they represent" - has to be answered by dequantizing in Python. Reading ggml-aarch64.c:

    for b in range(0, nrow*n_per_row, nrows_interleaved*n_per_row):      # groups of 4 rows
        for x in range(n_per_row // 32):                                  # 32-element blocks
            for i in range(4):  quantize_row_q4_0_ref(src + b + i*n_per_row + x*32 -> tmp[i])
            write make_block_q4_0x4(tmp)          # 4 scales, then 4 x 16 bytes of nibbles

i.e. a "row" is n_per_row contiguous elements (= one output channel), and 4 rows are interleaved.

Usage: conv_roundtrip.py <orig_f16.gguf> <converted.gguf> [tensor_name]
"""
import struct
import sys

sys.path.insert(0, '.')
from conv_int8 import parse, F16, Q4_0_4X4   # same parser, so the offsets are the ones the tool used


def dequant_q4_0_4x4(buf, off, nrow, n_per_row):
    """Yield (row, col_in_row, value) for a [nrow, n_per_row] matrix in the 4x4 layout.

    Two things make this layout easy to get wrong, and my first attempt got both wrong (it reported
    max error == the full weight scale, which is what an offset looks like, not quantization noise):
      * make_block_q4_0x4 writes out.qs[i] = in[src_id].qs[src_offset] with
            src_id    = (i % 16) / 4                     # which of the 4 rows
            src_offset = (i // 16) * 4 + (i % 4)         # which byte of that row
        so the bytes INTERLEAVE the 4 rows (that is the whole point of 4x4 - one vector load feeds 4
        dot products). Reading it row-major is wrong.
      * the XOR mask 0x88 turns the (+8-biased) nibbles quantize_row_q4_0_ref wrote into two's-
        complement 4-bit values, so the value is d * signed4(nibble), NOT d * (nibble - 8).
    """
    nb = n_per_row // 32
    p = off
    for g in range(nrow // 4):
        for x in range(nb):
            scales = struct.unpack_from('<4e', buf, p)          # 4 fp16 scales
            qs = buf[p + 8: p + 8 + 64]                        # 64 interleaved bytes
            p += 72
            for i in range(64):
                row = g * 4 + (i % 16) // 4
                src_byte = (i // 16) * 4 + (i % 4)             # byte index within that row
                col = x * 32 + 2 * src_byte
                b = qs[i]
                for half in (0, 1):                            # low nibble = even element
                    n = (b & 0xF) if half == 0 else (b >> 4)
                    v = n - 8                                  # xor_mask 0x88 biases the nibbles
                    yield row, col + half, scales[(i % 16) // 4] * v


def halves(buf, off, n):
    return struct.unpack_from(f'<{n}e', buf, off)


def main():
    orig, conv = sys.argv[1], sys.argv[2]
    only = sys.argv[3] if len(sys.argv) > 3 else None
    a, b = parse(orig), parse(conv)
    amap = {t['name']: t for t in a['tensors']}
    fd_o, fd_c = open(orig, 'rb'), open(conv, 'rb')
    worst_total = 0.0
    for t in b['tensors']:
        if t['dt'] != Q4_0_4X4 or 'conv.weight' not in t['name'] or (only and only not in t['name']):
            continue
        o = amap[t['name']]
        if o['dt'] != F16 or o['nd'] != 3:
            print(f'  SKIP {t["name"]}: original is not 3-D f16')
            continue
        K, IC, OC = o['dims']
        n_per_row, nrow = K * IC, OC
        # NOTE: this project's loader assigns ne[] in FILE order (verified at runtime in Exp685), so
        # compare against dims as stored - the parser's reversed 'ne' is the gguf-spec convention and
        # is NOT what the binary sees. Comparing against it made this very check cry wolf.
        if nrow != t['dims'][1] or n_per_row != t['dims'][0]:
            print(f'  FAIL {t["name"]}: declared dims={t["dims"]} does not match [{n_per_row}, {nrow}]')
            worst_total = 1e9
            continue
        ob = fd_o.read(0)  # placate linters; reads use seek/read below
        fd_o.seek(a['data_start'] + o['off'])
        raw = fd_o.read(n_per_row * nrow * 2)
        fd_c.seek(b['data_start'] + t['off'])
        cbuf = fd_c.read(72 * (nrow // 4) * (n_per_row // 32))
        orig_vals = halves(raw, 0, n_per_row * nrow)
        # expected quantization error: half a step of the per-block scale, i.e. <= max|block|/15
        err = [0.0] * (n_per_row * nrow)
        seen = [0] * (n_per_row * nrow)
        for row, col, v in dequant_q4_0_4x4(cbuf, 0, nrow, n_per_row):
            idx = row * n_per_row + col
            err[idx] = abs(v - orig_vals[idx])
            seen[idx] += 1
        dup = sum(1 for s in seen if s != 1)
        mx = max(err)
        avg = sum(err) / len(err)
        scale = max(abs(x) for x in orig_vals) or 1.0
        flag = ''
        if dup:
            flag = f'  <-- {dup} elements written 0 or 2 times: LAYOUT MODEL IS WRONG'
        elif mx > 0.25 * scale:
            flag = '  <-- error far above quantization noise: WRONG MAPPING'
        print(f'  {t["name"][:44]:46s} K={K:2d} IC={IC:5d} OC={OC:5d} '
              f'max|err|={mx:.5f} avg={avg:.6f} weight_scale={scale:.3f}{flag}')
        worst_total = max(worst_total, mx / scale)
    print(f'worst relative error = {worst_total:.4f}  '
          f'({"consistent with q4_0 quantization: bytes are RIGHT" if worst_total < 0.25 else "BYTES ARE WRONG or the layout model is wrong"})')


if __name__ == '__main__':
    main()
