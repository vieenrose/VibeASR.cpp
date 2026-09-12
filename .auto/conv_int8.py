#!/usr/bin/env python3
"""conv_int8.py <in.gguf> <out.gguf> - put the non-depthwise conv weights on the blocked-int8 path.

Exp685, the converter half of the conv-int8 project. For every 3-D F16 conv weight that is NOT
depthwise, rewrite it as a 2-D Q4_0_4x4 tensor [K*IC, OC]:

  * why  : those conv dots are 12.3% of VAE seconds (Exp683) and the VAE is kernel-RATE-bound rather
           than weight-traffic-bound (Exp681), so running them through the blocked 4x4 gemm instead
           of F16 dots is the lever.
  * legal: the blocked layout needs the matmul row (= ne[0]) to be a multiple of the 32-element
           block. K*IC already is for all 14 such convs here (128..16384), which is why NO kernel
           padding is needed - the Exp684 spike established that, and this script re-checks it and
           refuses to run if it stops being true.
  * free : a 3-D [K,IC,OC] F16 tensor's memory order (kw fastest, then ic) is already the row-major
           [K*IC, OC] matrix im2col produces rows in, so the bytes are reinterpreted, not reordered.

Everything else is copied byte-for-byte, the KV section is copied verbatim (no new metadata keys are
needed - the runtime recovers K from row/IC using its architecture table), and tensor order is kept.
The runtime side lives in src/vae.cpp (vae_conv_1d_i8 / vae_conv_geom); it auto-detects the layout,
so an unconverted file still works, which is the fallback.

Usage: conv_int8.py in.gguf out.gguf [--quantizer ./quant4x4] [--dry]
"""
import os
import struct
import subprocess
import sys

F32, F16, Q4_0_4X4 = 0, 1, 31
# element size and block size for every type this model's ggufs contain (same table conv_zero.py
# uses); a missing entry raises rather than guessing a tensor size, which is the safe failure.
TSIZE = {F32: 4, F16: 2, 2: 18, 3: 20, 8: 34, 12: 144, 13: 176, 14: 208, Q4_0_4X4: 18, 32: 18, 33: 18,
         37: 17}
BLK = {F32: 1, F16: 1, 2: 32, 3: 32, 8: 32, 12: 256, 13: 256, 14: 256, Q4_0_4X4: 32, 32: 32, 33: 32,
       37: 16}
SCAL = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<B', 10: '<Q', 11: '<q', 12: '<d'}


def _rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _str(f):
    return f.read(_rd(f, '<Q')).decode('utf-8', 'replace')


def _val(f, t):
    if t == 8:
        return _str(f)
    if t in SCAL:
        return _rd(f, SCAL[t])
    if t == 9:
        et, n = _rd(f, '<I'), _rd(f, '<Q')
        return [_str(f) for _ in range(n)] if et == 8 else [_rd(f, SCAL[et]) for _ in range(n)]
    raise ValueError(f'gguf kv type {t} - extend this parser rather than guessing')


def parse(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF', 'not a gguf'
    ver = _rd(f, '<I')
    n_tensors = _rd(f, '<Q')
    n_kv = _rd(f, '<Q')
    align = 32
    for _ in range(n_kv):
        key, t = _str(f), _rd(f, '<I')
        v = _val(f, t)
        if key == 'general.alignment':
            align = int(v)
    infos_start = f.tell()
    tensors = []
    for _ in range(n_tensors):
        name = _str(f)
        nd = _rd(f, '<I')
        dims = [_rd(f, '<Q') for _ in range(nd)]
        dt = _rd(f, '<I')
        off = _rd(f, '<Q')
        ne = tuple(reversed(dims))
        nb = 1
        for d in ne:
            nb *= d
        tensors.append(dict(name=name, nd=nd, dims=dims, dt=dt, off=off, ne=ne,
                            nbytes=nb * TSIZE[dt] // BLK[dt] if dt != F32 and dt != F16 else nb * TSIZE[dt]))
    data_start = f.tell()
    while (data_start % align) != 0:
        data_start += 1
    f.close()
    return dict(ver=ver, n=n_tensors, align=align, infos_start=infos_start, data_start=data_start,
                tensors=tensors)


def w_str(out, s):
    b = s.encode()
    out.write(struct.pack('<Q', len(b)) + b)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    q = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == '--quantizer' else '.auto/quant4x4'
    dry = '--dry' in sys.argv
    info = parse(src)
    real = os.path.getsize(src)
    # Parse-then-offset correctness matters more here than the measurement itself: writing the wrong
    # byte ranges would produce a plausible-looking file. Validate the parse against the input first
    # (conv_zero.py's discipline) - predicted_end may legitimately fall short by trailing pad.
    pred = info['data_start'] + max(t['off'] + t['nbytes'] for t in info['tensors'])
    if pred > real:
        sys.exit(f'input parse MISMATCH: predicted_end={pred} > size={real} - refusing to write')
    if real - pred > info['align']:
        print(f'note: input has {real - pred} bytes past the last tensor (metadata tail?)')

    targets, others = [], 0
    for t in info['tensors']:
        if t['nd'] == 3 and t['dims'][1] != 1 and t['dt'] == F16 and 'conv.weight' in t['name']:
            K, IC, OC = t['dims']
            row = K * IC
            if row % BLK[Q4_0_4X4] or OC % 4:
                sys.exit(f'REFUSING: {t["name"]} row={row} OC={OC} is not blocked-legal; '
                         f'this script does not implement kernel padding (see Exp684)')
            targets.append(t)
        elif t['nd'] == 3 and t['dims'][1] != 1 and t['dt'] == F16:
            others += 1
    print(f'{len(info["tensors"])} tensors, {len(targets)} convs to convert, '
          f'{others} other 3-D f16 tensors left alone, alignment {info["align"]}')
    if dry:
        for t in targets:
            print(f'  {t["name"][:46]:48s} {t["dims"]} -> [{t["dims"][0]*t["dims"][1]}, {t["dims"][2]}] q4_0_4x4')
        return

    # quantize each target into a temp blob first, so a failure leaves the output untouched
    blobs, tmp = {}, {}
    for t in targets:
        K, IC, OC = t['dims']
        path = f'/tmp/convint8_{abs(hash(t["name"]))}.bin'
        r = subprocess.run([q, src, str(info['data_start'] + t['off']), str(K * IC), str(OC), path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f'quantizer failed on {t["name"]}: {r.stderr.strip()}')
        tmp[t['name']], blobs[t['name']] = path, int(r.stdout.strip())
    before = sum(t['nbytes'] for t in targets)
    after = sum(blobs.values())
    print(f'conv weights: {before/1e6:.1f} MB f16 -> {after/1e6:.1f} MB q4_0_4x4 '
          f'({100.0*after/before:.0f}% of the bytes)')

    align, out = info['align'], open(dst, 'wb')
    with open(src, 'rb') as f:
        # infos_start is the offset just after the KV section, and the header (magic, version,
        # n_tensors, n_kv) is inside that prefix - the count must NOT be written again or every
        # later field shifts by 8 bytes (my first run did exactly that).
        out.write(f.read(info['infos_start']))
    offset, placed = 0, []
    for t in info['tensors']:
        if t['name'] in blobs:
            # This model's loader assigns ne[] in FILE order (verified: a 3-D conv [K,IC,OC] arrives
            # as ne=[K,IC,OC], and the runtime reads ne[0] as the kernel size), so the 2-D form must
            # be written [K*IC, OC] to land as ne=[row, OC]. My first run wrote [OC, row] and the
            # runtime guard caught it - a silent version of that would have been strides garbage.
            dims, dt, size = [t['dims'][0] * t['dims'][1], t['dims'][2]], Q4_0_4X4, blobs[t['name']]
        else:
            dims, dt, size = t['dims'], t['dt'], t['nbytes']
        t['_off'], t['_dims'], t['_dt'], t['_size'] = offset, dims, dt, size
        placed.append(t)
        offset += size
        while offset % align:
            offset += 1                                 # every tensor starts on an aligned offset
    for t in placed:
        w_str(out, t['name'])
        out.write(struct.pack('<I', len(t['_dims'])))
        for d in t['_dims']:
            out.write(struct.pack('<Q', d))
        out.write(struct.pack('<I', t['_dt']))
        out.write(struct.pack('<Q', t['_off']))
    # data_start is the aligned position after the tensor-info section. The converted infos are
    # SHORTER (2 dims instead of 3), so this is not the input's data_start and must not be assumed.
    while (out.tell() % align) != 0:
        out.write(b'\0')
    new_data_start = out.tell()
    with open(src, 'rb') as f:
        for t in placed:
            assert out.tell() == new_data_start + t['_off'], \
                f'offset drift on {t["name"]}: {out.tell()} vs {new_data_start + t["_off"]}'
            if t['name'] in tmp:
                with open(tmp[t['name']], 'rb') as qf:
                    out.write(qf.read())
            else:
                f.seek(info['data_start'] + t['off'])
                out.write(f.read(t['nbytes']))
            while (out.tell() % align) != 0:
                out.write(b'\0')
    out.close()
    for p in tmp.values():
        os.remove(p)
    print(f'wrote {dst}: {os.path.getsize(dst)/1e6:.1f} MB (input {real/1e6:.1f} MB), '
          f'data_start {info["data_start"]} -> {new_data_start}')

    chk = parse(dst)
    pred = max(t['off'] + t['nbytes'] for t in chk['tensors'])
    print(f'verify: {chk["n"]} tensors, predicted_end={chk["data_start"] + pred} '
          f'actual={os.path.getsize(dst)} {"OK" if chk["data_start"] + pred == os.path.getsize(dst) else "MISMATCH"}')
    n_i8 = sum(1 for t in chk['tensors'] if t['dt'] == Q4_0_4X4 and t['nd'] == 2 and 'conv.weight' in t['name'])
    print(f'verify: {n_i8} of {len(targets)} target convs read back as 2-D q4_0_4x4')


if __name__ == '__main__':
    main()
