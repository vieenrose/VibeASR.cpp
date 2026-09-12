#!/usr/bin/env python3
"""Dump weights as raw f32 / raw quantized bytes for conv_i8_ab.cpp (Exp687)."""
import struct, sys
sys.path.insert(0, '.')
from conv_int8 import parse, F16, Q4_0_4X4

M = '/home/user/vibe-asr-streaming-1p5b/models-streaming/'
# auto-pick the tensors my converter actually moved to Q4_0_4_4 (the non-depthwise convs, IC > 1)
_PICK = [t['name'] for t in parse(M + 'vae-encoder-convint8.gguf')['tensors']
         if t['dt'] == Q4_0_4X4 and 'conv' in t['name']][:2]
print('picked:', _PICK)
cases = [
    ('CONTROL', M + 'vae-encoder-f16.gguf', M + 'vae-encoder-q4x4ffn.gguf',
     'acoustic.stages.0.0.ffn.linear2.weight'),
    ('CONTROL2', M + 'vae-encoder-f16.gguf', M + 'vae-encoder-q4x4ffn.gguf',
     'acoustic.stages.3.0.ffn.linear2.weight'),
    ('CONV', M + 'vae-encoder-q4x4ffn.gguf', M + 'vae-encoder-convint8.gguf', _PICK[0]),
    ('CONV2', M + 'vae-encoder-q4x4ffn.gguf', M + 'vae-encoder-convint8.gguf', _PICK[1]),
]
args = []
for tag, fref, fq, name in cases:
    a, b = parse(fref), parse(fq)
    amap = {t['name']: t for t in a['tensors']}; bmap = {t['name']: t for t in b['tensors']}
    o, t = amap[name], bmap[name]
    fd = open(fref, 'rb'); fd.seek(a['data_start'] + o['off'])
    raw = fd.read(o['nbytes']); fd.close()
    if o['dt'] != F16: print(f'{tag}: ref not f16, skip'); continue
    h = struct.unpack(f'<{o["nbytes"]//2}e', raw)
    open(f'ab_{tag}_ref.bin', 'wb').write(struct.pack(f'<{len(h)}f', *h))
    ne0, ne1 = o['dims'][0], (o['dims'][1] if o['nd'] > 1 else 1)
    if o['nd'] == 3:
        K, IC, OC = o['dims']; ne0, ne1 = K * IC, OC
    fe = open(fq, 'rb'); fe.seek(b['data_start'] + t['off'])
    q = fe.read(t['nbytes']); fe.close()
    if t['dt'] != Q4_0_4X4:
        open(f'ab_{tag}_q.bin', 'wb').write(b''); qfile = '-'
    else:
        open(f'ab_{tag}_q.bin', 'wb').write(q); qfile = f'ab_{tag}_q.bin'
    print(f'{tag}: name={name} k={ne0} m={ne1} ref_f32_bytes={ne0*ne1*4} q_bytes={len(q)}')
    args += [tag, str(ne0), str(ne1), f'ab_{tag}_ref.bin', qfile]
open('ab_args.txt', 'w').write(' '.join(args))
