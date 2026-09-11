#!/usr/bin/env python3
"""Post-quantize type audit (Exp593): every shipped gguf's sensitive tensors must
have the expected per-tensor types. This is the guard against llama-quantize's
silent demotions (Exp493: it demoted token_embd q6_K -> q4_0 with no warning,
costing 0.7 pp WER) and against shipping the wrong head variant.
Exits nonzero on mismatch. Usage: check_tensors.py <file.gguf> ..."""
import struct, sys
TYPES = {0:'f32', 1:'f16', 2:'q4_0', 3:'q4_1', 6:'q5_0', 7:'q5_1', 8:'q8_0',
         12:'q4_K', 13:'q5_K', 14:'q6_K', 31:'q4_0_4x4', 32:'q4_0_4x8',
         33:'q4_0_8x8', 37:'i8_s'}
def rd(fmt, f):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))
def rd_str(f):
    (ln,) = rd('<Q', f)
    return f.read(ln).decode('utf-8', 'replace')
_SCALAR = {0:('<B',1), 1:('<b',1), 2:('<H',2), 3:('<h',2), 4:('<I',4),
            5:('<i',4), 6:('<f',4), 7:('<B',1), 10:('<Q',8), 11:('<q',8), 12:('<d',8)}
def rd_val(f, t):
    if t == 8: return rd_str(f)
    if t in _SCALAR: return rd(_SCALAR[t][0], f)[0]
    if t == 9:
        et = rd('<I', f)[0]; n = rd('<Q', f)[0]
        if et == 8: return [rd_str(f) for _ in range(n)]
        fmt, sz = _SCALAR[et]
        return [rd(fmt, f)[0] for _ in range(n)]
    raise ValueError(t)
def tensors(path):
    out = {}
    with open(path, 'rb') as f:
        assert f.read(4) == b'GGUF'
        ver, nt, nkv = rd('<IQQ', f)
        assert ver == 3
        for _ in range(nkv):
            k = rd_str(f); t = rd('<I', f)[0]; rd_val(f, t)
        for _ in range(nt):
            name = rd_str(f)
            nd = rd('<I', f)[0]
            dims = [rd('<Q', f)[0] for _ in range(nd)]
            ty = rd('<I', f)[0]
            off = rd('<Q', f)[0]
            out[name] = (TYPES.get(ty, f'?{ty}'), dims)
    return out
# tensor-name pattern -> required type
RULES = [
    ('token_embd',   'q6_K',    'embeddings must keep source precision (Exp493)'),
    ('tok_embd',     'q6_K',    '(alias)'),
    ('output.weight', None,     'head must be q6_K (shipped) or q8_0 (faster variant) - never q5_K or below'),
    ('ffn',          'q4_0_4x4','ffn linears must be blocked-int8 (Exp476-483)'),
]
FAIL = 0
for path in sys.argv[1:]:
    ts = tensors(path)
    print(f"== {path}: {len(ts)} tensors")
    by_type = {}
    for name, (ty, dims) in ts.items():
        by_type.setdefault(ty, []).append(name)
    for ty in sorted(by_type):
        print(f"   {ty:10s} n={len(by_type[ty]):4d}")
    # sensitive-tensor checks
    emb = [n for n in ts if 'embd' in n]
    for n in emb:
        if ts[n][0] != 'q6_K':
            print(f"   FAIL {n}: {ts[n][0]} (expected q6_K)"); FAIL = 1
        else: print(f"   ok   {n}: {ts[n][0]}")
    for n in ('output.weight',):
        if n in ts:
            if ts[n][0] not in ('q6_K', 'q8_0'):
                print(f"   FAIL {n}: {ts[n][0]} (expected q6_K or q8_0)"); FAIL = 1
            else: print(f"   ok   {n}: {ts[n][0]}")
    # ffn-on-blocked-int8 only applies when the file is a 4x4-from-source build
    # (the F16 VAE's linears are f16 by design)
    ffn_bad = [n for n in ts if '.ffn.' in n and 'weight' in n and ts[n][0] != 'q4_0_4x4'] if '4x4' in path else []
    if ffn_bad:
        print(f"   FAIL {len(ffn_bad)} ffn weights not q4_0_4x4 (e.g. {ffn_bad[0]}: {ts[ffn_bad[0]][0]})"); FAIL = 1
    else:
        if '4x4' in path:
            nffn = len([n for n in ts if '.ffn.' in n and 'weight' in n])
            if nffn: print(f"   ok   {nffn} ffn weights: q4_0_4x4")
sys.exit(FAIL)
