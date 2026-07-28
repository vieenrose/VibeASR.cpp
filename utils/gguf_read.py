"""Minimal GGUF reader.

The gguf-py bundled with the pinned llama.cpp calls ndarray.newbyteorder(), removed
in numpy 2.x, so it cannot open these files in this venv. Only tensor discovery and
raw block extraction are needed here, and both are a short read of a documented
format — simpler than pinning an older numpy for the whole toolchain.

Enough to lift I2_S ternary blocks straight out of the LM without requantizing:
they are already packed 4 weights per byte, the same layout ternary_pack() writes.
"""

import struct
from dataclasses import dataclass

import numpy as np

GGUF_MAGIC = 0x46554747  # "GGUF"

# Only the types this model actually uses.
TYPE_NAMES = {0: "F32", 1: "F16", 14: "Q6_K", 36: "I2_S", 39: "I8_S"}


@dataclass
class TensorInfo:
    name: str
    dims: tuple
    dtype: int
    offset: int          # relative to the start of the data section
    nbytes: int

    @property
    def type_name(self):
        return TYPE_NAMES.get(self.dtype, f"type{self.dtype}")


class Gguf:
    def __init__(self, path):
        self.path = path
        self.buf = np.memmap(path, mode="r", dtype=np.uint8)
        self.tensors = {}
        self.kv = {}
        self._parse()

    def _u32(self, o): return int(struct.unpack_from("<I", self.buf, o)[0]), o + 4
    def _u64(self, o): return int(struct.unpack_from("<Q", self.buf, o)[0]), o + 8

    def _str(self, o):
        n, o = self._u64(o)
        return bytes(self.buf[o:o + n]).decode("utf-8", "replace"), o + n

    def _skip_value(self, o, t):
        # Sized scalars, or a recursive array. Metadata values are skipped rather
        # than decoded: nothing here needs them, and decoding every type is a lot
        # of surface for no gain.
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in fixed:
            return o + fixed[t]
        if t == 8:                       # string
            n, o = self._u64(o)
            return o + n
        if t == 9:                       # array
            et, o = self._u32(o)
            n, o = self._u64(o)
            for _ in range(n):
                o = self._skip_value(o, et)
            return o
        raise ValueError(f"unknown gguf value type {t}")

    def _parse(self):
        o = 0
        magic, o = self._u32(o)
        if magic != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        self.version, o = self._u32(o)
        n_tensors, o = self._u64(o)
        n_kv, o = self._u64(o)

        alignment = 32
        for _ in range(n_kv):
            key, o = self._str(o)
            vt, o = self._u32(o)
            if key == "general.alignment" and vt == 4:
                alignment = int(struct.unpack_from("<I", self.buf, o)[0])
            o = self._skip_value(o, vt)

        infos = []
        for _ in range(n_tensors):
            name, o = self._str(o)
            nd, o = self._u32(o)
            dims = []
            for _ in range(nd):
                d, o = self._u64(o)
                dims.append(d)
            dt, o = self._u32(o)
            off, o = self._u64(o)
            infos.append(TensorInfo(name, tuple(dims), dt, off, 0))

        self.data_start = (o + alignment - 1) // alignment * alignment
        # Sizes come from the gaps between offsets; the last runs to EOF. Avoids
        # reimplementing per-type block sizing for custom types like I2_S.
        ordered = sorted(infos, key=lambda t: t.offset)
        for i, t in enumerate(ordered):
            end = ordered[i + 1].offset if i + 1 < len(ordered) else (len(self.buf) - self.data_start)
            t.nbytes = end - t.offset
            self.tensors[t.name] = t

    def raw(self, name) -> np.ndarray:
        t = self.tensors[name]
        s = self.data_start + t.offset
        return np.asarray(self.buf[s:s + t.nbytes])


if __name__ == "__main__":
    import sys
    from collections import Counter
    g = Gguf(sys.argv[1])
    print(f"gguf v{g.version}, {len(g.tensors)} tensors")
    print("types:", dict(Counter(t.type_name for t in g.tensors.values())))
    for name, t in list(g.tensors.items())[:4]:
        print(f"  {name:36} {t.type_name:6} {t.dims} {t.nbytes} B")
    print("  ...")
    for name, t in g.tensors.items():
        if name.startswith("blk.0."):
            print(f"  {name:36} {t.type_name:6} {t.dims} {t.nbytes} B")


def unpack_i2s(raw: np.ndarray, k: int, n: int) -> np.ndarray:
    """ggml I2_S bytes -> dense int8 [n, k] in {-1,0,+1}.

    The layout is BIT-PLANE MAJOR over 128-element groups, not the sequential
    4-per-byte packing it looks like. Established by feeding one-hot vectors
    through ggml's own kernel and seeing which packed slot answered
    (tests/test_i2s_layout.cc --probe, cosine 1.0000 on every probe):

        element j of a 128-group, with g = j // 32 and i = j % 32,
        lives in byte i at bit-pair (3 - g)

    which is what the AVX2 kernel does: xq8_0 = bits 6-7 (elements 0-31),
    xq8_1 = bits 4-5 (32-63), xq8_2 = bits 2-3 (64-95), xq8_3 = bits 0-1 (96-127).
    The NEON kernel reads 64-element blocks instead and compensates by permuting
    the ACTIVATIONS (I2S_Y_BASE), so the file layout is the same either way.

    Codes are u = w + 1 over {0,1,2}; code 3 is unused.
    """
    if (k * n) % 128:
        raise ValueError("I2_S groups are 128 elements; k*n must be a multiple of 128")
    data = raw[: k * n // 4]
    # [groups, 32] bytes -> 4 bit-planes -> [groups, 4, 32] -> flat element order
    per_group = data.reshape(-1, 32)
    planes = np.stack([(per_group >> s) & 3 for s in (6, 4, 2, 0)], axis=1)
    codes = planes.reshape(-1)
    return (codes.astype(np.int8) - 1).reshape(n, k)


def i2s_scale(raw: np.ndarray, k: int, n: int) -> float:
    """Per-tensor scale: first float of the 32-byte tail."""
    tail = raw[k * n // 4:].tobytes()[:4]
    return float(np.frombuffer(tail, dtype=np.float32)[0])
