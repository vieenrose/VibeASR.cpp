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
