"""Export an N-layer Qwen2.5 decoder step with a KV cache, on the ternary custom op.

One decode step: a single token in, updated caches out. The caches enter and leave
through the signature so the engine can ALIAS each pair to one TensorBuffer — the
same trick MossLiteEngine uses, which is what keeps the cache off the host boundary.

Weight tensors are inputs, not constants, because the LiteRT dispatcher will not
hand constant tensors to a custom kernel. For 28 layers that is 196 weight inputs
plus 56 scales plus 56 cache pairs; they are bound once and reused every step.

  python export_decoder.py --layers 2 --ctx 64      # validate the pattern
  python export_decoder.py --layers 28 --ctx 512    # the real thing
"""

import argparse

import numpy as np
import torch

from litert_torch.generative.custom_ops.dynamic_update_slice import dynamic_update_slice as dus

import gguf_read as G
import ternary_op as T

LM = "/home/luigi/VibeASR.cpp/models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf"
N_HEAD, N_KV, HEAD_DIM, ROPE_THETA, EPS = 12, 2, 128, 1_000_000.0, 1e-6


def load_proj(g, name):
    t = g.tensors[name]
    k, n = int(t.dims[0]), int(t.dims[1])
    raw = g.raw(name)
    return G.unpack_i2s(raw, k, n).astype(np.float32), G.i2s_scale(raw, k, n)


def f32(g, name):
    return torch.from_numpy(np.frombuffer(g.raw(name).tobytes(), dtype=np.float32).copy())


def rope(x, pos):
    half = x.shape[-1] // 2
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos.float().reshape(-1, 1) * inv
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


class Decoder(torch.nn.Module):
    """Norm weights and biases stay baked in (they are float and tiny). Only the
    ternary projections come in as arguments, since only they need the custom op."""

    def __init__(self, layers, ctx, meta):
        super().__init__()
        self.L, self.ctx = layers, ctx
        for i, m in enumerate(meta):
            for k, v in m.items():
                self.register_buffer(f"l{i}_{k}", v)

    def forward(self, x, pos, *rest):
        # rest = per layer: qw,qs,kw,ks,vw,vs,ow,os,gw,gs,uw,us,dw,ds then k$i,v$i
        n_w = self.L * 14
        W, caches = rest[:n_w], rest[n_w:]
        k_out, v_out = [], []
        # Causal mask over the cache: only slots already written (< pos+1) count.
        idx = torch.arange(self.ctx).reshape(1, 1, -1)
        mask = (idx <= pos.reshape(-1, 1, 1)).float()
        neg = (1.0 - mask) * -1e30

        for i in range(self.L):
            w = W[i * 14:(i + 1) * 14]
            qw, qs, kw, ks, vw, vs, ow, os_, gw, gs, uw, us, dw, ds = w
            k_cache, v_cache = caches[2 * i], caches[2 * i + 1]

            h = rms(x, getattr(self, f"l{i}_attn_norm"))
            q = torch.ops.voxsum.ternary_matmul(h, qw, qs) + getattr(self, f"l{i}_qb")
            k = torch.ops.voxsum.ternary_matmul(h, kw, ks) + getattr(self, f"l{i}_kb")
            v = torch.ops.voxsum.ternary_matmul(h, vw, vs) + getattr(self, f"l{i}_vb")

            q = rope(q.view(1, N_HEAD, HEAD_DIM).transpose(0, 1), pos).transpose(0, 1)
            k = rope(k.view(1, N_KV, HEAD_DIM).transpose(0, 1), pos).transpose(0, 1)
            v = v.view(1, N_KV, HEAD_DIM)

            # Write this step into the cache at `pos`. dynamic_update_slice touches
            # ONE row; the obvious one-hot scatter
            #     k_cache * (1 - hot) + k * hot
            # is O(ctx) instead, reading and rewriting the whole cache per layer per
            # token. Measured at ctx=64 that cost ~2 ms/layer on top of the block
            # time, and it grows linearly with context.
            # Indices must be 0-D scalars; a shape-[1] tensor is rejected with
            # "operand #2 must be variadic of 0D tensor".
            p0 = pos[0]
            z0 = torch.zeros_like(p0)
            k_new = dus(k_cache, k, (p0, z0, z0))
            v_new = dus(v_cache, v, (p0, z0, z0))
            k_out.append(k_new)
            v_out.append(v_new)

            # GQA without materializing expanded K/V: group the queries instead.
            qg = q.view(1, N_KV, N_HEAD // N_KV, HEAD_DIM)
            att = torch.einsum("tgnd,sgd->gnts", qg, k_new) / (HEAD_DIM ** 0.5)
            att = (att + neg.reshape(1, 1, 1, -1)).softmax(-1)
            o = torch.einsum("gnts,sgd->tgnd", att, v_new).reshape(1, N_HEAD * HEAD_DIM)
            x = x + torch.ops.voxsum.ternary_matmul(o, ow, os_)

            h2 = rms(x, getattr(self, f"l{i}_ffn_norm"))
            gate = torch.ops.voxsum.ternary_matmul(h2, gw, gs)
            up = torch.ops.voxsum.ternary_matmul(h2, uw, us)
            x = x + torch.ops.voxsum.ternary_matmul(
                torch.nn.functional.silu(gate) * up, dw, ds)

        return (x, *k_out, *v_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"decoder_{args.layers}L_{args.ctx}.tflite"

    g = G.Gguf(LM)
    meta, weights = [], []
    for i in range(args.layers):
        p = f"blk.{i}."
        m = {"attn_norm": f32(g, p + "attn_norm.weight"),
             "ffn_norm": f32(g, p + "ffn_norm.weight"),
             "qb": f32(g, p + "attn_q.bias"),
             "kb": f32(g, p + "attn_k.bias"),
             "vb": f32(g, p + "attn_v.bias")}
        meta.append(m)
        for tag, gname in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"),
                           ("o", "attn_output"), ("g", "ffn_gate"),
                           ("u", "ffn_up"), ("d", "ffn_down")):
            dense, sc = load_proj(g, p + gname + ".weight")
            weights.append(T.pack_ternary(torch.from_numpy(dense).to(torch.int8)))
            weights.append(torch.full((dense.shape[0],), float(sc)))
        print(f"  layer {i} loaded", end="\r")
    print(f"loaded {args.layers} layers ({len(weights)} weight tensors)      ")

    dim = 1536
    x = torch.randn(1, dim) * 0.5
    pos = torch.tensor([3])
    caches = []
    for _ in range(args.layers):
        caches.append(torch.zeros(args.ctx, N_KV, HEAD_DIM))
    for _ in range(args.layers):
        caches.append(torch.zeros(args.ctx, N_KV, HEAD_DIM))
    # interleave k,v per layer to match the forward signature
    inter = []
    for i in range(args.layers):
        inter += [caches[i], caches[args.layers + i]]

    mod = Decoder(args.layers, args.ctx, meta).eval()
    sample = (x, pos, *weights, *inter)
    with torch.no_grad():
        y = mod(*sample)
    print(f"eager ok: out rms={y[0].pow(2).mean().sqrt():.6g}, "
          f"{len(y) - 1} cache outputs")

    import litert_torch
    litert_torch.convert(mod, sample).export(out)
    print(f"wrote {out}")

    np.asarray(y[0].numpy()).tofile(f"dec_{args.layers}L_expected.bin")
    names = ["dec_x.bin", "dec_pos.bin"]
    x.numpy().tofile("dec_x.bin")
    pos.numpy().astype(np.int64).tofile("dec_pos.bin")
    for j, t in enumerate(weights):
        t.numpy().tofile(f"dec_w{j:03d}.bin")
        names.append(f"dec_w{j:03d}.bin")
    for j, t in enumerate(inter):
        t.numpy().tofile(f"dec_c{j:03d}.bin")
        names.append(f"dec_c{j:03d}.bin")
    with open(f"dec_{args.layers}L_manifest.txt", "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"wrote {len(names)} input files + manifest")


if __name__ == "__main__":
    main()
