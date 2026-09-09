#!/usr/bin/env python3
"""Convert streaming-1.5B LM (normal BF16, NOT BitNet) to GGUF F16 via staging dir.

- Reads ./models-pt safetensors, writes LM-only (renamed) safetensors into
  models-streaming/staging/ + flattened config + tokenizer files.
- Calls convert-ms-to-gguf-bitnet.py on staging (plain Qwen2 path, no ternary).
- Works around the add_bos_token duplicate-key bug by stripping the key in staging.
No in-place modification of models-pt (low disk overhead).
"""
import sys, os, json, shutil, subprocess
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

STAGING = Path("models-streaming/staging")
OUT = Path("models-streaming/streaming-lm-f16.gguf")

def run_command(cmd, env=None):
    print(f"Executing: {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=True, env=env)

def main():
    src = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("models-pt").resolve()
    utils_dir = Path(__file__).parent.resolve()
    project_root = utils_dir.parent
    convert_script = utils_dir / "convert-ms-to-gguf-bitnet.py"

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    # 1. Collect LM tensors from all shards, split into 2 staging files (~1.5GB each)
    single = src / "model.safetensors"
    inputs = [single] if single.is_file() else sorted(src.glob("model-*-of-*.safetensors"))
    buckets = [{}, {}]
    for f in inputs:
        with safe_open(str(f), framework="pt") as sf:
            for name in sf.keys():
                t = sf.get_tensor(name)
                if name.startswith("model.language_model."):
                    out_name = "model." + name[len("model.language_model."):]
                elif name == "lm_head.weight":
                    out_name = name
                else:
                    continue
                # shard by layer index hash for balance
                buckets[hash(out_name) % 2][out_name] = t
    for i, b in enumerate(buckets):
        p = STAGING / f"model-0000{i+1}-of-00002.safetensors"
        save_file(b, str(p))
        print(f"staging {p.name}: {len(b)} tensors")
    # index file
    wmap = {}
    for i, b in enumerate(buckets):
        fn = f"model-0000{i+1}-of-00002.safetensors"
        for k in b:
            wmap[k] = fn
    with open(STAGING / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": sum(t.nelement() * 2 for b in buckets for t in b.values())},
                   "weight_map": wmap}, f)
    # 2. Flattened decoder config
    with open(src / "config.json") as f:
        cfg = json.load(f)
    with open(STAGING / "config.json", "w") as f:
        json.dump(cfg["decoder_config"], f, indent=2)
    # 3. Tokenizer files (strip add_bos_token for dup-key workaround)
    for fn in ["vocab.json", "merges.txt", "tokenizer.json", "added_tokens.json",
               "special_tokens_map.json", "tokenizer_config.json"]:
        s = src / fn
        if s.exists():
            shutil.copy2(str(s), str(STAGING / fn))
    with open(STAGING / "tokenizer_config.json") as f:
        tokcfg = json.load(f)
    tokcfg.pop("add_bos_token", None)
    with open(STAGING / "tokenizer_config.json", "w") as f:
        json.dump(tokcfg, f, indent=2)
    print("Staging ready.")
    # 4. Convert
    gguf_py = str((project_root / "3rdparty" / "llama.cpp" / "gguf-py").resolve())
    env = os.environ.copy()
    env["PYTHONPATH"] = gguf_py + os.pathsep + env.get("PYTHONPATH", "")
    run_command([sys.executable, str(convert_script), str(STAGING),
                 "--vocab-type", "bpe", "--outtype", "f16",
                 "--concurrency", "1", "--outfile", str(OUT.resolve()),
                 "--pad-vocab", "--skip-unknown"], env=env)
    print(f"DONE: {OUT} ({OUT.stat().st_size/2**30:.2f} GiB)")

if __name__ == "__main__":
    main()
