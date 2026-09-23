#!/usr/bin/env python3
"""
Conv-int8 converter (hybrid): Python for structure + C++ quantization helper.
Creates a quantized VAE model with 14 non-dw conv weights padded to K=32 and
quantized to Q4_0_4_4. DW convs stay F16. FFN already Q4_0_4_4.
"""
import gguf
import numpy as np
import sys
import os
import subprocess
import tempfile
import shutil

TARGET_NAMES = {
    # Only the first downsample layer (K=8, IC=1) needs padding to K=32 to make K*IC=32
    # Other downsample layers already have K*IC multiple of 32
    'acoustic.downsample_layers.0.0.conv.conv.weight',
    'semantic.downsample_layers.0.0.conv.conv.weight',
}

# True kernel sizes for all 14 downsample convs (acoustic 0-6, semantic 0-6)
# Only the first layer (index 0 and 7) are padded to K=32 in the converter.
# The metadata still contains all 14 true K sizes for the VAE's causal padding logic.
TRUE_K_SIZES = [8, 4, 4, 8, 16, 16, 16, 8, 4, 4, 8, 16, 16, 16]

QUANTIZE_HELPER = os.path.join(os.path.dirname(__file__), 'quantize_q4_0_4_4')

def quantize_f16_to_q4_0_4_4(f16_data, n_rows, n_per_row):
    """Quantize F16 data to Q4_0_4_4 using C++ helper.
    f16_data: numpy array of float16 with shape (n_rows * n_per_row,)
    Returns: bytes of Q4_0_4_4 data
    """
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f_in:
        f16_data.tofile(f_in)
        in_fname = f_in.name
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f_out:
        out_fname = f_out.name

    try:
        result = subprocess.run([
            QUANTIZE_HELPER, in_fname, out_fname,
            str(n_rows), str(n_per_row)
        ], capture_output=True, text=True, check=True)
        print(f"  Quantized: {result.stdout.strip()}")
        with open(out_fname, 'rb') as f:
            return f.read()
    except subprocess.CalledProcessError as e:
        print(f"  Quantization failed: {e.stderr}")
        raise
    finally:
        os.unlink(in_fname)
        if os.path.exists(out_fname):
            os.unlink(out_fname)

def to_native(v):
    """Convert numpy types to Python native types."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return [to_native(x) for x in v]
    if isinstance(v, dict):
        return {k: to_native(v2) for k, v2 in v.items()}
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='replace')
    return v

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_f16.gguf> <output_convint8.gguf>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    print(f"Reading {in_path}...")
    reader = gguf.GGUFReader(in_path)

    print(f"Writing {out_path}...")
    writer = gguf.GGUFWriter(out_path, "vibeasr-vae", use_temp_file=True)

    # Parse metadata properly from GGUF fields
    # Each field represents one key-value pair with parts:
    # [key_len, key_bytes, value_type, value_len, value_data]
    meta = {}
    for f in reader.fields.values():
        parts = f.parts
        if len(parts) >= 5:
            # Extract key
            key_bytes = parts[1]
            if hasattr(key_bytes, 'tobytes'):
                key = key_bytes.tobytes().decode('utf-8', errors='replace')
            elif hasattr(key_bytes, 'tobytes'):
                key = key_bytes.tobytes().decode('utf-8', errors='replace')
            else:
                key = bytes(key_bytes).decode('utf-8', errors='replace')
            
            # Extract value
            val = parts[4]
            # Convert value based on its type
            if val.dtype == np.uint8 and val.ndim == 1:
                # String value
                v = val.tobytes().decode('utf-8', errors='replace')
            elif hasattr(val, 'item'):
                v = val.item()
            elif hasattr(val, 'tolist'):
                v = val.tolist()
            else:
                v = val
            
            meta[key] = v

    for k, v in meta.items():
        v = to_native(v)
        # Extra safety: ensure ALL numpy types are converted to Python native
        if hasattr(v, 'item') and callable(v.item):
            v = v.item()
        elif hasattr(v, 'tolist') and callable(v.tolist):
            v = v.tolist()
        try:
            if isinstance(v, str):
                writer.add_string(k, v)
            elif isinstance(v, int):
                writer.add_uint32(k, v)
            elif isinstance(v, float):
                writer.add_float32(k, v)
            elif isinstance(v, (list, tuple)):
                writer.add_array(k, v)
            elif isinstance(v, bool):
                writer.add_bool(k, v)
            else:
                print(f"  Skipping metadata {k}: type {type(v)}")
        except Exception as e:
            print(f"  Warning: failed to add metadata {k}: {e}")

    # Add vae.kernel_size metadata (all 14 true kernel sizes for VAE's causal padding)
    writer.add_array("vae.kernel_size", TRUE_K_SIZES)

    # Process tensors
    print("Processing tensors...")
    # Map from tensor name to true_K for the 2 padded tensors
    TRUE_K_MAP = {
        'acoustic.downsample_layers.0.0.conv.conv.weight': 8,
        'semantic.downsample_layers.0.0.conv.conv.weight': 8,
    }
    for t in reader.tensors:
        name = t.name
        data = t.data  # numpy memmap
        logical_shape = [int(s) for s in t.shape]  # [K, IC, OC]
        tensor_type = t.tensor_type  # 1 = F16

        if name in TARGET_NAMES:
            # Target conv weight: pad K to 32, quantize to Q4_0_4_4
            K_orig = int(logical_shape[0])
            K_pad = 32
            IC = int(logical_shape[1])
            OC = int(logical_shape[2])

            true_K = TRUE_K_MAP[name]

            print(f"  {name}: F16 [K={K_orig}, IC={IC}, OC={OC}] -> pad K={K_orig}->{K_pad} -> Q4_0_4_4")

            # Data layout in GGUF: [OC, IC, K] (K fastest)
            # Pad K dimension (last axis) from K_orig to 32
            if K_orig < K_pad:
                pad_width = K_pad - K_orig
                data = np.pad(data, ((0,0), (0,0), (0, pad_width)), mode='constant')
                new_shape = [K_pad, IC, OC]
                print(f"    Padded data shape: {data.shape}")
            else:
                new_shape = logical_shape

            # Quantize to Q4_0_4_4
            # n_rows = OC, n_per_row = 32 * IC (multiple of 32)
            n_rows = OC
            n_per_row = 32 * IC

            # Flatten to [n_rows, n_per_row] for quantization (row-major)
            # data is currently [OC, IC, 32] in Fortran order
            # We need to reshape to [OC, 32*IC] row-major
            data_reshaped = data.reshape(OC, -1).astype(np.float16)
            assert data_reshaped.shape == (n_rows, n_per_row), f"Shape mismatch: {data_reshaped.shape} vs ({n_rows}, {n_per_row})"

            print(f"    Quantizing: n_rows={n_rows}, n_per_row={n_per_row}")
            q_bytes = quantize_f16_to_q4_0_4_4(data_reshaped.flatten(), n_rows, n_per_row)

            # Add to writer as Q4_0_4_4 tensor
            # The data must be a numpy array for add_tensor
            # For quantized tensors, the 'byte shape' is [n_rows, bytes_per_row]
            # Each block of 32 elements becomes 18 bytes in Q4_0_4_4
            blocks_per_row = n_per_row // 32
            bytes_per_row = blocks_per_row * 18  # 18 bytes per 32-element block
            q_array = np.frombuffer(q_bytes, dtype=np.uint8).reshape(n_rows, bytes_per_row)
            writer.add_tensor(name, q_array, raw_dtype=gguf.constants.GGMLQuantizationType.Q4_0_4_4)
            print(f"    Added Q4_0_4_4 tensor: {len(q_bytes)} bytes")

        else:
            # Non-target tensor: copy as-is
            new_shape = logical_shape
            writer.add_tensor(name, data, raw_shape=new_shape, raw_dtype=tensor_type)

    print("Writing file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Done: {out_path}")

if __name__ == "__main__":
    main()