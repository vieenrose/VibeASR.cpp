#!/usr/bin/env python3
"""
Conv-int8 converter: pad the 14 non-dw conv weights from K in {4,8,16} to K=32
with trailing zeros (back-padding), then write a new F16 GGUF.
Exp684 validated: back-zero padding is exact; im2col row length becomes 32*IC (multiple of 32).
"""
import gguf
import numpy as np
import sys
import os

TARGET_NAMES = {
    'acoustic.downsample_layers.0.0.conv.conv.weight',
    'acoustic.downsample_layers.1.0.conv.conv.weight',
    'acoustic.downsample_layers.2.0.conv.conv.weight',
    'acoustic.downsample_layers.3.0.conv.conv.weight',
    'acoustic.downsample_layers.4.0.conv.conv.weight',
    'acoustic.downsample_layers.5.0.conv.conv.weight',
    'acoustic.downsample_layers.6.0.conv.conv.weight',
    'semantic.downsample_layers.0.0.conv.conv.weight',
    'semantic.downsample_layers.1.0.conv.conv.weight',
    'semantic.downsample_layers.2.0.conv.conv.weight',
    'semantic.downsample_layers.3.0.conv.conv.weight',
    'semantic.downsample_layers.4.0.conv.conv.weight',
    'semantic.downsample_layers.5.0.conv.conv.weight',
    'semantic.downsample_layers.6.0.conv.conv.weight',
}

# True kernel sizes for the 14 downsample convs (for vae.kernel_size metadata)
# Order: acoustic.0..6, semantic.0..6
TRUE_K_SIZES = [8, 4, 4, 8, 16, 16, 16, 8, 4, 4, 8, 16, 16, 16]

def to_native(v):
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
    return v

def decode_string(arr):
    """Decode uint8 array to string."""
    return bytes(arr.tolist()).decode('utf-8')

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_f16.gguf> <output_f16_padded.gguf>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    print(f"Reading {in_path}...")
    reader = gguf.GGUFReader(in_path)

    print(f"Writing {out_path}...")
    writer = gguf.GGUFWriter(out_path, "vibeasr-vae", use_temp_file=True)

    # Add required metadata manually (from the F16 model's fields)
    writer.add_uint32("general.architecture", 11)  # string length for "vibeasr-vae"
    writer.add_string("general.architecture", "vibeasr-vae")
    writer.add_string("general.name", "VibeASR VAE Encoder")
    writer.add_uint32("general.file_type", 0)  # F16
    writer.add_uint32("general.version", 1)

    writer.add_uint32("vae.acoustic.output_dim", 64)
    writer.add_array("vae.acoustic.stage_depths", [3, 3, 3, 3, 3, 3, 8])
    writer.add_array("vae.acoustic.downsample_strides", [8, 5, 5, 4, 2, 2])
    writer.add_uint32("vae.semantic.output_dim", 128)
    writer.add_array("vae.semantic.stage_depths", [3, 3, 3, 3, 3, 3, 8])
    writer.add_array("vae.semantic.downsample_strides", [8, 5, 5, 4, 2, 2])
    writer.add_float32("vae.rms_norm_eps", 1e-6)
    writer.add_array("vae.kernel_size", TRUE_K_SIZES)
    writer.add_float32("vae.layer_scale_init_value", 1e-6)

    # Process tensors
    print("Processing tensors...")
    for t in reader.tensors:
        name = t.name
        data = t.data  # numpy memmap, layout [OC, IC, K]
        logical_shape = [int(s) for s in t.shape]  # [K, IC, OC]

        if name in TARGET_NAMES:
            K_orig = int(logical_shape[0])
            K_pad = 32
            if K_orig < K_pad:
                pad_width = K_pad - K_orig
                data = np.pad(data, ((0,0), (0,0), (0, pad_width)), mode='constant')
                new_shape = [K_pad, logical_shape[1], logical_shape[2]]
                print(f"  Padded {name}: K={K_orig}->{K_pad}, data shape {data.shape}")
            else:
                new_shape = logical_shape
        else:
            new_shape = logical_shape

        data = np.ascontiguousarray(data, dtype=np.float16)
        writer.add_tensor(name, data, raw_shape=new_shape, raw_dtype=gguf.GGMLQuantizationType.F16)

    print("Writing file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Done: {out_path}")

if __name__ == "__main__":
    main()