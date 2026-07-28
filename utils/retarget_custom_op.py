"""Rewrite STABLEHLO_CUSTOM_CALL into a TFLite CUSTOM op.

litert_torch lowers stablehlo.custom_call to the BUILTIN opcode
STABLEHLO_CUSTOM_CALL, which the interpreter has no kernel for:

    ValueError: Didn't find op for builtin opcode 'STABLEHLO_CUSTOM_CALL'

But LiteRtAddCustomOpKernelOption dispatches on a CUSTOM opcode's `custom_code`
string, so the graph has to carry a genuine CUSTOM op instead. There is no
converter flag exposed for that (no tfl dialect in the Python bindings), so this
edits the flatbuffer directly — a two-field change on the operator code, which is
far less fragile than it sounds and keeps us off converter internals.

    python retarget_custom_op.py in.tflite out.tflite voxsum.ternary_matmul
"""

import sys

import flatbuffers
from litert_converter import schema_py_generated as schema

# Read from the schema rather than hardcoded: these enum values shift between
# schema revisions (STABLEHLO_CUSTOM_CALL is 173 here, and 178 is
# STABLEHLO_EXPONENTIAL — a hardcoded guess silently rewrites the wrong op).
STABLEHLO_CUSTOM_CALL = schema.BuiltinOperator.STABLEHLO_CUSTOM_CALL
CUSTOM = schema.BuiltinOperator.CUSTOM


def retarget(in_path: str, out_path: str, custom_code: str, version: int = 1) -> int:
    with open(in_path, "rb") as f:
        model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(f.read()), 0))

    changed = 0
    for oc in model.operatorCodes:
        # deprecatedBuiltinCode caps at 127; anything above lives in builtinCode.
        builtin = max(oc.builtinCode or 0, oc.deprecatedBuiltinCode or 0)
        if builtin != STABLEHLO_CUSTOM_CALL:
            continue
        oc.builtinCode = CUSTOM
        oc.deprecatedBuiltinCode = CUSTOM
        oc.customCode = custom_code
        oc.version = version
        changed += 1

    if changed:
        # A CUSTOM op carries customOptions, not builtinOptions. Any stablehlo
        # options left attached would be read as garbage by the custom kernel.
        code_is_custom = {
            i for i, oc in enumerate(model.operatorCodes)
            if (oc.customCode or "") == custom_code
        }
        for sub in model.subgraphs:
            for op in sub.operators:
                if op.opcodeIndex in code_is_custom:
                    op.builtinOptions = None
                    op.builtinOptionsType = 0
                    if not op.customOptions:
                        op.customOptions = []
                    op.customOptionsFormat = 0

    b = flatbuffers.Builder(1024)
    b.Finish(model.Pack(b), file_identifier=b"TFL3")
    with open(out_path, "wb") as f:
        f.write(b.Output())
    return changed


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else "voxsum.ternary_matmul"
    n = retarget(src, dst, name)
    print(f"retargeted {n} opcode(s) to CUSTOM '{name}' -> {dst}")
