"""
Solver → ONNX Compiler

Pipeline:
  1. Parse solver source into op list
  2. Expand higher-order calls where possible
  3. Build traceable PyTorch model (crop → ops → pad)
  4. Export to ONNX
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from solver_parser import parse_solver, resolve_constants, extract_solver, CONSTANTS
from primitives import (
    PRIM_DISPATCH, PRIM_NEEDS_ARGS, BINARY_OPS,
    oh_decode, oh_encode, NUM_COLORS, CANVAS,
)

# ============================================================================
# Model: crop to actual size → run ops → pad to 30x30
# ============================================================================

class SolverModel(nn.Module):
    def __init__(self, grid_ops: List[Dict[str, Any]], h_in: int, w_in: int):
        super().__init__()
        self.grid_ops = grid_ops
        self.h_in = h_in
        self.w_in = w_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Crop to actual input size
        x = x[:, :, :self.h_in, :self.w_in]

        vars = {"I": x}

        for op in self.grid_ops:
            func = op["func"]
            args = op["args"]
            var = op["var"]

            fn = PRIM_DISPATCH.get(func)
            if fn is None:
                raise ValueError(f"Unknown primitive: {func}")

            needs = PRIM_NEEDS_ARGS.get(func, [])
            n_tensor = 2 if func in BINARY_OPS else 1

            # Resolve args
            resolved = []
            for a in args:
                if isinstance(a, str) and a in vars:
                    resolved.append(vars[a])
                elif isinstance(a, str) and a in CONSTANTS:
                    resolved.append(CONSTANTS[a])
                elif isinstance(a, (int, float)):
                    resolved.append(a)
                elif isinstance(a, tuple):
                    resolved.append(a)
                else:
                    resolved.append(a)

            tensor_args = resolved[:n_tensor]
            extra_args = resolved[n_tensor:]

            kwargs = {}
            for i, name in enumerate(needs):
                if i < len(extra_args):
                    val = extra_args[i]
                    if isinstance(val, str) and val in CONSTANTS:
                        val = CONSTANTS[val]
                    kwargs[name] = int(val) if isinstance(val, (int, float)) else val

            result = fn(*tensor_args, **kwargs)
            vars[var] = result

        # Get final output
        out = vars.get("O", vars[list(vars.keys())[-1]])

        # Pad to 30x30
        if out.shape[2] < CANVAS or out.shape[3] < CANVAS:
            padded = torch.zeros(1, NUM_COLORS, CANVAS, CANVAS, device=out.device, dtype=out.dtype)
            padded[:, 0, :, :] = 1.0  # background
            padded[:, :, :out.shape[2], :out.shape[3]] = out
            out = padded
        elif out.shape[2] > CANVAS or out.shape[3] > CANVAS:
            out = out[:, :, :CANVAS, :CANVAS]

        return out


# ============================================================================
# Solver → flat grid ops compiler
# ============================================================================

def compile_solver(solver_source: str, task_num: int,
                   output_dir: str = ".",
                   h_in: int = 30, w_in: int = 30) -> Optional[str]:
    """Compile solver to ONNX."""
    ops = parse_solver(solver_source)
    if not ops:
        return None

    grid_ops = expand_ops(ops)
    if grid_ops is None:
        return None

    grid_ops = fix_crop_args(grid_ops)

    if not grid_ops:
        return None

    model = SolverModel(grid_ops, h_in, w_in)
    dummy = torch.zeros(1, NUM_COLORS, CANVAS, CANVAS)

    try:
        model.eval()
        with torch.no_grad():
            out = model(dummy)
            assert out.shape == (1, NUM_COLORS, CANVAS, CANVAS)

        onnx_path = str(Path(output_dir) / f"task{task_num:03d}.onnx")
        torch.onnx.export(
            model, dummy, onnx_path,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
        )
        # Fix: inline external data so the .onnx is self-contained
        try:
            import onnx
            m = onnx.load(onnx_path)
            onnx.save_model(m, onnx_path)
            data_file = Path(onnx_path).with_suffix(".onnx.data")
            if data_file.exists():
                data_file.unlink()
        except Exception:
            pass
        return onnx_path
    except Exception as e:
        print(f"  task{task_num:03d}: EXPORT FAILED: {e}")
        return None


def expand_ops(ops: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Expand higher-order calls into flat grid-level ops."""
    vars = {}
    grid_ops = []

    for op in ops:
        func = op["func"]
        var = op["var"]
        args = op["args"]

        if func == "identity":
            ref = args[0]
            if ref in vars:
                vars[var] = vars[ref]
            else:
                vars[var] = {"type": "grid", "var": ref}
            continue

        if func in PRIM_DISPATCH:
            resolved = []
            for a in args:
                if a in vars and vars[a]["type"] == "grid":
                    resolved.append(vars[a]["var"])
                elif a in CONSTANTS:
                    resolved.append(a)
                elif a in PRIM_DISPATCH:
                    resolved.append(a)
                else:
                    resolved.append(a)

            grid_ops.append({"var": var, "func": func, "args": resolved})
            vars[var] = {"type": "grid", "var": var}
            continue

        # Higher-order: skip (can't expand to grid ops)
        return None

    # Check if final output is defined
    last_var = ops[-1]["var"] if ops else None
    if last_var and last_var not in vars:
        return None

    return grid_ops


def fix_crop_args(ops: list) -> list:
    """Fix crop operations that take tuple args."""
    TUPLE_CONSTANTS = {
        "ZERO_BY_TWO": (0, 2), "TWO_BY_ZERO": (2, 0),
        "TWO_BY_TWO": (2, 2), "THREE_BY_THREE": (3, 3),
        "UNITY": (1, 1), "NEG_UNITY": (-1, -1),
        "ORIGIN": (0, 0),
    }

    result = []
    for op in ops:
        if op["func"] == "crop" and len(op["args"]) == 3:
            grid_arg = op["args"][0]
            start = op["args"][1]
            dims = op["args"][2]

            if isinstance(start, str) and start in TUPLE_CONSTANTS:
                start = TUPLE_CONSTANTS[start]
            if isinstance(dims, str) and dims in TUPLE_CONSTANTS:
                dims = TUPLE_CONSTANTS[dims]

            if isinstance(start, tuple) and isinstance(dims, tuple):
                top, left = start
                h, w = dims
                result.append({"var": op["var"], "func": "crop",
                               "args": [grid_arg, top, left, h, w]})
            else:
                pass
        else:
            result.append(op)

    return result


# ============================================================================
# Compile + verify
# ============================================================================

def compile_and_verify(solver_source: str, task_num: int,
                       task_data: dict, output_dir: str = ".",
                       h_in: int = 30, w_in: int = 30) -> Tuple[Optional[str], bool]:
    """Compile and verify on training examples."""
    onnx_path = compile_solver(solver_source, task_num, output_dir, h_in, w_in)
    if onnx_path is None:
        return None, False

    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name

        all_match = True
        for example in task_data.get("train", []):
            input_oh = _encode_grid(example["input"])
            output_oh = _encode_grid(example["output"])

            onnx_out = sess.run(None, {input_name: input_oh})[0]
            match = (onnx_out.argmax(axis=1) == output_oh.argmax(axis=1)).all()
            if not match:
                all_match = False

        return onnx_path, all_match
    except Exception as e:
        print(f"  task{task_num:03d}: VERIFY FAILED: {e}")
        return onnx_path, False


def _encode_grid(grid: list):
    """Encode Python grid to one-hot numpy array."""
    import numpy as np
    arr = np.array(grid, dtype=np.int64)
    H, W = arr.shape
    oh = np.zeros((1, NUM_COLORS, CANVAS, CANVAS), dtype=np.float32)
    for c in range(NUM_COLORS):
        oh[0, c, :H, :W] = (arr == c).astype(np.float32)
    return oh


# ============================================================================
# Analysis
# ============================================================================

def analyze_tasks():
    """Analyze which tasks can be compiled."""
    from solver_parser import extract_all_solvers

    solvers = extract_all_solvers()
    compilable = []
    needs_objects = []

    for name, source in sorted(solvers.items()):
        ops = parse_solver(source)
        grid_ops = expand_ops(ops)

        if grid_ops is not None:
            tn = int(name.split("_")[1], 16) if "_" in name else 0
            compilable.append(tn)
        else:
            tn = int(name.split("_")[1], 16) if "_" in name else 0
            needs_objects.append(tn)

    print(f"Compilable: {len(compilable)}")
    print(f"Needs objects/higher-order: {len(needs_objects)}")
    return compilable, needs_objects


if __name__ == "__main__":
    analyze_tasks()
