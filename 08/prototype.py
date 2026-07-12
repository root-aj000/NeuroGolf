"""
NeuroGolf Hybrid Prototype: PyTorch tracing → ONNX

Design:
  1. Primitives are @staticmethod methods on `Primitives` class
     operating on (1, 10, H, W) one-hot float tensors.
  2. `NeuroGolfModel(nn.Module)` chains ops in its forward().
  3. For each task: build model → trace → export ONNX → verify.
  4. Solver source is parsed into a list of (op, args) steps.
"""

import json, re, ast, sys, os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_COLORS = 10
CANVAS = 30


# ============================================================================
# Primitives: pure torch ops on (1, 10, H, W) one-hot tensors
# ============================================================================

class Primitives:
    """All primitives as static torch functions.

    Every function takes (tensor, ...) where tensor is (1,10,H,W) float.
    Returns (1,10,H',W') float.
    """

    # --- Spatial transforms (H,W preserve) ---

    @staticmethod
    def rot90(x: torch.Tensor) -> torch.Tensor:
        """90° clockwise: transpose then flip left-right."""
        # (1,10,H,W) → (1,10,W,H) → flip W
        return torch.flip(torch.transpose(x, 2, 3), [3])

    @staticmethod
    def rot180(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, [2, 3])

    @staticmethod
    def rot270(x: torch.Tensor) -> torch.Tensor:
        """270° clockwise = 90° counter-clockwise."""
        return torch.transpose(torch.flip(x, [3]), 2, 3)

    @staticmethod
    def hmirror(x: torch.Tensor) -> torch.Tensor:
        """Flip rows (horizontal mirror = flip along axis 2)."""
        return torch.flip(x, [2])

    @staticmethod
    def vmirror(x: torch.Tensor) -> torch.Tensor:
        """Flip columns (vertical mirror = flip along axis 3)."""
        return torch.flip(x, [3])

    @staticmethod
    def dmirror(x: torch.Tensor) -> torch.Tensor:
        """Transpose (main diagonal mirror)."""
        return torch.transpose(x, 2, 3)

    @staticmethod
    def cmirror(x: torch.Tensor) -> torch.Tensor:
        """Anti-diagonal mirror: flip rows → transpose → flip rows."""
        return torch.flip(torch.transpose(torch.flip(x, [2]), 2, 3), [2])

    # --- Cropping (output smaller) ---

    @staticmethod
    def crop(x: torch.Tensor, top: int, left: int, h: int, w: int) -> torch.Tensor:
        return x[:, :, top:top+h, left:left+w]

    @staticmethod
    def trim(x: torch.Tensor) -> torch.Tensor:
        return x[:, :, 1:-1, 1:-1]

    @staticmethod
    def tophalf(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[2]
        return x[:, :, :h//2, :]

    @staticmethod
    def bottomhalf(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[2]
        return x[:, :, h//2:, :]

    @staticmethod
    def lefthalf(x: torch.Tensor) -> torch.Tensor:
        w = x.shape[3]
        return x[:, :, :, :w//2]

    @staticmethod
    def righthalf(x: torch.Tensor) -> torch.Tensor:
        w = x.shape[3]
        return x[:, :, :, w//2:]

    # --- Concatenation (output larger) ---

    @staticmethod
    def hconcat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Concatenate along width (axis=3)."""
        return torch.cat([a, b], dim=3)

    @staticmethod
    def vconcat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Concatenate along height (axis=2)."""
        return torch.cat([a, b], dim=2)

    # --- Upscaling ---

    @staticmethod
    def hupscale(x: torch.Tensor, factor: int) -> torch.Tensor:
        """Repeat each column factor times horizontally."""
        # (1,10,H,W) → (1,10,H,W,1) → tile → (1,10,H,W*factor)
        r1 = x.unsqueeze(4)                          # (1,10,H,W,1)
        tiled = r1.repeat(1, 1, 1, 1, factor)        # (1,10,H,W,factor)
        B, C, H, W = x.shape
        return tiled.reshape(B, C, H, W * factor)

    @staticmethod
    def vupscale(x: torch.Tensor, factor: int) -> torch.Tensor:
        """Repeat each row factor times vertically."""
        r1 = x.unsqueeze(3)                          # (1,10,H,1,W)
        tiled = r1.repeat(1, 1, 1, factor, 1)        # (1,10,H,factor,W)
        B, C, H, W = x.shape
        return tiled.reshape(B, C, H * factor, W)

    @staticmethod
    def upscale(x: torch.Tensor, factor: int) -> torch.Tensor:
        """Upscale both dimensions (pixel-repeat)."""
        r1 = x.unsqueeze(3).unsqueeze(5)             # (1,10,H,1,W,1)
        tiled = r1.repeat(1, 1, 1, factor, 1, factor) # (1,10,H,factor,W,factor)
        B, C, H, W = x.shape
        return tiled.reshape(B, C, H * factor, W * factor)

    @staticmethod
    def downscale(x: torch.Tensor, factor: int) -> torch.Tensor:
        """Nearest-neighbor downscale (take every factor-th pixel)."""
        return x[:, :, ::factor, ::factor]

    # --- Value operations (decode→op→encode) ---

    @staticmethod
    def _decode(x: torch.Tensor) -> torch.Tensor:
        """One-hot (1,10,H,W) → scalar (1,1,H,W) via argmax."""
        return x.argmax(dim=1, keepdim=True).float()

    @staticmethod
    def _encode(raw: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Scalar (1,1,H,W) → one-hot (1,10,H,W)."""
        # raw contains values 0..9 as float
        rng = torch.arange(NUM_COLORS, dtype=torch.float32).view(1, NUM_COLORS, 1, 1)
        return (raw == rng).float()

    @staticmethod
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h, w = a.shape[2], a.shape[3]
        d1 = Primitives._decode(a)
        d2 = Primitives._decode(b)
        return Primitives._encode((d1 + d2) % 10, h, w)

    @staticmethod
    def subtract(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h, w = a.shape[2], a.shape[3]
        d1 = Primitives._decode(a)
        d2 = Primitives._decode(b)
        return Primitives._encode((d1 - d2) % 10, h, w)

    @staticmethod
    def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h, w = a.shape[2], a.shape[3]
        d1 = Primitives._decode(a)
        d2 = Primitives._decode(b)
        return Primitives._encode((d1 * d2) % 10, h, w)

    @staticmethod
    def increment(x: torch.Tensor, delta: int = 1) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        d = Primitives._decode(x)
        return Primitives._encode((d + delta) % 10, h, w)

    @staticmethod
    def decrement(x: torch.Tensor, delta: int = 1) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        d = Primitives._decode(x)
        return Primitives._encode((d - delta) % 10, h, w)

    @staticmethod
    def double(x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        d = Primitives._decode(x)
        return Primitives._encode((d * 2) % 10, h, w)

    @staticmethod
    def minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h, w = a.shape[2], a.shape[3]
        d1 = Primitives._decode(a)
        d2 = Primitives._decode(b)
        return Primitives._encode(torch.min(d1, d2), h, w)

    @staticmethod
    def maximum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h, w = a.shape[2], a.shape[3]
        d1 = Primitives._decode(a)
        d2 = Primitives._decode(b)
        return Primitives._encode(torch.max(d1, d2), h, w)

    # --- Comparison (output: one-hot boolean) ---

    @staticmethod
    def equality(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h, w = a.shape[2], a.shape[3]
        d1 = Primitives._decode(a)
        d2 = Primitives._decode(b)
        eq = (d1 == d2).float()
        return Primitives._encode(eq, h, w)

    @staticmethod
    def greater(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h, w = a.shape[2], a.shape[3]
        d1 = Primitives._decode(a)
        d2 = Primitives._decode(b)
        gt = (d1 > d2).float()
        return Primitives._encode(gt, h, w)

    @staticmethod
    def less(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h, w = a.shape[2], a.shape[3]
        d1 = Primitives._decode(a)
        d2 = Primitives._decode(b)
        lt = (d1 < d2).float()
        return Primitives._encode(lt, h, w)

    # --- Color ops ---

    @staticmethod
    def replace(x: torch.Tensor, old_color: int, new_color: int) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        d = Primitives._decode(x)
        result = torch.where(d == old_color, float(new_color), d)
        return Primitives._encode(result, h, w)

    @staticmethod
    def cellwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Keep value where both agree, else 0."""
        h, w = a.shape[2], a.shape[3]
        d1 = Primitives._decode(a)
        d2 = Primitives._decode(b)
        result = torch.where(d1 == d2, d1, torch.zeros_like(d1))
        return Primitives._encode(result, h, w)

    @staticmethod
    def fill_masked(x: torch.Tensor, mask: torch.Tensor, color: int) -> torch.Tensor:
        """Fill positions where mask channel 1 > 0 with given color."""
        h, w = x.shape[2], x.shape[3]
        m = mask[:, 1:2, :, :]  # (1,1,H,W) — foreground channel
        bg = x * (1 - m)
        fg_color = torch.zeros_like(x)
        fg_color[:, color] = 1.0
        fg = fg_color * m
        return bg + fg

    @staticmethod
    def paint_masked(x: torch.Tensor, mask: torch.Tensor, color: int) -> torch.Tensor:
        return Primitives.fill_masked(x, mask, color)

    @staticmethod
    def ofcolor(x: torch.Tensor, color: int) -> torch.Tensor:
        """Returns one-hot mask where channel 1 = pixels matching color."""
        h, w = x.shape[2], x.shape[3]
        d = Primitives._decode(x)
        match = (d == color).float()  # (1,1,H,W)
        out = torch.zeros(1, NUM_COLORS, h, w, device=x.device, dtype=x.dtype)
        out[:, 0] = 1.0 - match.squeeze(1)
        out[:, 1] = match.squeeze(1)
        return out

    # --- Canvas ---

    @staticmethod
    def canvas(color: int, h: int, w: int, device='cpu') -> torch.Tensor:
        out = torch.zeros(1, NUM_COLORS, h, w, device=device)
        out[:, color] = 1.0
        return out

    # --- Merge (element-wise max on decoded) ---

    @staticmethod
    def merge(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h, w = a.shape[2], a.shape[3]
        d1 = Primitives._decode(a)
        d2 = Primitives._decode(b)
        return Primitives._encode(torch.max(d1, d2), h, w)

    # --- Switch (swap two colors) ---

    @staticmethod
    def switch(x: torch.Tensor, a: int, b: int) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        d = Primitives._decode(x)
        result = torch.where(d == a, float(b), torch.where(d == b, float(a), d))
        return Primitives._encode(result, h, w)

    # --- Upscale with separate factors ---

    @staticmethod
    def upscale_hv(x: torch.Tensor, h_factor: int, w_factor: int) -> torch.Tensor:
        """Upscale with separate horizontal and vertical factors."""
        r1 = x.unsqueeze(2).unsqueeze(4)
        tiled = r1.repeat(1, 1, h_factor, 1, w_factor, 1)
        return tiled.reshape(1, x.shape[1], x.shape[2] * h_factor, x.shape[3] * w_factor)


# ============================================================================
# NeuroGolfModel: a traceable nn.Module
# ============================================================================

class NeuroGolfModel(nn.Module):
    """A model that chains primitives. Built dynamically per-task.

    Usage:
        model = NeuroGolfModel(operations=[
            ('hupscale', {'factor': 3}),
            ('vupscale', {'factor': 3}),
            ('cellwise', {}),
        ])
        output = model(input_tensor)
    """

    def __init__(self, operations: List[Tuple[str, Dict[str, Any]]]):
        super().__init__()
        self.operations = operations
        # Store constants as buffers so they're part of the graph
        for i, (op_name, args) in enumerate(operations):
            for k, v in args.items():
                if isinstance(v, (int, float)):
                    buf = torch.tensor(v, dtype=torch.float32)
                    self.register_buffer(f'op{i}_{k}', buf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P = Primitives
        for i, (op_name, args) in enumerate(self.operations):
            # Resolve int args from buffers
            resolved = {}
            for k, v in args.items():
                if isinstance(v, (int, float)):
                    resolved[k] = int(getattr(self, f'op{i}_{k}').item())
                else:
                    resolved[k] = v

            fn = getattr(P, op_name)

            if op_name in ('hconcat', 'vconcat', 'cellwise', 'add', 'subtract',
                           'multiply', 'minimum', 'maximum', 'merge',
                           'equality', 'greater', 'less'):
                # Binary ops: x is the "accumulator", second input stored separately
                second = resolved.pop('b', None)
                if second is not None:
                    x = fn(x, second, **resolved)
                else:
                    # For trace mode, second tensor comes from buffer
                    b_buf = getattr(self, f'op{i}_b', None)
                    if b_buf is not None:
                        x = fn(x, b_buf, **resolved)
                    else:
                        x = fn(x, **resolved)
            elif op_name in ('fill_masked', 'paint_masked'):
                mask = getattr(self, f'op{i}_mask', None)
                x = fn(x, mask, **resolved)
            else:
                x = fn(x, **resolved)

        return x


# ============================================================================
# Solver parser: extracts operation sequence from solver source
# ============================================================================

# DSL constant map
CONSTANTS = {
    'ZERO': 0, 'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4,
    'FIVE': 5, 'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9,
    'T': True, 'F': False,
}

# Primitive name mapping (DSL name → our method name)
PRIM_MAP = {
    'rot90': 'rot90', 'rot180': 'rot180', 'rot270': 'rot270',
    'hmirror': 'hmirror', 'vmirror': 'vmirror',
    'dmirror': 'dmirror', 'cmirror': 'cmirror',
    'crop': 'crop', 'trim': 'trim',
    'tophalf': 'tophalf', 'bottomhalf': 'bottomhalf',
    'lefthalf': 'lefthalf', 'righthalf': 'righthalf',
    'hconcat': 'hconcat', 'vconcat': 'vconcat',
    'hupscale': 'hupscale', 'vupscale': 'vupscale',
    'upscale': 'upscale', 'downscale': 'downscale',
    'add': 'add', 'subtract': 'subtract', 'multiply': 'multiply',
    'increment': 'increment', 'decrement': 'decrement',
    'double': 'double', 'minimum': 'minimum', 'maximum': 'maximum',
    'equality': 'equality', 'greater': 'greater', 'less': 'less',
    'cellwise': 'cellwise', 'replace': 'replace',
    'switch': 'switch', 'merge': 'merge',
    'ofcolor': 'ofcolor',
    'fill': 'fill_masked', 'paint': 'paint_masked',
    'canvas': 'canvas',
}

# Primitives that take extra integer args
PRIM_WITH_ARGS = {
    'hupscale': ['factor'], 'vupscale': ['factor'], 'upscale': ['factor'],
    'downscale': ['factor'],
    'crop': ['top', 'left', 'height', 'width'],
    'replace': ['old_color', 'new_color'],
    'switch': ['a', 'b'],
    'increment': ['delta'], 'decrement': ['delta'],
    'fill': ['color'], 'paint': ['color'],
    'ofcolor': ['color'],
    'canvas': ['color', 'h', 'w'],
}


def parse_solver(source: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse solver Python source into operation list.

    Example:
        x1 = hupscale(I, THREE)
        x2 = vupscale(x1, THREE)
        O = cellwise(x2, x1)
    → [('hupscale', {'factor': 3}), ('vupscale', {'factor': 3}), ('cellwise', {})]
    """
    lines = source.strip().split('\n')
    ops = []
    var_map = {}  # var_name → index in ops list (or 'input')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('def ') or line.startswith('#'):
            continue

        # Parse assignment: x1 = func(args...)  or  O = func(args...)
        m = re.match(r'(\w+)\s*=\s*(\w+)\((.*)\)', line)
        if not m:
            continue

        var_name = m.group(1)
        func_name = m.group(2)
        args_str = m.group(3)

        if func_name in ('return',):
            continue

        if func_name not in PRIM_MAP:
            # Skip unknown functions (higher-order, object-level, etc.)
            var_map[var_name] = None
            continue

        prim_name = PRIM_MAP[func_name]

        # Parse arguments
        args = _parse_args(args_str, var_map)
        var_map[var_name] = len(ops)

        ops.append((prim_name, args))

    return ops


def _parse_args(args_str: str, var_map: dict) -> dict:
    """Parse argument string, resolving constants."""
    args = {}
    parts = _smart_split(args_str)

    if not parts:
        return args

    prim_name = None  # will be set by caller context

    # First arg is usually the input (skip it — it's the accumulator)
    # Remaining args are parameters
    param_idx = 0
    for i, part in enumerate(parts):
        part = part.strip()
        if part in CONSTANTS:
            val = CONSTANTS[part]
        elif part.isdigit():
            val = int(part)
        elif part.startswith("'") and part.endswith("'"):
            val = part[1:-1]
        else:
            # Variable reference (input tensor) — skip
            continue

        param_idx += 1

    # Return all non-variable args as a dict
    result = {}
    param_idx = 0
    for part in parts:
        part = part.strip()
        if part in CONSTANTS:
            result[f'arg{param_idx}'] = CONSTANTS[part]
            param_idx += 1
        elif part.isdigit():
            result[f'arg{param_idx}'] = int(part)
            param_idx += 1

    return result


def _smart_split(s: str) -> List[str]:
    """Split by commas, respecting nested parens."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


# ============================================================================
# Quick manual builder (for demo without parser)
# ============================================================================

def build_task001_model() -> NeuroGolfModel:
    """Build task001: hupscale(3) → vupscale(3) → cellwise with hconcat(x3,x3,x3)."""
    # Solver:
    #   x1 = hupscale(I, 3)         → 3x9
    #   x2 = vupscale(x1, 3)        → 9x9 (pixel-level upscale)
    #   x3 = hconcat(I, I)           → 3x6
    #   x4 = hconcat(x3, I)          → 3x9 (block-tiling)
    #   x5 = vconcat(x4, x4)         → 6x9
    #   x6 = vconcat(x5, x4)         → 9x9 (block-tiling)
    #   O = cellwise(x2, x6)         → 9x9 (keep where both match)
    #
    # In our model, we need to store x2 as an intermediate for cellwise.
    # The NeuroGolfModel chains ops on a single tensor, but cellwise needs two.
    # Solution: use a tuple-based approach or a "fork" pattern.

    # For the prototype, let's trace two sub-models and combine manually.
    # Actually, let's just implement the full pipeline directly in forward().
    return NeuroGolfModel_task001()


class NeuroGolfModel_task001(nn.Module):
    """Task001-specific model showing the full chain."""

    def __init__(self):
        super().__init__()
        self.factor3 = torch.tensor(3, dtype=torch.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P = Primitives
        # Branch 1: pixel-level upscale
        x1 = P.hupscale(x, 3)
        x2 = P.vupscale(x1, 3)

        # Branch 2: block-tiling
        x3 = P.hconcat(x, x)
        x4 = P.hconcat(x3, x)
        x5 = P.vconcat(x4, x4)
        x6 = P.vconcat(x5, x4)

        # Combine
        return P.cellwise(x2, x6)


# ============================================================================
# Trace → Export → Verify
# ============================================================================

def trace_and_export(model: nn.Module, task_id: int, input_tensor: torch.Tensor,
                     output_path: str = None):
    """Trace a model and export to ONNX."""
    if output_path is None:
        output_path = f'task{task_id:03d}.onnx'

    model.eval()

    # Trace
    with torch.no_grad():
        traced = torch.jit.trace(model, input_tensor)

    # Export
    torch.onnx.export(
        model,
        input_tensor,
        output_path,
        opset_version=17,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,  # static shapes only!
    )

    print(f'Exported: {output_path}')
    return output_path


def verify_onnx(onnx_path: str, input_np: np.ndarray, expected_np: np.ndarray):
    """Verify exported ONNX produces correct output."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    # Run ONNX
    onnx_out = sess.run(None, {input_name: input_np})[0]

    # Compare
    match = np.array_equal(np.argmax(onnx_out, axis=1),
                           np.argmax(expected_np, axis=1))
    print(f'ONNX output matches expected: {match}')
    if not match:
        print(f'  Expected (argmax):\n{np.argmax(expected_np, axis=1)[0]}')
        print(f'  Got (argmax):\n{np.argmax(onnx_out, axis=1)[0]}')
    return match


# ============================================================================
# Demo
# ============================================================================

def demo_task001():
    """End-to-end demo: build task001, trace, export, verify."""
    print("=" * 60)
    print("NeuroGolf Hybrid Prototype — Task 001")
    print("=" * 60)

    # Load task data
    task_path = Path(__file__).parent.parent / '07' / 'tasks' / 'task001.json'
    with open(task_path) as f:
        task = json.load(f)

    # Get a training example
    example = task['train'][0]
    input_grid = np.array(example['input'], dtype=np.int64)
    output_grid = np.array(example['output'], dtype=np.int64)

    h_in, w_in = input_grid.shape
    h_out, w_out = output_grid.shape

    print(f'Input:  {h_in}x{w_in}')
    print(f'Output: {h_out}x{w_out}')

    # One-hot encode
    def oh_encode(grid):
        oh = np.zeros((1, NUM_COLORS, grid.shape[0], grid.shape[1]), dtype=np.float32)
        for c in range(NUM_COLORS):
            oh[0, c] = (grid == c).astype(np.float32)
        return oh

    input_oh = oh_encode(input_grid)
    output_oh = oh_encode(output_grid)

    input_t = torch.from_numpy(input_oh)
    output_t = torch.from_numpy(output_oh)

    # Build model
    model = NeuroGolfModel_task001()

    # Run PyTorch
    with torch.no_grad():
        pt_out = model(input_t)

    pt_match = (pt_out.argmax(dim=1).numpy() == output_oh.argmax(axis=1)).all()
    print(f'PyTorch output matches: {pt_match}')

    if not pt_match:
        print('PyTorch mismatch — debugging...')
        print(f'  Expected:\n{output_oh.argmax(axis=1)[0]}')
        print(f'  Got:\n{pt_out.argmax(dim=1).numpy()[0]}')
        return

    # Export to ONNX
    onnx_path = str(Path(__file__).parent / 'task001.onnx')
    trace_and_export(model, 1, input_t, onnx_path)

    # Verify
    verify_onnx(onnx_path, input_oh, output_oh)

    # Also test with other training examples
    print('\nVerifying on all training examples:')
    for i, ex in enumerate(task['train']):
        inp = oh_encode(np.array(ex['input'], dtype=np.int64))
        out = oh_encode(np.array(ex['output'], dtype=np.int64))
        inp_t = torch.from_numpy(inp)
        with torch.no_grad():
            pred = model(inp_t)
        match = (pred.argmax(dim=1).numpy() == out.argmax(axis=1)).all()
        print(f'  Example {i}: {"PASS" if match else "FAIL"}')


def demo_arithmetic():
    """Demo: arithmetic primitives."""
    print("\n" + "=" * 60)
    print("NeuroGolf Hybrid Prototype — Arithmetic Demo")
    print("=" * 60)

    P = Primitives

    def make_grid(values):
        """Create one-hot from small grid."""
        g = np.array(values, dtype=np.int64)
        oh = np.zeros((1, NUM_COLORS, g.shape[0], g.shape[1]), dtype=np.float32)
        for c in range(NUM_COLORS):
            oh[0, c] = (g == c).astype(np.float32)
        return torch.from_numpy(oh)

    a = make_grid([[1, 2], [3, 4]])
    b = make_grid([[5, 6], [7, 8]])

    with torch.no_grad():
        add_result = P.add(a, b)
        sub_result = P.subtract(a, b)
        mul_result = P.multiply(a, b)

    print(f"a = [[1,2],[3,4]]")
    print(f"b = [[5,6],[7,8]]")
    print(f"add:      {add_result.argmax(dim=1).numpy()[0]}")
    print(f"subtract: {sub_result.argmax(dim=1).numpy()[0]}")
    print(f"multiply: {mul_result.argmax(dim=1).numpy()[0]}")

    # Verify
    expected_add = (np.array([[1,2],[3,4]]) + np.array([[5,6],[7,8]])) % 10
    expected_sub = (np.array([[1,2],[3,4]]) - np.array([[5,6],[7,8]])) % 10
    expected_mul = (np.array([[1,2],[3,4]]) * np.array([[5,6],[7,8]])) % 10

    assert np.array_equal(add_result.argmax(dim=1).numpy()[0], expected_add)
    assert np.array_equal(sub_result.argmax(dim=1).numpy()[0], expected_sub)
    assert np.array_equal(mul_result.argmax(dim=1).numpy()[0], expected_mul)
    print("All arithmetic checks PASS")


def demo_spatial():
    """Demo: spatial transforms."""
    print("\n" + "=" * 60)
    print("NeuroGolf Hybrid Prototype — Spatial Demo")
    print("=" * 60)

    P = Primitives

    def make_grid(values):
        g = np.array(values, dtype=np.int64)
        oh = np.zeros((1, NUM_COLORS, g.shape[0], g.shape[1]), dtype=np.float32)
        for c in range(NUM_COLORS):
            oh[0, c] = (g == c).astype(np.float32)
        return torch.from_numpy(oh)

    grid = make_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    with torch.no_grad():
        r90 = P.rot90(grid)
        r180 = P.rot180(grid)
        hm = P.hmirror(grid)
        vm = P.vmirror(grid)
        dm = P.dmirror(grid)
        h2 = P.hupscale(grid, 2)
        v2 = P.vupscale(grid, 2)
        u2 = P.upscale(grid, 2)
        d2 = P.downscale(u2, 2)

    print(f"Input 3x3:")
    print(f"  {grid.argmax(dim=1).numpy()[0]}")
    print(f"rot90:")
    print(f"  {r90.argmax(dim=1).numpy()[0]}")
    print(f"hupscale(2):")
    print(f"  {h2.argmax(dim=1).numpy()[0]}")
    print(f"vupscale(2):")
    print(f"  {v2.argmax(dim=1).numpy()[0]}")
    print(f"upscale(2):")
    print(f"  {u2.argmax(dim=1).numpy()[0]}")
    print(f"downscale(upscale(2), 2):")
    print(f"  {d2.argmax(dim=1).numpy()[0]}")

    # Verify round-trip: pixel-repeat upscale then nearest downscale recovers original
    assert np.array_equal(d2.argmax(dim=1).numpy()[0],
                           grid.argmax(dim=1).numpy()[0])
    print("Downscale round-trip: PASS")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    demo_arithmetic()
    demo_spatial()
    demo_task001()
