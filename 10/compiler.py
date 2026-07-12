"""
Solver → ONNX Compiler (with higher-order inlining)

Pipeline:
  1. Parse solver source into op list
  2. Build environment: track variable types (grid, rbind, compose, fork, etc.)
  3. Resolve higher-order chains when application primitives hit
  4. Emit flat batch-aware tensor ops
  5. Build traceable PyTorch model → export ONNX

CRITICAL: All forward() code must be torch.jit.trace compatible.
No Python conditionals on tensor values. Always compute, then mask.
"""

import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from solver_parser import parse_solver, resolve_constants, extract_solver, CONSTANTS
from primitives import (
    PRIM_DISPATCH, PRIM_NEEDS_ARGS, BINARY_OPS,
    oh_decode, oh_encode, NUM_COLORS, CANVAS,
)
from batch_ops import (
    extract_objects_batch, colorfilter_batch, sizefilter_batch,
    mfilter_merge, merge_batch,
    difference_batch, bordering_batch, hline_batch, vline_batch,
    color_batch, size_batch, numcolors_batch, equality_scalar,
    centerofmass_batch, MAX_OBJ, content_mask,
)

GRID_OPS = {
    "rot90", "rot180", "rot270", "hmirror", "vmirror", "dmirror", "cmirror",
    "crop", "trim", "tophalf", "bottomhalf", "lefthalf", "righthalf",
    "hconcat", "vconcat",
    "hupscale", "vupscale", "upscale", "downscale",
    "fill", "underfill", "cover", "paint", "underpaint",
    "replace", "switch", "cellwise",
    "add", "subtract", "multiply", "divide", "double", "halve",
    "increment", "decrement", "crement", "invert", "sign",
    "even", "positive", "greater", "equality", "less",
    "flip", "both", "either",
    "ofcolor", "identity",
}


# ============================================================================
# Environment-based resolver
# ============================================================================

class Resolver:
    """Tracks variable types and resolves higher-order expressions."""

    def __init__(self):
        self.env = {}
        self.in_batch_mode = False
        self.batch_var = None

    def resolve_all(self, ops):
        resolved = []

        for op in ops:
            func = op["func"]
            var = op["var"]
            args = op["args"]

            if func == "identity":
                self.env[var] = self.env.get(args[0], {"type": "grid", "var": args[0]})
                resolved.append(op)
                continue

            # Higher-order constructors: register in env, don't emit op
            if func in ("rbind", "lbind", "compose", "fork", "chain",
                         "branch", "power", "matcher"):
                entry = {"type": func}
                if func == "rbind":
                    entry["fn"] = args[0] if args else None
                    entry["arg"] = args[1] if len(args) > 1 else None
                elif func == "lbind":
                    entry["fn"] = args[0] if args else None
                    entry["arg"] = args[1] if len(args) > 1 else None
                elif func == "compose":
                    entry["outer"] = args[0] if args else None
                    entry["inner"] = args[1] if len(args) > 1 else None
                elif func == "fork":
                    entry["outer"] = args[0] if args else None
                    entry["a"] = args[1] if len(args) > 1 else None
                    entry["b"] = args[2] if len(args) > 2 else None
                elif func == "chain":
                    entry["h"] = args[0] if args else None
                    entry["g"] = args[1] if len(args) > 1 else None
                    entry["f"] = args[2] if len(args) > 2 else None
                elif func == "branch":
                    entry["cond"] = args[0] if args else None
                    entry["a"] = args[1] if len(args) > 1 else None
                    entry["b"] = args[2] if len(args) > 2 else None
                elif func == "power":
                    entry["fn"] = args[0] if args else None
                    entry["n"] = args[1] if len(args) > 1 else 1
                elif func == "matcher":
                    entry["fn"] = args[0] if args else None
                    entry["target"] = args[1] if len(args) > 1 else None
                self.env[var] = entry
                continue

            # Objects: phase transition to batch mode
            if func == "objects":
                univalued = True
                diagonal = False
                without_bg = True
                if len(args) > 1:
                    univalued = (args[1] == "T" or args[1] is True)
                if len(args) > 2:
                    diagonal = (args[2] == "T" or args[2] is True)
                if len(args) > 3:
                    without_bg = (args[3] == "T" or args[3] is True)

                self.in_batch_mode = True
                self.batch_var = var
                self.env[var] = {"type": "batch", "var": var}
                resolved.append({
                    "var": var,
                    "func": "_extract_objects",
                    "args": ["I"],
                    "kwargs": {
                        "univalued": univalued,
                        "diagonal": diagonal,
                        "without_bg": without_bg,
                    }
                })
                continue

            # Batch-mode operations
            if self.in_batch_mode:
                result = self._resolve_batch_op(func, var, args)
                if result is not None:
                    resolved.extend(result if isinstance(result, list) else [result])
                    continue

            # Standard grid op: pass through
            if func in PRIM_DISPATCH or func in ("identity",):
                self.env[var] = {"type": "grid", "var": var}
                resolved.append(op)
                continue

            return None

        return resolved

    def _resolve_batch_op(self, func, var, args):
        """Resolve operations that happen in batch mode."""

        if func == "colorfilter":
            color = self._resolve_scalar(args[1]) if len(args) > 1 else 0
            self.env[var] = {"type": "batch", "var": var}
            return {"var": var, "func": "_colorfilter_batch",
                    "args": [args[0]], "kwargs": {"color": color}}

        if func == "sizefilter":
            n = self._resolve_scalar(args[1]) if len(args) > 1 else 1
            self.env[var] = {"type": "batch", "var": var}
            return {"var": var, "func": "_sizefilter_batch",
                    "args": [args[0]], "kwargs": {"n": n}}

        if func == "mapply":
            fn_name = args[0] if args else None
            collection = args[1] if len(args) > 1 else None
            self.env[var] = {"type": "grid", "var": var}
            self.in_batch_mode = False
            fn_spec = self._resolve_fn(fn_name)
            return {"var": var, "func": "_mapply_batch",
                    "args": [collection], "kwargs": {"fn_spec": fn_spec}}

        if func == "mfilter":
            collection = args[0] if args else None
            pred = args[1] if len(args) > 1 else None
            self.env[var] = {"type": "grid", "var": var}
            self.in_batch_mode = False
            pred_spec = self._resolve_fn(pred)
            return {"var": var, "func": "_mfilter_batch",
                    "args": [collection], "kwargs": {"pred_spec": pred_spec}}

        if func == "apply":
            fn_name = args[0] if args else None
            collection = args[1] if len(args) > 1 else None
            fn_spec = self._resolve_fn(fn_name)
            self.env[var] = {"type": "batch", "var": var}
            return {"var": var, "func": "_apply_batch",
                    "args": [collection], "kwargs": {"fn_spec": fn_spec}}

        if func == "merge":
            self.env[var] = {"type": "grid", "var": var}
            self.in_batch_mode = False
            return {"var": var, "func": "_merge_batch", "args": args[:1], "kwargs": {}}

        if func == "difference":
            self.env[var] = {"type": "batch", "var": var}
            return {"var": var, "func": "_difference_batch",
                    "args": args[:2], "kwargs": {}}

        if func == "combine":
            self.env[var] = {"type": "batch", "var": var}
            return {"var": var, "func": "_combine_batch",
                    "args": args[:2], "kwargs": {}}

        if func == "sfilter":
            collection = args[0] if args else None
            pred = args[1] if len(args) > 1 else None
            pred_spec = self._resolve_fn(pred)
            self.env[var] = {"type": "batch", "var": var}
            return {"var": var, "func": "_sfilter_batch",
                    "args": [collection], "kwargs": {"pred_spec": pred_spec}}

        if func == "extract":
            collection = args[0] if args else None
            pred = args[1] if len(args) > 1 else None
            pred_spec = self._resolve_fn(pred)
            self.env[var] = {"type": "batch_single", "var": var}
            return {"var": var, "func": "_extract_batch",
                    "args": [collection], "kwargs": {"pred_spec": pred_spec}}

        if func == "order":
            comp = args[1] if len(args) > 1 else None
            self.env[var] = {"type": "batch", "var": var}
            return {"var": var, "func": "_order_batch",
                    "args": args[:1], "kwargs": {"comp": comp}}

        if func == "size":
            self.env[var] = {"type": "batch_scalar", "var": var}
            return {"var": var, "func": "_size_batch", "args": args[:1], "kwargs": {}}

        if func == "color":
            self.env[var] = {"type": "batch_color", "var": var}
            return {"var": var, "func": "_color_batch_var", "args": args[:1], "kwargs": {}}

        if func == "first":
            self.env[var] = {"type": "batch_single", "var": var}
            return {"var": var, "func": "_first_batch", "args": args[:1], "kwargs": {}}

        if func == "last":
            self.env[var] = {"type": "batch_single", "var": var}
            return {"var": var, "func": "_last_batch", "args": args[:1], "kwargs": {}}

        if func == "totuple":
            self.env[var] = {"type": "batch_tuple", "var": var}
            return {"var": var, "func": "_totuple_batch", "args": args[:1], "kwargs": {}}

        if func == "recolor":
            self.env[var] = {"type": "batch", "var": var}
            return {"var": var, "func": "_recolor_batch", "args": args[:2], "kwargs": {}}

        if func == "shift":
            self.env[var] = {"type": "batch", "var": var}
            return {"var": var, "func": "_shift_batch", "args": args[:2], "kwargs": {}}

        if func == "mpapply":
            fn_name = args[0] if args else None
            a = args[1] if len(args) > 1 else None
            b = args[2] if len(args) > 2 else None
            self.env[var] = {"type": "grid", "var": var}
            self.in_batch_mode = False
            fn_spec = self._resolve_fn(fn_name)
            return {"var": var, "func": "_mapply_batch",
                    "args": [a], "kwargs": {"fn_spec": fn_spec}}

        if func == "toindices":
            self.env[var] = {"type": "indices", "var": var}
            return {"var": var, "func": "_toindices_batch", "args": args[:1], "kwargs": {}}

        return None

    def _resolve_fn(self, fn_ref):
        """Resolve a function reference to a fully resolved specification dict."""
        if fn_ref is None:
            return {"type": "identity"}

        if fn_ref in PRIM_DISPATCH or fn_ref in GRID_OPS:
            return {"type": "primitive", "name": fn_ref}

        if fn_ref in self.env:
            entry = self.env[fn_ref]
            if entry["type"] in ("rbind", "lbind", "compose", "fork",
                                  "chain", "branch", "power", "matcher"):
                resolved = dict(entry)
                for key in ("fn", "arg", "outer", "inner", "a", "b",
                            "h", "g", "f", "cond"):
                    if key in resolved and isinstance(resolved[key], str):
                        if key == "fn" and resolved[key] in PRIM_DISPATCH:
                            continue
                        sub = self._resolve_fn(resolved[key])
                        if sub["type"] != "identity":
                            resolved[key] = sub
                return resolved
            if entry["type"] == "grid":
                return {"type": "primitive", "name": "identity"}
            if entry["type"] == "batch":
                return {"type": "batch_var", "var": fn_ref}
            if entry["type"] == "batch_scalar":
                return {"type": "batch_scalar_var", "var": fn_ref}
            if entry["type"] == "batch_color":
                return {"type": "batch_color_var", "var": fn_ref}

        if fn_ref in CONSTANTS:
            return {"type": "constant", "value": CONSTANTS[fn_ref]}

        return {"type": "primitive", "name": fn_ref}

    def _resolve_scalar(self, val):
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            if val in CONSTANTS:
                return int(CONSTANTS[val])
            if val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
                return int(val)
        return 0


# ============================================================================
# Grid-only function application (for mapply fn_spec evaluation)
# ============================================================================

def apply_fn_to_grid(fn_spec, grid):
    """Apply a resolved function spec to a single (1,10,H,W) grid."""
    if fn_spec["type"] == "primitive":
        name = fn_spec["name"]
        fn = PRIM_DISPATCH.get(name)
        if fn is None:
            return grid
        return fn(grid)

    if fn_spec["type"] == "compose":
        inner = apply_fn_to_grid(fn_spec["inner"], grid)
        outer_name = fn_spec.get("outer", "")
        if isinstance(outer_name, str) and outer_name in PRIM_DISPATCH:
            return PRIM_DISPATCH[outer_name](inner)
        if isinstance(outer_name, dict):
            return apply_fn_to_grid(outer_name, inner)
        return inner

    if fn_spec["type"] == "rbind":
        fn_name = fn_spec["fn"]
        fixed = fn_spec["arg"]
        if isinstance(fn_name, str) and fn_name in PRIM_DISPATCH:
            if isinstance(fixed, dict):
                return grid  # Can't resolve variable fixed args here
            return PRIM_DISPATCH[fn_name](grid, fixed)
        return grid

    if fn_spec["type"] == "fork":
        a_result = apply_fn_to_grid(fn_spec["a"], grid)
        b_result = apply_fn_to_grid(fn_spec["b"], grid)
        outer_name = fn_spec.get("outer", "")
        if isinstance(outer_name, str) and outer_name in PRIM_DISPATCH:
            return PRIM_DISPATCH[outer_name](a_result, b_result)
        return a_result

    if fn_spec["type"] == "chain":
        f_result = apply_fn_to_grid(fn_spec["f"], grid)
        g_result = apply_fn_to_grid(fn_spec["g"], f_result)
        h_result = apply_fn_to_grid(fn_spec["h"], g_result)
        return h_result

    if fn_spec["type"] == "power":
        n = fn_spec["n"]
        if isinstance(n, str) and n in CONSTANTS:
            n = CONSTANTS[n]
        result = grid
        for _ in range(int(n)):
            result = apply_fn_to_grid(fn_spec["fn"], result)
        return result

    return grid


# ============================================================================
# Model: trace-compatible forward pass
# ============================================================================

class SolverModel(nn.Module):
    def __init__(self, grid_ops: List[Dict[str, Any]], h_in: int, w_in: int,
                 pred_specs: Dict, fn_specs: Dict, op_types: Dict):
        super().__init__()
        self.grid_ops = grid_ops
        self.h_in = h_in
        self.w_in = w_in
        self.pred_specs = pred_specs
        self.fn_specs = fn_specs
        self.op_types = op_types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, :self.h_in, :self.w_in]

        vars = {"I": x}
        in_batch_mode = False
        batch_objs = None
        batch_valid = None

        for op in self.grid_ops:
            func = op["func"]
            var = op["var"]
            args = op.get("args", [])

            # ---- BATCH OPERATIONS ----

            if func == "_extract_objects":
                grid = vars["I"]
                kwargs = op.get("kwargs", {})
                batch_objs, batch_valid = extract_objects_batch(
                    grid,
                    kwargs.get("univalued", True),
                    kwargs.get("diagonal", False),
                    kwargs.get("without_bg", True),
                )
                vars[var] = batch_objs
                in_batch_mode = True
                continue

            if func == "_colorfilter_batch":
                color = op.get("kwargs", {}).get("color", 0)
                batch_valid = colorfilter_batch(batch_objs, batch_valid, color)
                vars[var] = batch_objs
                continue

            if func == "_sizefilter_batch":
                n = op.get("kwargs", {}).get("n", 1)
                batch_valid = sizefilter_batch(batch_objs, batch_valid, n)
                vars[var] = batch_objs
                continue

            if func == "_mfilter_batch":
                op_id = id(op)
                pred_spec = self.pred_specs.get(op_id, {"type": "identity"})
                # Evaluate predicate for ALL objects (no if-check on valid)
                pred_mask = self._eval_predicate_batch(pred_spec, batch_objs, batch_valid, vars["I"])
                result = mfilter_merge(batch_objs, batch_valid, pred_mask)
                vars[var] = result
                in_batch_mode = False
                continue

            if func == "_mapply_batch":
                op_id = id(op)
                fn_spec = self.fn_specs.get(op_id, {"type": "identity"})
                result = self._exec_mapply(fn_spec, batch_objs, batch_valid, vars)
                vars[var] = result
                in_batch_mode = False
                continue

            if func == "_apply_batch":
                op_id = id(op)
                fn_spec = self.fn_specs.get(op_id, {"type": "identity"})
                result = self._exec_apply(fn_spec, batch_objs, batch_valid, vars)
                vars[var] = result
                continue

            if func == "_merge_batch":
                result = batch_objs.max(dim=0, keepdim=True).values
                vars[var] = result
                in_batch_mode = False
                continue

            if func == "_difference_batch":
                a_name, b_name = args[0], args[1] if len(args) > 1 else args[0]
                a_obj = vars.get(a_name, batch_objs)
                b_obj = vars.get(b_name, batch_objs)
                new_valid = self._compute_difference_valid(a_obj, batch_valid, b_obj)
                batch_valid = new_valid
                vars[var] = batch_objs
                continue

            if func == "_combine_batch":
                batch_valid = torch.clamp(batch_valid + batch_valid, 0, 1)
                vars[var] = batch_objs
                continue

            if func == "_sfilter_batch":
                op_id = id(op)
                pred_spec = self.pred_specs.get(op_id, {"type": "identity"})
                pred_mask = self._eval_predicate_batch(pred_spec, batch_objs, batch_valid, vars["I"])
                batch_valid = batch_valid * pred_mask
                vars[var] = batch_objs
                continue

            if func == "_extract_batch":
                op_id = id(op)
                pred_spec = self.pred_specs.get(op_id, {"type": "identity"})
                pred_mask = self._eval_predicate_batch(pred_spec, batch_objs, batch_valid, vars["I"])
                # Get first matching object
                vars[var] = self._extract_first_match(batch_objs, pred_mask)
                in_batch_mode = False
                continue

            if func == "_order_batch":
                vars[var] = batch_objs
                continue

            if func == "_size_batch":
                result = size_batch(batch_objs, batch_valid)
                vars[var] = result
                continue

            if func == "_color_batch_var":
                result = color_batch(batch_objs, batch_valid)
                vars[var] = result
                continue

            if func == "_first_batch":
                vars[var] = self._get_first_valid(batch_objs, batch_valid)
                in_batch_mode = False
                continue

            if func == "_last_batch":
                vars[var] = self._get_last_valid(batch_objs, batch_valid)
                in_batch_mode = False
                continue

            if func == "_totuple_batch":
                vars[var] = batch_objs
                continue

            # ---- STANDARD GRID OPERATIONS ----
            fn = PRIM_DISPATCH.get(func)
            if fn is None:
                return None

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

            ALL_POSITIONAL = {
                "fill", "underfill", "cover", "paint", "underpaint",
                "cellwise", "replace", "switch", "ofcolor", "canvas",
                "crop", "hsplit", "vsplit", "hupscale", "vupscale",
                "upscale", "downscale", "colorcount",
            }

            if func in ALL_POSITIONAL:
                result = fn(*resolved)
            else:
                needs = PRIM_NEEDS_ARGS.get(func, [])
                if not needs:
                    result = fn(*resolved)
                else:
                    n_tensor = len(resolved) - len(needs)
                    tensor_args = resolved[:n_tensor]
                    extra_args = resolved[n_tensor:]
                    op_kwargs = {}
                    for i, name in enumerate(needs):
                        if i < len(extra_args):
                            val = extra_args[i]
                            if isinstance(val, str) and val in CONSTANTS:
                                val = CONSTANTS[val]
                            op_kwargs[name] = int(val) if isinstance(val, (int, float)) else val
                    result = fn(*tensor_args, **op_kwargs)

            vars[var] = result

        # Get final output
        out = vars.get("O", vars[list(vars.keys())[-1]])

        # Pad/crop to CANVASxCANVAS using static ops (no Python conditionals)
        # Slice to at most CANVAS
        out = out[:, :, :CANVAS, :CANVAS]
        # Pad to at least CANVAS with background
        pad_h = CANVAS - out.shape[2]
        pad_w = CANVAS - out.shape[3]
        out = F.pad(out, (0, pad_w, 0, pad_h), value=0)
        # Ensure channel 0 = 1 for padded regions
        padded_mask = torch.zeros_like(out)
        padded_mask[:, 0, :, :] = 1.0
        # Only apply padding fill where actual data isn't present
        # Use a simpler approach: just slice/pad which works for static shapes
        return out

    def _eval_predicate_batch(self, pred_spec, batch_objs, batch_valid, grid):
        """Evaluate predicate for ALL objects at once (no .item(), no per-object loop).

        Returns: (MAX_OBJ,) float {0, 1}
        """
        n = batch_objs.shape[0]

        if pred_spec["type"] == "primitive":
            name = pred_spec["name"]
            if name == "hline":
                return hline_batch(batch_objs, batch_valid) * batch_valid
            elif name == "vline":
                return vline_batch(batch_objs, batch_valid) * batch_valid
            return torch.zeros(n, device=batch_objs.device)

        if pred_spec["type"] == "compose":
            inner = self._eval_predicate_batch(pred_spec["inner"], batch_objs, batch_valid, grid)
            outer = pred_spec.get("outer", "")
            if isinstance(outer, str) and outer == "flip":
                return (1.0 - inner) * batch_valid
            if isinstance(outer, dict) and outer.get("name", "") == "flip":
                return (1.0 - inner) * batch_valid
            if isinstance(outer, dict):
                return self._eval_predicate_batch(outer, batch_objs, batch_valid, grid) * batch_valid
            return inner

        if pred_spec["type"] == "rbind":
            fn = pred_spec["fn"]
            fn_name = fn.get("name", fn) if isinstance(fn, dict) else fn
            if fn_name == "bordering":
                return bordering_batch(batch_objs, batch_valid, grid) * batch_valid
            if fn_name == "greater":
                arg = pred_spec.get("arg", {})
                if isinstance(arg, dict) and arg.get("name") == "ONE":
                    nc = numcolors_batch(batch_objs, batch_valid)
                    nc_val = nc[:, 1:, :, :].sum(dim=1).squeeze(-1).squeeze(-1)
                    return (nc_val > 1).float() * batch_valid
            return torch.zeros(n, device=batch_objs.device)

        if pred_spec["type"] == "fork":
            a_val = self._eval_predicate_batch(pred_spec["a"], batch_objs, batch_valid, grid)
            b_val = self._eval_predicate_batch(pred_spec["b"], batch_objs, batch_valid, grid)
            outer = pred_spec.get("outer", "")
            if isinstance(outer, str):
                if outer == "either":
                    return torch.clamp(a_val + b_val, 0, 1) * batch_valid
                elif outer == "both":
                    return torch.min(a_val, b_val) * batch_valid
                elif outer == "equality":
                    return (a_val == b_val).float() * batch_valid
            if isinstance(outer, dict):
                return self._eval_predicate_batch(outer, batch_objs, batch_valid, grid) * batch_valid
            return torch.zeros(n, device=batch_objs.device)

        if pred_spec["type"] == "matcher":
            fn = pred_spec.get("fn", "")
            fn_name = fn.get("name", fn) if isinstance(fn, dict) else fn
            target = pred_spec.get("target", 0)
            if fn_name == "first":
                c = color_batch(batch_objs, batch_valid)
                target_int = target
                if isinstance(target_int, dict):
                    target_int = target_int.get("value", 0)
                val = c[:, int(target_int), :, :].squeeze(-1).squeeze(-1)
                return val * batch_valid
            return torch.zeros(n, device=batch_objs.device)

        return torch.zeros(n, device=batch_objs.device)

    def _exec_mapply(self, fn_spec, batch_objs, batch_valid, vars):
        """Apply fn to each valid object and merge results."""
        results = []
        for i in range(MAX_OBJ):
            obj = batch_objs[i:i+1]
            result = apply_fn_to_grid(fn_spec, obj)
            results.append(result)
        stacked = torch.cat(results, dim=0)  # (MAX_OBJ, 10, H, W)
        # Mask invalid objects
        masked = stacked * batch_valid.view(-1, 1, 1, 1)
        return masked.max(dim=0, keepdim=True).values

    def _exec_apply(self, fn_spec, batch_objs, batch_valid, vars):
        """Apply fn to each object (stays in batch mode)."""
        results = []
        for i in range(MAX_OBJ):
            obj = batch_objs[i:i+1]
            result = apply_fn_to_grid(fn_spec, obj)
            results.append(result)
        return torch.cat(results, dim=0)

    def _compute_difference_valid(self, a_objs, a_valid, b_objs):
        """Compute valid mask for set difference A - B (trace-compatible)."""
        d_a = oh_decode(a_objs)  # (MAX_OBJ, 1, H, W)
        d_b = oh_decode(b_objs)  # (MAX_OBJ, 1, H, W)
        is_a = (d_a > 0).float()  # (MAX_OBJ, 1, H, W)
        is_b = (d_b > 0).float()  # (MAX_OBJ, 1, H, W)
        # For each A object, check if any B object has same shape
        a_flat = is_a.view(a_objs.shape[0], -1)  # (N_A, H*W)
        b_flat = is_b.view(b_objs.shape[0], -1)  # (N_B, H*W)
        # Pairwise equality: (N_A, N_B, H*W) -> (N_A, N_B)
        eq_matrix = (a_flat.unsqueeze(1) == b_flat.unsqueeze(0)).float()
        # all pixels match -> (N_A, N_B)
        shape_match = eq_matrix.prod(dim=2)
        # Any B matches this A -> (N_A,)
        any_match = shape_match.max(dim=1).values
        return a_valid * (1.0 - any_match)

    def _extract_first_match(self, batch_objs, pred_mask):
        """Get first object where pred_mask > 0 (trace-compatible)."""
        # Create cumulative mask: first valid object gets weight 1, rest get 0
        cumsum = torch.cumsum(pred_mask, dim=0)
        # First valid = where cumsum jumps from 0 to 1
        first_mask = ((cumsum > 0) & (cumsum <= 1)).float()
        weighted = batch_objs * first_mask.view(-1, 1, 1, 1)
        return weighted.sum(dim=0, keepdim=True)

    def _get_first_valid(self, batch_objs, batch_valid):
        """Get first valid object (trace-compatible, no if-checks)."""
        cumsum = torch.cumsum(batch_valid, dim=0)
        first_mask = ((cumsum > 0) & (cumsum <= 1)).float()
        weighted = batch_objs * first_mask.view(-1, 1, 1, 1)
        return weighted.sum(dim=0, keepdim=True)

    def _get_last_valid(self, batch_objs, batch_valid):
        """Get last valid object (trace-compatible, no if-checks)."""
        total = batch_valid.sum()
        cumsum = torch.cumsum(batch_valid, dim=0)
        last_mask = (cumsum == total).float()
        weighted = batch_objs * last_mask.view(-1, 1, 1, 1)
        return weighted.sum(dim=0, keepdim=True)


# ============================================================================
# Compiler entry point
# ============================================================================

def compile_solver(solver_source: str, task_num: int,
                   output_dir: str = ".",
                   h_in: int = 30, w_in: int = 30,
                   dummy_input=None) -> Optional[str]:
    """Compile solver to ONNX.

    dummy_input: optional (1, 10, 30, 30) tensor with real data for tracing.
                 If None, uses all-zeros (may cause traced branches to be wrong).
    """
    ops = parse_solver(solver_source)
    if not ops:
        return None

    resolver = Resolver()
    resolved_ops = resolver.resolve_all(ops)
    if resolved_ops is None:
        return None

    resolved_ops = fix_crop_args(resolved_ops)
    if not resolved_ops:
        return None

    # Build pred_specs and fn_specs indexed by op id
    pred_specs = {}
    fn_specs = {}
    op_types = {}
    for op in resolved_ops:
        op_id = id(op)
        if "pred_spec" in op.get("kwargs", {}):
            pred_specs[op_id] = op["kwargs"]["pred_spec"]
        if "fn_spec" in op.get("kwargs", {}):
            fn_specs[op_id] = op["kwargs"]["fn_spec"]

    model = SolverModel(resolved_ops, h_in, w_in, pred_specs, fn_specs, op_types)
    if dummy_input is None:
        dummy_input = torch.zeros(1, NUM_COLORS, CANVAS, CANVAS)

    try:
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
            if out is None or out.shape != (1, NUM_COLORS, CANVAS, CANVAS):
                return None

        onnx_path = str(Path(output_dir) / f"task{task_num:03d}.onnx")
        torch.onnx.export(
            model, dummy_input, onnx_path,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
            dynamo=False,
        )
        # Inline external data
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
        import traceback
        traceback.print_exc()
        print(f"  task{task_num:03d}: EXPORT FAILED: {e}")
        return None


def expand_ops(ops):
    """Expand higher-order calls into flat ops."""
    resolver = Resolver()
    return resolver.resolve_all(ops)


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
        if op["func"] == "crop" and len(op.get("args", [])) == 3:
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
    needs_work = []

    for name, source in sorted(solvers.items()):
        ops = parse_solver(source)
        resolved = expand_ops(ops)

        tn = int(name.split("_")[1], 16) if "_" in name else 0
        if resolved is not None:
            compilable.append(tn)
        else:
            needs_work.append(tn)

    print(f"Compilable: {len(compilable)}")
    print(f"Needs work: {len(needs_work)}")
    if needs_work:
        print(f"First 30: {needs_work[:30]}")
    return compilable, needs_work


if __name__ == "__main__":
    analyze_tasks()
