"""ONNX graph-builder-compatible primitives.

Each function:
  - Takes (nodes, inits, tensor_name, h, w, prefix, **kwargs)
  - Appends ONNX nodes/initializers to the shared lists
  - Returns (output_tensor_name, new_h, new_w)
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from typing import List, Tuple, Optional

CANVAS_H = 30
CANVAS_W = 30
NUM_COLORS = 10


# ============================================================================
# Low-level helpers
# ============================================================================

def _node(nodes, op_type, inputs, outputs, **attrs):
    node = helper.make_node(op_type, inputs, outputs, **attrs)
    nodes.append(node)
    return node


def _init(inits, name, array):
    arr = np.asarray(array)
    init = numpy_helper.from_array(arr, name=name)
    inits.append(init)
    return name


def _fresh(prefix, suffix="out"):
    return f"{prefix}_{suffix}"


def _oh_decode(nodes, inits, x, prefix):
    """Decode one-hot (1,10,H,W) to scalar float (1,1,H,W) via ArgMax."""
    out = _fresh(prefix, "dec")
    _node(nodes, "ArgMax", [x], [out], axis=1, keepdims=1)
    f = _fresh(prefix, "decf")
    _node(nodes, "Cast", [out], [f], to=TensorProto.FLOAT)
    return f


def _oh_encode(nodes, inits, scalar, h, w, prefix):
    """Encode scalar float (1,1,H,W) to one-hot (1,10,H,W)."""
    rng = _init(inits, f"{prefix}_rng",
                np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    eq = _fresh(prefix, "eq")
    _node(nodes, "Equal", [scalar, rng], [eq])
    out = _fresh(prefix, "enc")
    _node(nodes, "Cast", [eq], [out], to=TensorProto.FLOAT)
    return out


def make_model(nodes, initializers, task_id, in_h=CANVAS_H, in_w=CANVAS_W):
    graph = helper.make_graph(
        nodes, f"task_{task_id:03d}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                       [1, NUM_COLORS, CANVAS_H, CANVAS_W])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       [1, NUM_COLORS, CANVAS_H, CANVAS_W])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model


# ============================================================================
# Pad to canvas
# ============================================================================

def pad_canvas(nodes, inits, x, h, w, prefix="pad"):
    if h == CANVAS_H and w == CANVAS_W:
        return x, h, w
    pad_h = CANVAS_H - h
    pad_w = CANVAS_W - w
    pads = _init(inits, f"{prefix}_pads",
                 np.array([0, 0, 0, 0, 0, 0, pad_h, pad_w], dtype=np.int64))
    zero = _init(inits, f"{prefix}_val", np.array(0.0, dtype=np.float32))
    out = _fresh(prefix)
    _node(nodes, "Pad", [x, pads, zero], [out], mode="constant")
    return out, CANVAS_H, CANVAS_W


# ============================================================================
# Geometric transforms — pure ONNX
# ============================================================================

def rot90(nodes, inits, x, h, w, prefix="rot90", **kw):
    t = _fresh(prefix, "t")
    _node(nodes, "Transpose", [x], [t], perm=[0, 1, 3, 2])
    idx = _init(inits, f"{prefix}_idx",
                np.array(list(range(h - 1, -1, -1)), dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Gather", [t, idx], [out], axis=3)
    return out, w, h


def rot180(nodes, inits, x, h, w, prefix="r180", **kw):
    r1 = _fresh(prefix, "r1")
    idx_r = _init(inits, f"{prefix}_ir",
                  np.array(list(range(h - 1, -1, -1)), dtype=np.int64))
    _node(nodes, "Gather", [x, idx_r], [r1], axis=2)
    idx_c = _init(inits, f"{prefix}_ic",
                  np.array(list(range(w - 1, -1, -1)), dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Gather", [r1, idx_c], [out], axis=3)
    return out, h, w


def rot270(nodes, inits, x, h, w, prefix="r270", **kw):
    t = _fresh(prefix, "t")
    _node(nodes, "Transpose", [x], [t], perm=[0, 1, 3, 2])
    idx = _init(inits, f"{prefix}_idx",
                np.array(list(range(w - 1, -1, -1)), dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Gather", [t, idx], [out], axis=2)
    return out, w, h


def flip(nodes, inits, x, h, w, prefix="flip", axis=2, **kw):
    """Flip along given axis. axis=2 → hmirror, axis=3 → vmirror."""
    ax_name = "h" if axis == 2 else "v"
    idx = _init(inits, f"{prefix}_idx",
                np.array(list(range(h - 1, -1, -1)) if axis == 2
                         else list(range(w - 1, -1, -1)), dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Gather", [x, idx], [out], axis=axis)
    new_h = w if axis == 2 else h
    new_w = h if axis == 2 else w
    if axis == 2:
        return out, h, w
    return out, h, w


def hmirror(nodes, inits, x, h, w, prefix="hmir", **kw):
    return flip(nodes, inits, x, h, w, prefix, axis=2)


def vmirror(nodes, inits, x, h, w, prefix="vmir", **kw):
    return flip(nodes, inits, x, h, w, prefix, axis=3)


def cmirror(nodes, inits, x, h, w, prefix="cmir", **kw):
    idx_r = _init(inits, f"{prefix}_ir",
                  np.array(list(range(h - 1, -1, -1)), dtype=np.int64))
    t1 = _fresh(prefix, "t1")
    _node(nodes, "Gather", [x, idx_r], [t1], axis=2)
    idx_c = _init(inits, f"{prefix}_ic",
                  np.array(list(range(w - 1, -1, -1)), dtype=np.int64))
    t2 = _fresh(prefix, "t2")
    _node(nodes, "Gather", [t1, idx_c], [t2], axis=3)
    out = _fresh(prefix)
    _node(nodes, "Transpose", [t2], [out], perm=[0, 1, 3, 2])
    return out, w, h


def dmirror(nodes, inits, x, h, w, prefix="dmir", **kw):
    out = _fresh(prefix)
    _node(nodes, "Transpose", [x], [out], perm=[0, 1, 3, 2])
    return out, w, h


# ============================================================================
# Spatial splitting
# ============================================================================

def tophalf(nodes, inits, x, h, w, prefix="top", **kw):
    half = h // 2
    starts = _init(inits, f"{prefix}_s", np.array([0], dtype=np.int64))
    ends = _init(inits, f"{prefix}_e", np.array([half], dtype=np.int64))
    axes = _init(inits, f"{prefix}_a", np.array([2], dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Slice", [x, starts, ends, axes], [out])
    return out, half, w


def bottomhalf(nodes, inits, x, h, w, prefix="bot", **kw):
    half = h // 2
    starts = _init(inits, f"{prefix}_s", np.array([half], dtype=np.int64))
    ends = _init(inits, f"{prefix}_e", np.array([h], dtype=np.int64))
    axes = _init(inits, f"{prefix}_a", np.array([2], dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Slice", [x, starts, ends, axes], [out])
    return out, h - half, w


def lefthalf(nodes, inits, x, h, w, prefix="lef", **kw):
    half = w // 2
    starts = _init(inits, f"{prefix}_s", np.array([0], dtype=np.int64))
    ends = _init(inits, f"{prefix}_e", np.array([half], dtype=np.int64))
    axes = _init(inits, f"{prefix}_a", np.array([3], dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Slice", [x, starts, ends, axes], [out])
    return out, h, half


def righthalf(nodes, inits, x, h, w, prefix="rig", **kw):
    half = w // 2
    starts = _init(inits, f"{prefix}_s", np.array([half], dtype=np.int64))
    ends = _init(inits, f"{prefix}_e", np.array([w], dtype=np.int64))
    axes = _init(inits, f"{prefix}_a", np.array([3], dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Slice", [x, starts, ends, axes], [out])
    return out, h, w - half


# ============================================================================
# Concatenation
# ============================================================================

def vconcat(nodes, inits, x1, x2, h1, h2, w, prefix="vc", **kw):
    out = _fresh(prefix)
    _node(nodes, "Concat", [x1, x2], [out], axis=2)
    return out, h1 + h2, w


def hconcat(nodes, inits, x1, x2, h, w1, w2, prefix="hc", **kw):
    out = _fresh(prefix)
    _node(nodes, "Concat", [x1, x2], [out], axis=3)
    return out, h, w1 + w2


# ============================================================================
# Upscaling
# ============================================================================

def hupscale(nodes, inits, x, h, w, factor=3, prefix="hup", **kw):
    rshp1 = _init(inits, f"{prefix}_sh1",
                  np.array([1, 10, h, w, 1], dtype=np.int64))
    t1 = _fresh(prefix, "rh1")
    _node(nodes, "Reshape", [x, rshp1], [t1])
    reps = _init(inits, f"{prefix}_rp",
                 np.array([1, 1, 1, 1, factor], dtype=np.int64))
    t2 = _fresh(prefix, "tl")
    _node(nodes, "Tile", [t1, reps], [t2])
    rshp2 = _init(inits, f"{prefix}_sh2",
                  np.array([1, 10, h, w * factor], dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Reshape", [t2, rshp2], [out])
    return out, h, w * factor


def vupscale(nodes, inits, x, h, w, factor=3, prefix="vup", **kw):
    rshp1 = _init(inits, f"{prefix}_sh1",
                  np.array([1, 10, h, 1, w], dtype=np.int64))
    t1 = _fresh(prefix, "rh1")
    _node(nodes, "Reshape", [x, rshp1], [t1])
    reps = _init(inits, f"{prefix}_rp",
                 np.array([1, 1, 1, factor, 1], dtype=np.int64))
    t2 = _fresh(prefix, "tl")
    _node(nodes, "Tile", [t1, reps], [t2])
    rshp2 = _init(inits, f"{prefix}_sh2",
                  np.array([1, 10, h * factor, w], dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Reshape", [t2, rshp2], [out])
    return out, h * factor, w


def upscale(nodes, inits, x, h, w, factor=3, prefix="up", **kw):
    rshp1 = _init(inits, f"{prefix}_sh1",
                  np.array([1, 10, h, 1, w, 1], dtype=np.int64))
    t1 = _fresh(prefix, "rh1")
    _node(nodes, "Reshape", [x, rshp1], [t1])
    reps = _init(inits, f"{prefix}_rp",
                 np.array([1, 1, 1, factor, 1, factor], dtype=np.int64))
    t2 = _fresh(prefix, "tl")
    _node(nodes, "Tile", [t1, reps], [t2])
    rshp2 = _init(inits, f"{prefix}_sh2",
                  np.array([1, 10, h * factor, w * factor], dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Reshape", [t2, rshp2], [out])
    return out, h * factor, w * factor


def downscale(nodes, inits, x, h, w, factor=3, prefix="dwn", **kw):
    starts = _init(inits, f"{prefix}_s",
                   np.array([0, 0, 0, 0], dtype=np.int64))
    ends = _init(inits, f"{prefix}_e",
                 np.array([1, NUM_COLORS, h, w], dtype=np.int64))
    axes = _init(inits, f"{prefix}_a",
                 np.array([0, 1, 2, 3], dtype=np.int64))
    steps = _init(inits, f"{prefix}_st",
                  np.array([1, 1, factor, factor], dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Slice", [x, starts, ends, axes, steps], [out])
    return out, h // factor, w // factor


# ============================================================================
# Cropping / trimming
# ============================================================================

def crop(nodes, inits, x, h, w, top=0, left=0, height=None, width=None,
         prefix="crop", **kw):
    if height is None:
        height = h
    if width is None:
        width = w
    starts = _init(inits, f"{prefix}_s",
                   np.array([0, 0, top, left], dtype=np.int64))
    ends = _init(inits, f"{prefix}_e",
                 np.array([1, NUM_COLORS, top + height, left + width], dtype=np.int64))
    axes = _init(inits, f"{prefix}_a",
                 np.array([0, 1, 2, 3], dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Slice", [x, starts, ends, axes], [out])
    return out, height, width


def trim(nodes, inits, x, h, w, prefix="trim", **kw):
    return crop(nodes, inits, x, h, w, top=1, left=1,
                height=h - 2, width=w - 2, prefix=prefix)


# ============================================================================
# Arithmetic
# ============================================================================

def add(nodes, inits, x1, x2, h, w, prefix="add", **kw):
    d1 = _oh_decode(nodes, inits, x1, f"{prefix}_d1")
    d2 = _oh_decode(nodes, inits, x2, f"{prefix}_d2")
    raw = _fresh(prefix, "raw")
    _node(nodes, "Add", [d1, d2], [raw])
    ten = _init(inits, f"{prefix}_10", np.array(10.0, dtype=np.float32))
    mod = _fresh(prefix, "mod")
    _node(nodes, "Mod", [raw, ten], [mod], fmod=1)
    out = _oh_encode(nodes, inits, mod, h, w, f"{prefix}_enc")
    return out, h, w


def subtract(nodes, inits, x1, x2, h, w, prefix="sub", **kw):
    d1 = _oh_decode(nodes, inits, x1, f"{prefix}_d1")
    d2 = _oh_decode(nodes, inits, x2, f"{prefix}_d2")
    raw = _fresh(prefix, "raw")
    _node(nodes, "Sub", [d1, d2], [raw])
    ten = _init(inits, f"{prefix}_10", np.array(10.0, dtype=np.float32))
    mod = _fresh(prefix, "mod")
    _node(nodes, "Mod", [raw, ten], [mod], fmod=1)
    out = _oh_encode(nodes, inits, mod, h, w, f"{prefix}_enc")
    return out, h, w


def multiply(nodes, inits, x1, x2, h, w, prefix="mul", **kw):
    d1 = _oh_decode(nodes, inits, x1, f"{prefix}_d1")
    d2 = _oh_decode(nodes, inits, x2, f"{prefix}_d2")
    raw = _fresh(prefix, "raw")
    _node(nodes, "Mul", [d1, d2], [raw])
    ten = _init(inits, f"{prefix}_10", np.array(10.0, dtype=np.float32))
    mod = _fresh(prefix, "mod")
    _node(nodes, "Mod", [raw, ten], [mod], fmod=1)
    out = _oh_encode(nodes, inits, mod, h, w, f"{prefix}_enc")
    return out, h, w


def divide(nodes, inits, x1, x2, h, w, prefix="div", **kw):
    out = _fresh(prefix)
    _node(nodes, "Div", [x1, x2], [out])
    return out, h, w


def increment(nodes, inits, x, h, w, prefix="inc", delta=1, **kw):
    d = _oh_decode(nodes, inits, x, f"{prefix}_d")
    c = _init(inits, f"{prefix}_c", np.array(float(delta), dtype=np.float32))
    raw = _fresh(prefix, "raw")
    _node(nodes, "Add", [d, c], [raw])
    ten = _init(inits, f"{prefix}_10", np.array(10.0, dtype=np.float32))
    mod = _fresh(prefix, "mod")
    _node(nodes, "Mod", [raw, ten], [mod], fmod=1)
    out = _oh_encode(nodes, inits, mod, h, w, f"{prefix}_enc")
    return out, h, w


def decrement(nodes, inits, x, h, w, prefix="dec", delta=1, **kw):
    d = _oh_decode(nodes, inits, x, f"{prefix}_d")
    c = _init(inits, f"{prefix}_c", np.array(float(delta), dtype=np.float32))
    neg = _fresh(prefix, "neg")
    _node(nodes, "Neg", [c], [neg])
    raw = _fresh(prefix, "raw")
    _node(nodes, "Add", [d, neg], [raw])
    ten = _init(inits, f"{prefix}_10", np.array(10.0, dtype=np.float32))
    mod = _fresh(prefix, "mod")
    _node(nodes, "Mod", [raw, ten], [mod], fmod=1)
    out = _oh_encode(nodes, inits, mod, h, w, f"{prefix}_enc")
    return out, h, w


def crement(nodes, inits, x, h, w, prefix="cre", delta=1, **kw):
    return increment(nodes, inits, x, h, w, prefix, delta)


def double(nodes, inits, x, h, w, prefix="dbl", **kw):
    d = _oh_decode(nodes, inits, x, f"{prefix}_d")
    c = _init(inits, f"{prefix}_c", np.array(2.0, dtype=np.float32))
    raw = _fresh(prefix, "raw")
    _node(nodes, "Mul", [d, c], [raw])
    ten = _init(inits, f"{prefix}_10", np.array(10.0, dtype=np.float32))
    mod = _fresh(prefix, "mod")
    _node(nodes, "Mod", [raw, ten], [mod], fmod=1)
    out = _oh_encode(nodes, inits, mod, h, w, f"{prefix}_enc")
    return out, h, w


def negate(nodes, inits, x, h, w, prefix="neg", **kw):
    out = _fresh(prefix)
    _node(nodes, "Neg", [x], [out])
    return out, h, w


def minimum(nodes, inits, x1, x2, h, w, prefix="min", **kw):
    out = _fresh(prefix)
    _node(nodes, "Min", [x1, x2], [out])
    return out, h, w


def maximum(nodes, inits, x1, x2, h, w, prefix="max", **kw):
    out = _fresh(prefix)
    _node(nodes, "Max", [x1, x2], [out])
    return out, h, w


# ============================================================================
# Cellwise / comparison
# ============================================================================

def cellwise(nodes, inits, x1, x2, h, w, func="add", prefix="cell", **kw):
    op_map = {"add": "Add", "sub": "Sub", "mul": "Mul",
              "div": "Div", "max": "Max", "min": "Min"}
    op = op_map.get(func, "Add")
    out = _fresh(prefix)
    _node(nodes, op, [x1, x2], [out])
    return out, h, w


def both(nodes, inits, x1, x2, h, w, prefix="both", **kw):
    d1 = _oh_decode(nodes, inits, x1, f"{prefix}_d1")
    d2 = _oh_decode(nodes, inits, x2, f"{prefix}_d2")
    z = _init(inits, f"{prefix}_z", np.array(0.5, dtype=np.float32))
    b1 = _fresh(prefix, "b1")
    _node(nodes, "Greater", [d1, z], [b1])
    b2 = _fresh(prefix, "b2")
    _node(nodes, "Greater", [d2, z], [b2])
    a = _fresh(prefix, "a")
    _node(nodes, "And", [b1, b2], [a])
    f = _fresh(prefix, "f")
    _node(nodes, "Cast", [a], [f], to=TensorProto.FLOAT)
    one = _init(inits, f"{prefix}_1", np.array(1.0, dtype=np.float32))
    inv = _fresh(prefix, "inv")
    _node(nodes, "Sub", [one, f], [inv])
    zeros29 = _init(inits, f"{prefix}_z29", np.zeros((1, 8, h, w), dtype=np.float32))
    out = _fresh(prefix)
    _node(nodes, "Concat", [inv, f, zeros29], [out], axis=1)
    return out, h, w


def either(nodes, inits, x1, x2, h, w, prefix="eith", **kw):
    d1 = _oh_decode(nodes, inits, x1, f"{prefix}_d1")
    d2 = _oh_decode(nodes, inits, x2, f"{prefix}_d2")
    z = _init(inits, f"{prefix}_z", np.array(0.5, dtype=np.float32))
    b1 = _fresh(prefix, "b1")
    _node(nodes, "Greater", [d1, z], [b1])
    b2 = _fresh(prefix, "b2")
    _node(nodes, "Greater", [d2, z], [b2])
    a = _fresh(prefix, "a")
    _node(nodes, "Or", [b1, b2], [a])
    f = _fresh(prefix, "f")
    _node(nodes, "Cast", [a], [f], to=TensorProto.FLOAT)
    one = _init(inits, f"{prefix}_1", np.array(1.0, dtype=np.float32))
    inv = _fresh(prefix, "inv")
    _node(nodes, "Sub", [one, f], [inv])
    zeros29 = _init(inits, f"{prefix}_z29", np.zeros((1, 8, h, w), dtype=np.float32))
    out = _fresh(prefix)
    _node(nodes, "Concat", [inv, f, zeros29], [out], axis=1)
    return out, h, w


def equality(nodes, inits, x1, x2, h, w, prefix="eq", **kw):
    d1 = _oh_decode(nodes, inits, x1, f"{prefix}_d1")
    d2 = _oh_decode(nodes, inits, x2, f"{prefix}_d2")
    eq = _fresh(prefix, "eq")
    _node(nodes, "Equal", [d1, d2], [eq])
    f = _fresh(prefix, "f")
    _node(nodes, "Cast", [eq], [f], to=TensorProto.FLOAT)
    one = _init(inits, f"{prefix}_1", np.array(1.0, dtype=np.float32))
    inv = _fresh(prefix, "inv")
    _node(nodes, "Sub", [one, f], [inv])
    zeros29 = _init(inits, f"{prefix}_z29", np.zeros((1, 8, h, w), dtype=np.float32))
    out = _fresh(prefix)
    _node(nodes, "Concat", [inv, f, zeros29], [out], axis=1)
    return out, h, w


def greater(nodes, inits, x1, x2, h, w, prefix="gt", **kw):
    d1 = _oh_decode(nodes, inits, x1, f"{prefix}_d1")
    d2 = _oh_decode(nodes, inits, x2, f"{prefix}_d2")
    gt = _fresh(prefix, "gt")
    _node(nodes, "Greater", [d1, d2], [gt])
    f = _fresh(prefix, "f")
    _node(nodes, "Cast", [gt], [f], to=TensorProto.FLOAT)
    one = _init(inits, f"{prefix}_1", np.array(1.0, dtype=np.float32))
    inv = _fresh(prefix, "inv")
    _node(nodes, "Sub", [one, f], [inv])
    zeros29 = _init(inits, f"{prefix}_z29", np.zeros((1, 8, h, w), dtype=np.float32))
    out = _fresh(prefix)
    _node(nodes, "Concat", [inv, f, zeros29], [out], axis=1)
    return out, h, w


def less(nodes, inits, x1, x2, h, w, prefix="lt", **kw):
    d1 = _oh_decode(nodes, inits, x1, f"{prefix}_d1")
    d2 = _oh_decode(nodes, inits, x2, f"{prefix}_d2")
    lt = _fresh(prefix, "lt")
    _node(nodes, "Less", [d1, d2], [lt])
    f = _fresh(prefix, "f")
    _node(nodes, "Cast", [lt], [f], to=TensorProto.FLOAT)
    one = _init(inits, f"{prefix}_1", np.array(1.0, dtype=np.float32))
    inv = _fresh(prefix, "inv")
    _node(nodes, "Sub", [one, f], [inv])
    zeros29 = _init(inits, f"{prefix}_z29", np.zeros((1, 8, h, w), dtype=np.float32))
    out = _fresh(prefix)
    _node(nodes, "Concat", [inv, f, zeros29], [out], axis=1)
    return out, h, w


def even(nodes, inits, x, h, w, prefix="even", **kw):
    two = _init(inits, f"{prefix}_2", np.array(2.0, dtype=np.float32))
    rem = _fresh(prefix, "rem")
    _node(nodes, "Mod", [x, two], [rem], fmod=1)
    z = _init(inits, f"{prefix}_z", np.array(0.0, dtype=np.float32))
    eq = _fresh(prefix, "eq")
    _node(nodes, "Equal", [rem, z], [eq])
    out = _fresh(prefix)
    _node(nodes, "Cast", [eq], [out], to=TensorProto.FLOAT)
    return out, h, w


def sign(nodes, inits, x, h, w, prefix="sign", **kw):
    out = _fresh(prefix)
    _node(nodes, "Sign", [x], [out])
    return out, h, w


def positive(nodes, inits, x, h, w, prefix="pos", **kw):
    z = _init(inits, f"{prefix}_z", np.array(0.0, dtype=np.float32))
    gt = _fresh(prefix, "gt")
    _node(nodes, "Greater", [x, z], [gt])
    out = _fresh(prefix)
    _node(nodes, "Cast", [gt], [out], to=TensorProto.FLOAT)
    return out, h, w


def invert(nodes, inits, x, h, w, prefix="inv", **kw):
    neg = _fresh(prefix, "neg")
    _node(nodes, "Neg", [x], [neg])
    one = _init(inits, f"{prefix}_1", np.array(1.0, dtype=np.float32))
    out = _fresh(prefix)
    _node(nodes, "Add", [neg, one], [out])
    return out, h, w


# ============================================================================
# Color filtering / replacement (ONNX-based)
# ============================================================================

def ofcolor(nodes, inits, x, h, w, color=1, prefix="ofc", **kw):
    d = _oh_decode(nodes, inits, x, f"{prefix}_d")
    c = _init(inits, f"{prefix}_c", np.array(float(color), dtype=np.float32))
    eq = _fresh(prefix, "eq")
    _node(nodes, "Equal", [d, c], [eq])
    f = _fresh(prefix, "f")
    _node(nodes, "Cast", [eq], [f], to=TensorProto.FLOAT)
    one = _init(inits, f"{prefix}_1", np.array(1.0, dtype=np.float32))
    inv = _fresh(prefix, "inv")
    _node(nodes, "Sub", [one, f], [inv])
    zeros29 = _init(inits, f"{prefix}_z29", np.zeros((1, 8, h, w), dtype=np.float32))
    out = _fresh(prefix)
    _node(nodes, "Concat", [inv, f, zeros29], [out], axis=1)
    return out, h, w


def fill(nodes, inits, x, mask, h, w, color=1, prefix="fill", **kw):
    m_idx = _fresh(prefix, "midx")
    _node(nodes, "ArgMax", [mask], [m_idx], axis=1, keepdims=1)
    f = _fresh(prefix, "mf")
    _node(nodes, "Cast", [m_idx], [f], to=TensorProto.FLOAT)
    z = _init(inits, f"{prefix}_z", np.array(0.5, dtype=np.float32))
    m_bool = _fresh(prefix, "mb")
    _node(nodes, "Greater", [f, z], [m_bool])
    m = _fresh(prefix, "m")
    _node(nodes, "Cast", [m_bool], [m], to=TensorProto.FLOAT)
    one_hot_color = np.zeros((1, NUM_COLORS, h, w), dtype=np.float32)
    one_hot_color[0, color] = 1.0
    col = _init(inits, f"{prefix}_c", one_hot_color)
    fg = _fresh(prefix, "fg")
    _node(nodes, "Mul", [m, col], [fg])
    one = _init(inits, f"{prefix}_1", np.array(1.0, dtype=np.float32))
    inv_m = _fresh(prefix, "im")
    _node(nodes, "Sub", [one, m], [inv_m])
    bg = _fresh(prefix, "bg")
    _node(nodes, "Mul", [inv_m, x], [bg])
    out = _fresh(prefix)
    _node(nodes, "Add", [bg, fg], [out])
    return out, h, w


def replace(nodes, inits, x, h, w, old_color=0, new_color=1, prefix="rpl", **kw):
    ch_idx = _init(inits, f"{prefix}_ch",
                   np.array([old_color], dtype=np.int64))
    val = _fresh(prefix, "val")
    _node(nodes, "Gather", [x, ch_idx], [val], axis=1)
    z = _init(inits, f"{prefix}_z", np.array(0.5, dtype=np.float32))
    gt = _fresh(prefix, "gt")
    _node(nodes, "Greater", [val, z], [gt])
    mask_2d = _fresh(prefix, "m2d")
    _node(nodes, "Cast", [gt], [mask_2d], to=TensorProto.FLOAT)
    one_hot_new = np.zeros((1, NUM_COLORS, h, w), dtype=np.float32)
    one_hot_new[0, new_color] = 1.0
    new_col = _init(inits, f"{prefix}_nc", one_hot_new)
    rshp = _init(inits, f"{prefix}_sh",
                 np.array([1, 1, h, w], dtype=np.int64))
    mask_4d = _fresh(prefix, "m4d")
    _node(nodes, "Reshape", [mask_2d, rshp], [mask_4d])
    fill_val = _fresh(prefix, "fv")
    _node(nodes, "Mul", [mask_4d, new_col], [fill_val])
    one = _init(inits, f"{prefix}_1", np.array(1.0, dtype=np.float32))
    inv = _fresh(prefix, "inv")
    _node(nodes, "Sub", [one, mask_4d], [inv])
    keep = _fresh(prefix, "keep")
    _node(nodes, "Mul", [inv, x], [keep])
    out = _fresh(prefix)
    _node(nodes, "Add", [keep, fill_val], [out])
    return out, h, w


def underfill(nodes, inits, x, mask, h, w, color=1, prefix="ufill", **kw):
    return fill(nodes, inits, x, mask, h, w, color, prefix)


def underpaint(nodes, inits, x, obj, h, w, prefix="upaint", **kw):
    return fill(nodes, inits, x, obj, h, w, 1, prefix)


def cover(nodes, inits, x1, x2, h, w, prefix="cov", **kw):
    return fill(nodes, inits, x1, x2, h, w, 0, prefix)


def merge(nodes, inits, x1, x2, h, w, prefix="mrg", **kw):
    d1 = _oh_decode(nodes, inits, x1, f"{prefix}_d1")
    d2 = _oh_decode(nodes, inits, x2, f"{prefix}_d2")
    raw = _fresh(prefix)
    _node(nodes, "Max", [d1, d2], [raw])
    out = _oh_encode(nodes, inits, raw, h, w, f"{prefix}_enc")
    return out, h, w


def paint(nodes, inits, canvas, obj_mask, h, w, color=1, prefix="paint", **kw):
    return fill(nodes, inits, canvas, obj_mask, h, w, color, prefix)


# ============================================================================
# Canvas
# ============================================================================

def canvas(nodes, inits, color=0, h=CANVAS_H, w=CANVAS_W, prefix="cvs", **kw):
    arr = np.zeros((1, NUM_COLORS, h, w), dtype=np.float32)
    arr[0, color] = 1.0
    name = _init(inits, f"{prefix}_data", arr)
    out = _fresh(prefix)
    _node(nodes, "Identity", [name], [out])
    return out, h, w


def canvas_like(nodes, inits, x, h, w, color=0, prefix="cvsl", **kw):
    return canvas(nodes, inits, color, h, w, prefix)


# ============================================================================
# Shape / size (constant propagation via initializers)
# ============================================================================

def height(nodes, inits, x, h, w, prefix="ht", **kw):
    val = np.array(float(h), dtype=np.float32)
    name = _init(inits, f"{prefix}_val", val.reshape(1, 1, 1, 1))
    out = _fresh(prefix)
    _node(nodes, "Identity", [name], [out])
    return out, 1, 1


def width(nodes, inits, x, h, w, prefix="wd", **kw):
    val = np.array(float(w), dtype=np.float32)
    name = _init(inits, f"{prefix}_val", val.reshape(1, 1, 1, 1))
    out = _fresh(prefix)
    _node(nodes, "Identity", [name], [out])
    return out, 1, 1


def shape(nodes, inits, x, h, w, prefix="shp", **kw):
    h_val = np.array(float(h), dtype=np.float32).reshape(1, 1, 1, 1)
    w_val = np.array(float(w), dtype=np.float32).reshape(1, 1, 1, 1)
    h_name = _init(inits, f"{prefix}_h", h_val)
    w_name = _init(inits, f"{prefix}_w", w_val)
    out = _fresh(prefix)
    _node(nodes, "Concat", [h_name, w_name], [out], axis=3)
    return out, 1, 2


def size(nodes, inits, x, h, w, prefix="sz", **kw):
    val = np.array(float(h * w), dtype=np.float32).reshape(1, 1, 1, 1)
    name = _init(inits, f"{prefix}_val", val)
    out = _fresh(prefix)
    _node(nodes, "Identity", [name], [out])
    return out, 1, 1


def numcolors(nodes, inits, x, h, w, prefix="nclr", **kw):
    flat = _fresh(prefix, "flat")
    _node(nodes, "Reshape", [x, _init(inits, f"{prefix}_rsh",
           np.array([-1], dtype=np.int64))], [flat])
    u = _fresh(prefix, "u")
    _node(nodes, "Unique", [flat], [u, _fresh(prefix, "idx"),
          _fresh(prefix, "cnt"), _fresh(prefix, "nc")])
    nc_out = _fresh(prefix, "ncnt")
    _node(nodes, "Shape", [u], [nc_out])
    out = _fresh(prefix)
    _node(nodes, "Cast", [nc_out], [out], to=TensorProto.FLOAT)
    return out, 1, 1


# ============================================================================
# Color extraction
# ============================================================================

def color(nodes, inits, x, h, w, prefix="clr", **kw):
    flat = _fresh(prefix, "flat")
    _node(nodes, "Reshape", [x, _init(inits, f"{prefix}_rsh",
           np.array([-1], dtype=np.int64))], [flat])
    u = _fresh(prefix, "u")
    cnt = _fresh(prefix, "cnt")
    idx = _fresh(prefix, "idx")
    nc = _fresh(prefix, "nc")
    _node(nodes, "Unique", [flat], [u, idx, cnt, nc])
    max_idx = _fresh(prefix, "maxi")
    _node(nodes, "ArgMax", [cnt], [max_idx], axis=0, keepdims=1)
    out = _fresh(prefix)
    _node(nodes, "Gather", [u, max_idx], [out])
    return out, 1, 1


def palette(nodes, inits, x, h, w, prefix="pal", **kw):
    flat = _fresh(prefix, "flat")
    _node(nodes, "Reshape", [x, _init(inits, f"{prefix}_rsh",
           np.array([-1], dtype=np.int64))], [flat])
    u = _fresh(prefix, "u")
    cnt = _fresh(prefix, "cnt")
    idx = _fresh(prefix, "idx")
    nc = _fresh(prefix, "nc")
    _node(nodes, "Unique", [flat], [u, idx, cnt, nc])
    out = _fresh(prefix)
    _node(nodes, "Reshape", [u, _init(inits, f"{prefix}_sh",
           np.array([1, -1], dtype=np.int64))], [out])
    return out, 1, w


# ============================================================================
# Index operations
# ============================================================================

def asindices(nodes, inits, x, h, w, prefix="aidx", **kw):
    r = np.arange(h, dtype=np.float32).reshape(1, 1, h, 1)
    c = np.arange(w, dtype=np.float32).reshape(1, 1, 1, w)
    r_arr = np.tile(r, (1, 1, 1, w))
    c_arr = np.tile(c, (1, 1, h, 1))
    r_name = _init(inits, f"{prefix}_r", r_arr)
    c_name = _init(inits, f"{prefix}_c", c_arr)
    out = _fresh(prefix)
    _node(nodes, "Concat", [r_name, c_name], [out], axis=1)
    return out, h, w


def asobject(nodes, inits, x, h, w, prefix="aobj", **kw):
    mask = _fresh(prefix, "mask")
    z = _init(inits, f"{prefix}_z", np.array(0.0, dtype=np.float32))
    _node(nodes, "Greater", [x, z], [mask])
    out = _fresh(prefix)
    _node(nodes, "Cast", [mask], [out], to=TensorProto.FLOAT)
    return out, h, w


# ============================================================================
# Position / corner extraction
# ============================================================================

def center(nodes, inits, x, h, w, prefix="ctr", **kw):
    hr = np.array(float(h // 2), dtype=np.float32).reshape(1, 1, 1, 1)
    wr = np.array(float(w // 2), dtype=np.float32).reshape(1, 1, 1, 1)
    h_name = _init(inits, f"{prefix}_h", hr)
    w_name = _init(inits, f"{prefix}_w", wr)
    out = _fresh(prefix)
    _node(nodes, "Concat", [h_name, w_name], [out], axis=3)
    return out, 1, 2


def ulcorner(nodes, inits, x, h, w, prefix="ulc", **kw):
    return center(nodes, inits, x, 1, 2, prefix)


def urcorner(nodes, inits, x, h, w, prefix="urc", **kw):
    v = np.array([0.0, float(w - 1)], dtype=np.float32).reshape(1, 1, 1, 2)
    name = _init(inits, f"{prefix}_v", v)
    out = _fresh(prefix)
    _node(nodes, "Identity", [name], [out])
    return out, 1, 2


def llcorner(nodes, inits, x, h, w, prefix="llc", **kw):
    v = np.array([float(h - 1), 0.0], dtype=np.float32).reshape(1, 1, 1, 2)
    name = _init(inits, f"{prefix}_v", v)
    out = _fresh(prefix)
    _node(nodes, "Identity", [name], [out])
    return out, 1, 2


def lrcorner(nodes, inits, x, h, w, prefix="lrc", **kw):
    v = np.array([float(h - 1), float(w - 1)], dtype=np.float32).reshape(1, 1, 1, 2)
    name = _init(inits, f"{prefix}_v", v)
    out = _fresh(prefix)
    _node(nodes, "Identity", [name], [out])
    return out, 1, 2


def uppermost(nodes, inits, x, h, w, prefix="upm", **kw):
    flat = _fresh(prefix, "flat")
    _node(nodes, "Reshape", [x, _init(inits, f"{prefix}_r",
           np.array([-1], dtype=np.int64))], [flat])
    nz = _fresh(prefix, "nz")
    _node(nodes, "NonZero", [flat], [nz])
    first = _fresh(prefix, "f")
    _node(nodes, "Gather", [nz, _init(inits, f"{prefix}_i",
           np.array([0], dtype=np.int64))], [first])
    out = _fresh(prefix)
    _node(nodes, "Gather", [first, _init(inits, f"{prefix}_j",
           np.array([0], dtype=np.int64))], [out])
    return out, 1, 1


def lowermost(nodes, inits, x, h, w, prefix="low", **kw):
    flat = _fresh(prefix, "flat")
    _node(nodes, "Reshape", [x, _init(inits, f"{prefix}_r",
           np.array([-1], dtype=np.int64))], [flat])
    nz = _fresh(prefix, "nz")
    _node(nodes, "NonZero", [flat], [nz])
    shape_nz = _fresh(prefix, "snz")
    _node(nodes, "Shape", [nz], [shape_nz])
    last_idx = _fresh(prefix, "li")
    two = _init(inits, f"{prefix}_2", np.array(2, dtype=np.int64))
    _node(nodes, "Sub", [shape_nz, two], [last_idx])
    row = _fresh(prefix, "row")
    _node(nodes, "Gather", [nz, last_idx], [row])
    out = _fresh(prefix)
    _node(nodes, "Gather", [row, _init(inits, f"{prefix}_j",
           np.array([0], dtype=np.int64))], [out])
    return out, 1, 1


def leftmost(nodes, inits, x, h, w, prefix="lft", **kw):
    flat = _fresh(prefix, "flat")
    _node(nodes, "Reshape", [x, _init(inits, f"{prefix}_r",
           np.array([-1], dtype=np.int64))], [flat])
    nz = _fresh(prefix, "nz")
    _node(nodes, "NonZero", [flat], [nz])
    first = _fresh(prefix, "f")
    _node(nodes, "Gather", [nz, _init(inits, f"{prefix}_i",
           np.array([0], dtype=np.int64))], [first])
    out = _fresh(prefix)
    _node(nodes, "Gather", [first, _init(inits, f"{prefix}_j",
           np.array([1], dtype=np.int64))], [out])
    return out, 1, 1


def rightmost(nodes, inits, x, h, w, prefix="rgt", **kw):
    flat = _fresh(prefix, "flat")
    _node(nodes, "Reshape", [x, _init(inits, f"{prefix}_r",
           np.array([-1], dtype=np.int64))], [flat])
    nz = _fresh(prefix, "nz")
    _node(nodes, "NonZero", [flat], [nz])
    shape_nz = _fresh(prefix, "snz")
    _node(nodes, "Shape", [nz], [shape_nz])
    last_idx = _fresh(prefix, "li")
    two = _init(inits, f"{prefix}_2", np.array(2, dtype=np.int64))
    _node(nodes, "Sub", [shape_nz, two], [last_idx])
    row = _fresh(prefix, "row")
    _node(nodes, "Gather", [nz, last_idx], [row])
    out = _fresh(prefix)
    _node(nodes, "Gather", [row, _init(inits, f"{prefix}_j",
           np.array([1], dtype=np.int64))], [out])
    return out, 1, 1


# ============================================================================
# Object detection — ONNX-compatible approximations
# ============================================================================

def objects(nodes, inits, x, h, w, prefix="objs", **kw):
    return asobject(nodes, inits, x, h, w, prefix)


def partition(nodes, inits, x, h, w, prefix="part", **kw):
    return asobject(nodes, inits, x, h, w, prefix)


def fgpartition(nodes, inits, x, h, w, prefix="fgp", **kw):
    return asobject(nodes, inits, x, h, w, prefix)


def mfilter(nodes, inits, x, h, w, prefix="mf", **kw):
    return x, h, w


def sfilter(nodes, inits, x, h, w, prefix="sf", **kw):
    return x, h, w


def sizefilter(nodes, inits, x, h, w, prefix="sfilt", **kw):
    return x, h, w


# ============================================================================
# Shift / move — index-based
# ============================================================================

def shift(nodes, inits, x, h, w, di=0, dj=0, prefix="shf", **kw):
    pad_top = max(0, -di)
    pad_bot = max(0, di)
    pad_left = max(0, -dj)
    pad_right = max(0, dj)
    if pad_top == 0 and pad_bot == 0 and pad_left == 0 and pad_right == 0:
        src_r = list(range(di, h + di))
        src_c = list(range(dj, w + dj))
    else:
        src_r = list(range(max(0, di), min(h, h + di)))
        src_c = list(range(max(0, dj), min(w, w + dj)))
    out = _fresh(prefix)
    if len(src_r) == h and len(src_c) == w:
        _node(nodes, "Identity", [x], [out])
        return out, h, w
    out_h = min(h, max(1, len(src_r)))
    out_w = min(w, max(1, len(src_c)))
    _node(nodes, "Identity", [x], [out])
    return out, h, w


def move(nodes, inits, x, h, w, di=0, dj=0, prefix="mov", **kw):
    return shift(nodes, inits, x, h, w, di, dj, prefix)


# ============================================================================
# Splitting helpers
# ============================================================================

def hsplit(nodes, inits, x, h, w, prefix="hspl", **kw):
    hw = w // 2
    starts = _init(inits, f"{prefix}_s", np.array([0], dtype=np.int64))
    ends = _init(inits, f"{prefix}_e", np.array([hw], dtype=np.int64))
    axes = _init(inits, f"{prefix}_a", np.array([3], dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Slice", [x, starts, ends, axes], [out])
    return out, h, hw


def vsplit(nodes, inits, x, h, w, prefix="vspl", **kw):
    hh = h // 2
    starts = _init(inits, f"{prefix}_s", np.array([0], dtype=np.int64))
    ends = _init(inits, f"{prefix}_e", np.array([hh], dtype=np.int64))
    axes = _init(inits, f"{prefix}_a", np.array([2], dtype=np.int64))
    out = _fresh(prefix)
    _node(nodes, "Slice", [x, starts, ends, axes], [out])
    return out, hh, w


# ============================================================================
# Frontiers
# ============================================================================

def hfrontier(nodes, inits, x, h, w, prefix="hfr", **kw):
    flat = _fresh(prefix, "flat")
    _node(nodes, "Reshape", [x, _init(inits, f"{prefix}_r",
           np.array([-1], dtype=np.int64))], [flat])
    nz = _fresh(prefix, "nz")
    _node(nodes, "NonZero", [flat], [nz])
    row_vals = _fresh(prefix, "rv")
    _node(nodes, "Gather", [nz, _init(inits, f"{prefix}_j",
           np.array([0], dtype=np.int64))], [row_vals])
    unique_r = _fresh(prefix, "ur")
    _node(nodes, "Unique", [row_vals], [unique_r])
    mask = _fresh(prefix, "mask")
    z = _init(inits, f"{prefix}_z", np.array(0.0, dtype=np.float32))
    _node(nodes, "Cast", [unique_r], [mask], to=TensorProto.FLOAT)
    out = _fresh(prefix)
    _node(nodes, "Reshape", [mask, _init(inits, f"{prefix}_sh",
           np.array([1, 1, -1, 1], dtype=np.int64))], [out])
    return out, h, 1


def vfrontier(nodes, inits, x, h, w, prefix="vfr", **kw):
    flat = _fresh(prefix, "flat")
    _node(nodes, "Reshape", [x, _init(inits, f"{prefix}_r",
           np.array([-1], dtype=np.int64))], [flat])
    nz = _fresh(prefix, "nz")
    _node(nodes, "NonZero", [flat], [nz])
    col_vals = _fresh(prefix, "cv")
    _node(nodes, "Gather", [nz, _init(inits, f"{prefix}_j",
           np.array([1], dtype=np.int64))], [col_vals])
    unique_c = _fresh(prefix, "uc")
    _node(nodes, "Unique", [col_vals], [unique_c])
    mask = _fresh(prefix, "mask")
    _node(nodes, "Cast", [unique_c], [mask], to=TensorProto.FLOAT)
    out = _fresh(prefix)
    _node(nodes, "Reshape", [mask, _init(inits, f"{prefix}_sh",
           np.array([1, 1, 1, -1], dtype=np.int64))], [out])
    return out, 1, w


# ============================================================================
# Delta / box / backdrop / inbox / outbox — identity pass-through
# ============================================================================

def delta(nodes, inits, x, h, w, prefix="dlt", **kw):
    return x, h, w


def box(nodes, inits, x, h, w, prefix="box", **kw):
    return x, h, w


def backdrop(nodes, inits, x, h, w, prefix="bd", **kw):
    return x, h, w


def inbox(nodes, inits, x, h, w, prefix="inb", **kw):
    return x, h, w


def outbox(nodes, inits, x, h, w, prefix="outb", **kw):
    return x, h, w


def corners(nodes, inits, x, h, w, prefix="crn", **kw):
    return x, h, w


def frontiers(nodes, inits, x, h, w, prefix="frt", **kw):
    return x, h, w


# ============================================================================
# Line detection — identity pass-through
# ============================================================================

def hline(nodes, inits, x, h, w, prefix="hl", **kw):
    return x, h, w


def vline(nodes, inits, x, h, w, prefix="vl", **kw):
    return x, h, w


def connect(nodes, inits, x, h, w, prefix="con", **kw):
    return x, h, w


def shoot(nodes, inits, x, h, w, prefix="sho", **kw):
    return x, h, w


# ============================================================================
# Period detection — identity pass-through
# ============================================================================

def hperiod(nodes, inits, x, h, w, prefix="hp", **kw):
    val = np.array(1.0, dtype=np.float32).reshape(1, 1, 1, 1)
    name = _init(inits, f"{prefix}_v", val)
    out = _fresh(prefix)
    _node(nodes, "Identity", [name], [out])
    return out, 1, 1


def vperiod(nodes, inits, x, h, w, prefix="vp", **kw):
    val = np.array(1.0, dtype=np.float32).reshape(1, 1, 1, 1)
    name = _init(inits, f"{prefix}_v", val)
    out = _fresh(prefix)
    _node(nodes, "Identity", [name], [out])
    return out, 1, 1


def portrait(nodes, inits, x, h, w, prefix="prt", **kw):
    val = np.array(float(1 if h > w else 0), dtype=np.float32).reshape(1, 1, 1, 1)
    name = _init(inits, f"{prefix}_v", val)
    out = _fresh(prefix)
    _node(nodes, "Identity", [name], [out])
    return out, 1, 1


# ============================================================================
# Utility / tuple operations — pass-through
# ============================================================================

def astuple(nodes, inits, x, h, w, prefix="atpl", **kw):
    return x, h, w


def initset(nodes, inits, x, h, w, prefix="iset", **kw):
    return x, h, w


def totuple(nodes, inits, x, h, w, prefix="ttpl", **kw):
    return x, h, w


def combine(nodes, inits, x1, x2, h, w, prefix="cmb", **kw):
    return x1, h, w


def pair(nodes, inits, x1, x2, h, w, prefix="pr", **kw):
    return x1, h, w


def product(nodes, inits, x1, x2, h, w, prefix="prod", **kw):
    return x1, h, w


def insert(nodes, inits, x, h, w, prefix="ins", **kw):
    return x, h, w


def remove(nodes, inits, x, h, w, prefix="rem", **kw):
    return x, h, w


def other(nodes, inits, x, h, w, prefix="oth", **kw):
    return x, h, w


def first(nodes, inits, x, h, w, prefix="fst", **kw):
    return x, h, w


def last(nodes, inits, x, h, w, prefix="lst", **kw):
    return x, h, w


def extract(nodes, inits, x, h, w, prefix="ext", **kw):
    return x, h, w


def dedupe(nodes, inits, x, h, w, prefix="ddp", **kw):
    return x, h, w


def contained(nodes, inits, x, h, w, prefix="cnt", **kw):
    return x, h, w


def normalize(nodes, inits, x, h, w, prefix="nrm", **kw):
    return x, h, w


def compress(nodes, inits, x, h, w, prefix="cmp", **kw):
    return x, h, w


def index(nodes, inits, x, h, w, prefix="idx", **kw):
    return x, h, w


def interval(nodes, inits, x, h, w, prefix="ivl", **kw):
    return x, h, w


def toivec(nodes, inits, x, h, w, prefix="tiv", **kw):
    return x, h, w


def tojvec(nodes, inits, x, h, w, prefix="tjv", **kw):
    return x, h, w


def toindices(nodes, inits, x, h, w, prefix="tidx", **kw):
    return x, h, w


def toobject(nodes, inits, x, h, w, prefix="tobj", **kw):
    return x, h, w


def occurrences(nodes, inits, x, h, w, prefix="occ", **kw):
    return x, h, w


def position(nodes, inits, x, h, w, prefix="pos", **kw):
    return x, h, w


def mostcolor(nodes, inits, x, h, w, prefix="mcl", **kw):
    return x, h, w


def leastcolor(nodes, inits, x, h, w, prefix="lcl", **kw):
    return x, h, w


def mostcommon(nodes, inits, x, h, w, prefix="mcm", **kw):
    return x, h, w


def leastcommon(nodes, inits, x, h, w, prefix="lcm", **kw):
    return x, h, w


def colorcount(nodes, inits, x, h, w, prefix="cc", **kw):
    return x, h, w


def valmax(nodes, inits, x, h, w, prefix="vmax", **kw):
    return x, h, w


def valmin(nodes, inits, x, h, w, prefix="vmin", **kw):
    return x, h, w


def matcher(nodes, inits, x, h, w, prefix="mch", **kw):
    return x, h, w


def switch(nodes, inits, x, h, w, prefix="sw", **kw):
    return x, h, w


def minimum(nodes, inits, x1, x2, h, w, prefix="mn", **kw):
    d1 = _oh_decode(nodes, inits, x1, f"{prefix}_d1")
    d2 = _oh_decode(nodes, inits, x2, f"{prefix}_d2")
    raw = _fresh(prefix)
    _node(nodes, "Min", [d1, d2], [raw])
    out = _oh_encode(nodes, inits, raw, h, w, f"{prefix}_enc")
    return out, h, w


def maximum(nodes, inits, x1, x2, h, w, prefix="mx", **kw):
    d1 = _oh_decode(nodes, inits, x1, f"{prefix}_d1")
    d2 = _oh_decode(nodes, inits, x2, f"{prefix}_d2")
    raw = _fresh(prefix)
    _node(nodes, "Max", [d1, d2], [raw])
    out = _oh_encode(nodes, inits, raw, h, w, f"{prefix}_enc")
    return out, h, w


# ============================================================================
# Neighbor detection — identity pass-through
# ============================================================================

def neighbors(nodes, inits, x, h, w, prefix="nbs", **kw):
    return x, h, w


def dneighbors(nodes, inits, x, h, w, prefix="dnb", **kw):
    return x, h, w


# ============================================================================
# Gravitate — identity pass-through
# ============================================================================

def gravitate(nodes, inits, x, h, w, prefix="grv", **kw):
    return x, h, w


# ============================================================================
# Higher-order combinators — STUBS (waiting for user spec)
# ============================================================================

def compose(nodes, inits, x, h, w, prefix="cmp", **kw):
    return x, h, w


def chain(nodes, inits, x, h, w, prefix="chn", **kw):
    return x, h, w


def fork(nodes, inits, x, h, w, prefix="frk", **kw):
    return x, h, w


def lbind(nodes, inits, x, h, w, prefix="lb", **kw):
    return x, h, w


def rbind(nodes, inits, x, h, w, prefix="rb", **kw):
    return x, h, w


def apply(nodes, inits, x, h, w, prefix="apl", **kw):
    return x, h, w


def mapply(nodes, inits, x, h, w, prefix="map", **kw):
    return x, h, w


def branch(nodes, inits, x, h, w, prefix="br", **kw):
    return x, h, w


def power(nodes, inits, x, h, w, prefix="pow", **kw):
    return x, h, w


def repeat(nodes, inits, x, h, w, prefix="rpt", **kw):
    return x, h, w


def order(nodes, inits, x, h, w, prefix="ord", **kw):
    return x, h, w


def mpapply(nodes, inits, x, h, w, prefix="mpa", **kw):
    return x, h, w


def prapply(nodes, inits, x, h, w, prefix="pra", **kw):
    return x, h, w


def papply(nodes, inits, x, h, w, prefix="pap", **kw):
    return x, h, w


def rapply(nodes, inits, x, h, w, prefix="rap", **kw):
    return x, h, w


# ============================================================================
# Stack manipulation helpers (used by graph_builder)
# ============================================================================

def dup(nodes, inits, x, h, w, prefix="dup", **kw):
    return x, h, w


def swap(nodes, inits, x, h, w, prefix="swp", **kw):
    return x, h, w
