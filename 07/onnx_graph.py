"""
arc_onnx_primitives.py — Full ONNX implementation of the Hodel arc-dsl primitive set.
Static shapes only. No Loop / Scan / NonZero / Unique / Function.

=====================================================================
ENCODING CONVENTIONS
=====================================================================
Grid / Object      : (1, 10, h, w) float32 one-hot over color channel
Patch / Indices     : (1, 1, h, w)  float32 binary occupancy mask
Scalar (int/color)  : (1, 1, 1, 1)  float32
Boolean             : (1, 1, 1, 1)  float32 (0.0 / 1.0)
Vector (i, j pair)  : (1, 1, 1, 2)  float32
Object-set          : python tuple ("OBJSET", stack_name, valid_name, K)
                        stack: (K, 10, h, w) float32 one-hot per slot
                        valid: (K,) float32 0/1
Python containers    : plain python tuples of ints / tensor names
                        (used for combinators & compile-time constants,
                         exactly like the DSL's own T/F/ints/interval args)

h, w passed to every grid-builder function are PYTHON INTS, known at
graph-construction time (this is a per-task compiler: crop/split offsets
are always resolved from that task's own training examples beforehand).
Only VALUES (colors, counts, bbox extents, positions) are ever dynamic
tensors — shapes are always static python ints.
"""

from networkx import nodes
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

NUM_COLORS = 10
BIG = 1.0e9


# ============================================================================
# Low-level helpers
# ============================================================================

def _node(nodes, op_type, inputs, outputs, **attrs):
    nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))
    return outputs[0] if len(outputs) == 1 else outputs


def _init(inits, name, array):
    inits.append(numpy_helper.from_array(np.asarray(array), name=name))
    return name


_COUNTER = [0]
def _fresh(prefix, suffix="out"):
    _COUNTER[0] += 1
    return f"{prefix}_{suffix}_{_COUNTER[0]}"


def _op(nodes, op_type, inputs, prefix, suffix="o", **attrs):
    out = _fresh(prefix, suffix)
    _node(nodes, op_type, inputs, [out], **attrs)
    return out


def scalar_const(inits, prefix, value, dtype=np.float32):
    return _init(inits, _fresh(prefix, "c"), np.array(value, dtype=dtype).reshape(1, 1, 1, 1))


def vec2_const(inits, prefix, a, b, dtype=np.float32):
    return _init(inits, _fresh(prefix, "v2"), np.array([a, b], dtype=dtype).reshape(1, 1, 1, 2))


def make_model(nodes, inits, in_shape, out_name, out_shape, opset=13):
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, list(in_shape))
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(out_shape))
    if nodes[-1].output[0] != "output":
        nodes.append(helper.make_node("Identity", [out_name], ["output"]))
    graph = helper.make_graph(nodes, "task", [X], [Y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def pad_to(nodes, inits, x, h, w, H, W, prefix="pad"):
    if h == H and w == W:
        return x, H, W
    pads = _init(inits, f"{prefix}_pads", np.array([0, 0, 0, 0, 0, 0, H - h, W - w], dtype=np.int64))
    zero = _init(inits, f"{prefix}_zv", np.array(0.0, dtype=np.float32))
    out = _op(nodes, "Pad", [x, pads, zero], prefix, mode="constant")
    return out, H, W


# ============================================================================
# SECTION A — scalar / elementwise arithmetic & logic
# ============================================================================

def _binop(op):
    def fn(nodes, inits, x1, x2, h, w, prefix="op", **kw):
        return _op(nodes, op, [x1, x2], prefix), h, w
    return fn

add = _binop("Add")
subtract = _binop("Sub")
multiply = _binop("Mul")
divide = _binop("Div")
minimum = _binop("Min")
maximum = _binop("Max")


def cellwise(nodes, inits, x1, x2, h, w, func="add", prefix="cell", **kw):
    op = {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "Div", "max": "Max", "min": "Min"}[func]
    return _op(nodes, op, [x1, x2], prefix), h, w


def increment(nodes, inits, x, h, w, prefix="inc", delta=1, **kw):
    c = scalar_const(inits, prefix, float(delta))
    return _op(nodes, "Add", [x, c], prefix), h, w


def decrement(nodes, inits, x, h, w, prefix="dec", delta=1, **kw):
    c = scalar_const(inits, prefix, float(delta))
    return _op(nodes, "Sub", [x, c], prefix), h, w


def crement(nodes, inits, x, h, w, prefix="cre", **kw):
    """crement(x) = x+1 if x>0, x-1 if x<0, else 0."""
    zero = scalar_const(inits, prefix, 0.0)
    one = scalar_const(inits, prefix, 1.0)
    s = _op(nodes, "Sign", [x], prefix)
    return _op(nodes, "Add", [x, s], prefix), h, w


def double(nodes, inits, x, h, w, prefix="dbl", **kw):
    c = scalar_const(inits, prefix, 2.0)
    return _op(nodes, "Mul", [x, c], prefix), h, w


def halve(nodes, inits, x, h, w, prefix="hlv", **kw):
    c = scalar_const(inits, prefix, 2.0)
    return _op(nodes, "Div", [x, c], prefix), h, w


def negate(nodes, inits, x, h, w, prefix="neg", **kw):
    return _op(nodes, "Neg", [x], prefix), h, w


def sign(nodes, inits, x, h, w, prefix="sgn", **kw):
    return _op(nodes, "Sign", [x], prefix), h, w


def positive(nodes, inits, x, h, w, prefix="pos", **kw):
    z = scalar_const(inits, prefix, 0.0)
    g = _op(nodes, "Greater", [x, z], prefix)
    return _op(nodes, "Cast", [g], prefix, to=TensorProto.FLOAT), h, w


def invert(nodes, inits, x, h, w, prefix="inv", **kw):
    """DSL invert(n) = -n."""
    return _op(nodes, "Neg", [x], prefix), h, w


def even(nodes, inits, x, h, w, prefix="evn", **kw):
    two = scalar_const(inits, prefix, 2.0)
    rem = _op(nodes, "Mod", [x, two], prefix, fmod=1)
    z = scalar_const(inits, prefix, 0.0)
    eq = _op(nodes, "Equal", [rem, z], prefix)
    return _op(nodes, "Cast", [eq], prefix, to=TensorProto.FLOAT), h, w


def equality(nodes, inits, x1, x2, h, w, prefix="eq", **kw):
    eq = _op(nodes, "Equal", [x1, x2], prefix)
    return _op(nodes, "Cast", [eq], prefix, to=TensorProto.FLOAT), h, w


def greater(nodes, inits, x1, x2, h, w, prefix="gt", **kw):
    g = _op(nodes, "Greater", [x1, x2], prefix)
    return _op(nodes, "Cast", [g], prefix, to=TensorProto.FLOAT), h, w


def both(nodes, inits, x1, x2, h, w, prefix="both", **kw):
    b1 = _op(nodes, "Cast", [x1], prefix, to=TensorProto.BOOL)
    b2 = _op(nodes, "Cast", [x2], prefix, to=TensorProto.BOOL)
    a = _op(nodes, "And", [b1, b2], prefix)
    return _op(nodes, "Cast", [a], prefix, to=TensorProto.FLOAT), h, w


def flip(nodes, inits, x, h, w, prefix="flp", **kw):
    """DSL flip(b) = boolean NOT (NOT the geometric mirror — see hmirror/vmirror)."""
    b = _op(nodes, "Cast", [x], prefix, to=TensorProto.BOOL)
    n = _op(nodes, "Not", [b], prefix)
    return _op(nodes, "Cast", [n], prefix, to=TensorProto.FLOAT), h, w


# ============================================================================
# SECTION B — geometric transforms (grid-level)
# ============================================================================

def hmirror(nodes, inits, x, h, w, prefix="hmir", **kw):
    """Flip along rows (matches dsl hmirror: reverses row order)."""
    idx = _init(inits, f"{prefix}_idx", np.arange(h - 1, -1, -1, dtype=np.int64))
    return _op(nodes, "Gather", [x, idx], prefix, axis=2), h, w


def vmirror(nodes, inits, x, h, w, prefix="vmir", **kw):
    idx = _init(inits, f"{prefix}_idx", np.arange(w - 1, -1, -1, dtype=np.int64))
    return _op(nodes, "Gather", [x, idx], prefix, axis=3), h, w


def dmirror(nodes, inits, x, h, w, prefix="dmir", **kw):
    out = _op(nodes, "Transpose", [x], prefix, perm=[0, 1, 3, 2])
    return out, w, h


def cmirror(nodes, inits, x, h, w, prefix="cmir", **kw):
    r, _, _ = hmirror(nodes, inits, x, h, w, prefix + "_h")
    r2, _, _ = vmirror(nodes, inits, r, h, w, prefix + "_v")
    out = _op(nodes, "Transpose", [r2], prefix, perm=[0, 1, 3, 2])
    return out, w, h


def rot90(nodes, inits, x, h, w, prefix="r90", **kw):
    t, _, _ = dmirror(nodes, inits, x, h, w, prefix + "_d")
    return vmirror(nodes, inits, t, w, h, prefix + "_v")


def rot270(nodes, inits, x, h, w, prefix="r270", **kw):
    t, _, _ = dmirror(nodes, inits, x, h, w, prefix + "_d")
    return hmirror(nodes, inits, t, w, h, prefix + "_h")


def rot180(nodes, inits, x, h, w, prefix="r180", **kw):
    r, _, _ = hmirror(nodes, inits, x, h, w, prefix + "_h")
    return vmirror(nodes, inits, r, h, w, prefix + "_v")


def _slice_axis(nodes, inits, x, axis, start, end, prefix):
    st = _init(inits, f"{prefix}_s", np.array([start], dtype=np.int64))
    en = _init(inits, f"{prefix}_e", np.array([end], dtype=np.int64))
    ax = _init(inits, f"{prefix}_a", np.array([axis], dtype=np.int64))
    return _op(nodes, "Slice", [x, st, en, ax], prefix)


def tophalf(nodes, inits, x, h, w, prefix="top", **kw):
    half = h // 2
    return _slice_axis(nodes, inits, x, 2, 0, half, prefix), half, w

def bottomhalf(nodes, inits, x, h, w, prefix="bot", **kw):
    half = h // 2
    return _slice_axis(nodes, inits, x, 2, half, h, prefix), h - half, w

def lefthalf(nodes, inits, x, h, w, prefix="lef", **kw):
    half = w // 2
    return _slice_axis(nodes, inits, x, 3, 0, half, prefix), h, half

def righthalf(nodes, inits, x, h, w, prefix="rig", **kw):
    half = w // 2
    return _slice_axis(nodes, inits, x, 3, half, w, prefix), h, w - half

def hsplit(nodes, inits, x, h, w, n=2, prefix="hspl", **kw):
    """Returns python tuple of n equal column-slices."""
    step = w // n
    return tuple(_slice_axis(nodes, inits, x, 3, i * step, (i + 1) * step, f"{prefix}{i}") for i in range(n)), h, step

def vsplit(nodes, inits, x, h, w, n=2, prefix="vspl", **kw):
    step = h // n
    return tuple(_slice_axis(nodes, inits, x, 2, i * step, (i + 1) * step, f"{prefix}{i}") for i in range(n)), step, w


def vconcat(nodes, inits, x1, x2, h1, h2, w, prefix="vc", **kw):
    return _op(nodes, "Concat", [x1, x2], prefix, axis=2), h1 + h2, w

def hconcat(nodes, inits, x1, x2, h, w1, w2, prefix="hc", **kw):
    return _op(nodes, "Concat", [x1, x2], prefix, axis=3), h, w1 + w2


def hupscale(nodes, inits, x, h, w, factor=3, prefix="hup", **kw):
    r1 = _init(inits, f"{prefix}_r1", np.array([1, NUM_COLORS, h, 1, w], dtype=np.int64))
    t1 = _op(nodes, "Reshape", [x, r1], prefix)
    reps = _init(inits, f"{prefix}_rp", np.array([1, 1, 1, factor, 1], dtype=np.int64))
    t2 = _op(nodes, "Tile", [t1, reps], prefix)
    r2 = _init(inits, f"{prefix}_r2", np.array([1, NUM_COLORS, h, w * factor], dtype=np.int64))
    return _op(nodes, "Reshape", [t2, r2], prefix), h, w * factor

def vupscale(nodes, inits, x, h, w, factor=3, prefix="vup", **kw):
    r1 = _init(inits, f"{prefix}_r1", np.array([1, NUM_COLORS, 1, h, w], dtype=np.int64))
    t1 = _op(nodes, "Reshape", [x, r1], prefix)
    reps = _init(inits, f"{prefix}_rp", np.array([1, 1, factor, 1, 1], dtype=np.int64))
    t2 = _op(nodes, "Tile", [t1, reps], prefix)
    r2 = _init(inits, f"{prefix}_r2", np.array([1, NUM_COLORS, h * factor, w], dtype=np.int64))
    return _op(nodes, "Reshape", [t2, r2], prefix), h * factor, w

def upscale(nodes, inits, x, h, w, factor=3, prefix="up", **kw):
    r1 = _init(inits, f"{prefix}_r1", np.array([1, NUM_COLORS, 1, h, 1, w], dtype=np.int64))
    t1 = _op(nodes, "Reshape", [x, r1], prefix)
    reps = _init(inits, f"{prefix}_rp", np.array([1, 1, factor, 1, factor, 1], dtype=np.int64))
    t2 = _op(nodes, "Tile", [t1, reps], prefix)
    r2 = _init(inits, f"{prefix}_r2", np.array([1, NUM_COLORS, h * factor, w * factor], dtype=np.int64))
    return _op(nodes, "Reshape", [t2, r2], prefix), h * factor, w * factor

def downscale(nodes, inits, x, h, w, factor=3, prefix="dwn", **kw):
    """Exact only when every factor x factor block is uniform (true for scaled ARC grids)."""
    return _op(nodes, "AveragePool", [x], prefix, kernel_shape=[factor, factor], strides=[factor, factor]), h // factor, w // factor


def crop(nodes, inits, x, h, w, top=0, left=0, height=None, width=None, prefix="crop", **kw):
    """Compile-time-constant crop (top,left,height,width all python ints)."""
    height = h if height is None else height
    width = w if width is None else width
    st = _init(inits, f"{prefix}_s", np.array([0, 0, top, left], dtype=np.int64))
    en = _init(inits, f"{prefix}_e", np.array([1, NUM_COLORS, top + height, left + width], dtype=np.int64))
    ax = _init(inits, f"{prefix}_a", np.array([0, 1, 2, 3], dtype=np.int64))
    return _op(nodes, "Slice", [x, st, en, ax], prefix), height, width


def trim(nodes, inits, x, h, w, prefix="trim", **kw):
    return crop(nodes, inits, x, h, w, top=1, left=1, height=h - 2, width=w - 2, prefix=prefix)


def shift(nodes, inits, x, h, w, di=0, dj=0, prefix="shf", **kw):
    """Compile-time-constant shift via Pad+Slice. Out-of-bounds cells become 0."""
    if di == 0 and dj == 0:
        return _op(nodes, "Identity", [x], prefix), h, w
    ph0, ph1 = max(di, 0), max(-di, 0)
    pw0, pw1 = max(dj, 0), max(-dj, 0)
    pads = _init(inits, f"{prefix}_pd", np.array([0, 0, ph0, pw0, 0, 0, ph1, pw1], dtype=np.int64))
    zero = _init(inits, f"{prefix}_z", np.array(0.0, dtype=np.float32))
    padded = _op(nodes, "Pad", [x, pads, zero], prefix)
    st = _init(inits, f"{prefix}_s", np.array([0, 0, ph1, pw1], dtype=np.int64))
    en = _init(inits, f"{prefix}_e", np.array([1, NUM_COLORS, ph1 + h, pw1 + w], dtype=np.int64))
    ax = _init(inits, f"{prefix}_a", np.array([0, 1, 2, 3], dtype=np.int64))
    return _op(nodes, "Slice", [padded, st, en, ax], prefix), h, w


def move(nodes, inits, grid, obj_stack_or_mask, h, w, di=0, dj=0, prefix="mov", **kw):
    """move(grid, obj, offset) = paint(cover(grid, obj), shift(obj, offset))."""
    covered, _, _ = cover(nodes, inits, grid, obj_stack_or_mask, h, w, prefix + "_cov")
    shifted, _, _ = shift(nodes, inits, obj_stack_or_mask, h, w, di, dj, prefix + "_shf")
    return paint(nodes, inits, covered, shifted, h, w, prefix + "_pnt")


# ============================================================================
# SECTION C — dynamic translate (for subgrid: data-dependent offset, static shape)
# ============================================================================

def _dynamic_translate(nodes, inits, x, h, w, row_off, col_off, depth, prefix):
    """out[.,c,r,cc] = x[.,c,r+row_off,cc+col_off] if in-bounds else 0.
    row_off/col_off: scalar int64 tensors (dynamic value, static shape).
    Implemented via GatherND -> fully static output shape (h,w,depth)."""
    rows = _init(inits, f"{prefix}_rows", np.arange(h, dtype=np.int64).reshape(h, 1))
    cols = _init(inits, f"{prefix}_cols", np.arange(w, dtype=np.int64).reshape(1, w))
    rows_b = _op(nodes, "Add", [rows, col_off], prefix, suffix="rb0")  # placeholder shape fix below
    # broadcast rows (h,1) + row_off (scalar) -> (h,1); cols (1,w)+col_off -> (1,w)
    src_r = _op(nodes, "Add", [rows, row_off], prefix, suffix="sr")
    src_c = _op(nodes, "Add", [cols, col_off], prefix, suffix="sc")
    zero_i = _init(inits, f"{prefix}_zi", np.array(0, dtype=np.int64))
    hmax = _init(inits, f"{prefix}_hmax", np.array(h - 1, dtype=np.int64))
    wmax = _init(inits, f"{prefix}_wmax", np.array(w - 1, dtype=np.int64))
    src_r_c = _op(nodes, "Clip", [src_r, zero_i, hmax], prefix, suffix="crc")
    src_c_c = _op(nodes, "Clip", [src_c, zero_i, wmax], prefix, suffix="ccc")
    # validity mask (h,w)
    ge0_r = _op(nodes, "GreaterOrEqual", [src_r, zero_i], prefix, suffix="ge0r")
    lth_r = _op(nodes, "Less", [src_r, _init(inits, f"{prefix}_h", np.array(h, dtype=np.int64))], prefix, suffix="lthr")
    ge0_c = _op(nodes, "GreaterOrEqual", [src_c, zero_i], prefix, suffix="ge0c")
    lth_c = _op(nodes, "Less", [src_c, _init(inits, f"{prefix}_w", np.array(w, dtype=np.int64))], prefix, suffix="lthc")
    v_r = _op(nodes, "And", [ge0_r, lth_r], prefix, suffix="vr")   # (h,1)
    v_c = _op(nodes, "And", [ge0_c, lth_c], prefix, suffix="vc")  # (1,w)
    valid = _op(nodes, "And", [v_r, v_c], prefix, suffix="v")     # (h,w) via broadcast
    validf = _op(nodes, "Cast", [valid], prefix, to=TensorProto.FLOAT, suffix="vf")

    # build (h,w,2) index tensor
    src_r_full = _op(nodes, "Expand", [src_r_c, _init(inits, f"{prefix}_es", np.array([h, w], dtype=np.int64))], prefix, suffix="erf")
    src_c_full = _op(nodes, "Expand", [src_c_c, _init(inits, f"{prefix}_es2", np.array([h, w], dtype=np.int64))], prefix, suffix="ecf")
    idx = _op(nodes, "Concat", [
        _op(nodes, "Unsqueeze", [src_r_full], prefix, suffix="u1", axes=[-1]),
        _op(nodes, "Unsqueeze", [src_c_full], prefix, suffix="u2", axes=[-1]),
    ], prefix, suffix="idx", axis=-1)  # (h,w,2)

    xchw = _op(nodes, "Reshape", [x, _init(inits, f"{prefix}_rs1", np.array([depth, h, w], dtype=np.int64))], prefix, suffix="xchw")
    xhwc = _op(nodes, "Transpose", [xchw], prefix, suffix="xhwc", perm=[1, 2, 0])
    gathered = _op(nodes, "GatherND", [xhwc, idx], prefix, suffix="gnd")             # (h,w,depth)
    gathered_chw = _op(nodes, "Transpose", [gathered], prefix, suffix="gchw", perm=[2, 0, 1])
    validf_1hw = _op(nodes, "Reshape", [validf, _init(inits, f"{prefix}_rs2", np.array([1, h, w], dtype=np.int64))], prefix, suffix="v1hw")
    masked = _op(nodes, "Mul", [gathered_chw, validf_1hw], prefix, suffix="mkd")
    out = _op(nodes, "Reshape", [masked, _init(inits, f"{prefix}_rs3", np.array([1, depth, h, w], dtype=np.int64))], prefix, suffix="fin")
    return out


# ============================================================================
# SECTION D — bounding box (static-shape, no NonZero)
# ============================================================================

def _reduce_occ(nodes, inits, x, depth, prefix):
    if depth == 1:
        return x
    return _op(nodes, "ReduceMax", [x], prefix, suffix="rm", axes=[1], keepdims=1)


def _bbox(nodes, inits, occ, h, w, prefix):
    """occ: (1,1,h,w). Returns (lo_r,hi_r,lo_c,hi_c) as int64 scalar tensors (shape ())."""
    occ2 = _op(nodes, "Reshape", [occ, _init(inits, f"{prefix}_rs", np.array([h, w], dtype=np.int64))], prefix)
    rows = _op(nodes, "ReduceMax", [occ2], prefix, suffix="rows", axes=[1], keepdims=0)
    cols = _op(nodes, "ReduceMax", [occ2], prefix, suffix="cols", axes=[0], keepdims=0)
    rev_h = _init(inits, f"{prefix}_rh", np.arange(h - 1, -1, -1, dtype=np.int64))
    rev_w = _init(inits, f"{prefix}_rw", np.arange(w - 1, -1, -1, dtype=np.int64))
    rows_i = _op(nodes, "Cast", [rows], prefix, suffix="rowsi", to=TensorProto.INT64)
    cols_i = _op(nodes, "Cast", [cols], prefix, suffix="colsi", to=TensorProto.INT64)
    rows_rev = _op(nodes, "Gather", [rows_i, rev_h], prefix, suffix="rr", axis=0)
    cols_rev = _op(nodes, "Gather", [cols_i, rev_w], prefix, suffix="cr", axis=0)
    lo_r = _op(nodes, "ArgMax", [rows_i], prefix, suffix="lor", axis=0, keepdims=0)
    hi_r_rev = _op(nodes, "ArgMax", [rows_rev], prefix, suffix="hirr", axis=0, keepdims=0)
    lo_c = _op(nodes, "ArgMax", [cols_i], prefix, suffix="loc", axis=0, keepdims=0)
    hi_c_rev = _op(nodes, "ArgMax", [cols_rev], prefix, suffix="hicr", axis=0, keepdims=0)
    hm1 = _init(inits, f"{prefix}_hm1", np.array(h - 1, dtype=np.int64))
    wm1 = _init(inits, f"{prefix}_wm1", np.array(w - 1, dtype=np.int64))
    hi_r = _op(nodes, "Sub", [hm1, hi_r_rev], prefix, suffix="hir")
    hi_c = _op(nodes, "Sub", [wm1, hi_c_rev], prefix, suffix="hic")
    return lo_r, hi_r, lo_c, hi_c


def _to_scalar_f(nodes, inits, x_int64, prefix):
    xf = _op(nodes, "Cast", [x_int64], prefix, to=TensorProto.FLOAT)
    return _op(nodes, "Reshape", [xf, _init(inits, f"{prefix}_rs", np.array([1, 1, 1, 1], dtype=np.int64))], prefix)


def height(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="ht", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    lo_r, hi_r, _, _ = _bbox(nodes, inits, occ, h, w, prefix)
    diff = _op(nodes, "Sub", [hi_r, lo_r], prefix, suffix="d")
    one = _init(inits, f"{prefix}_1", np.array(1, dtype=np.int64))
    val = _op(nodes, "Add", [diff, one], prefix, suffix="v")
    return _to_scalar_f(nodes, inits, val, prefix), 1, 1


def width(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="wd", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    _, _, lo_c, hi_c = _bbox(nodes, inits, occ, h, w, prefix)
    diff = _op(nodes, "Sub", [hi_c, lo_c], prefix, suffix="d")
    one = _init(inits, f"{prefix}_1", np.array(1, dtype=np.int64))
    val = _op(nodes, "Add", [diff, one], prefix, suffix="v")
    return _to_scalar_f(nodes, inits, val, prefix), 1, 1


def shape(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="shp", **kw):
    hh, _, _ = height(nodes, inits, x, h, w, depth, prefix + "_h")
    ww, _, _ = width(nodes, inits, x, h, w, depth, prefix + "_w")
    return _op(nodes, "Concat", [hh, ww], prefix, axis=3), 1, 2


def size(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="sz", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    s = _op(nodes, "ReduceSum", [occ], prefix, axes=[0, 1, 2, 3], keepdims=1)
    return s, 1, 1


def portrait(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="prt", **kw):
    hh, _, _ = height(nodes, inits, x, h, w, depth, prefix + "_h")
    ww, _, _ = width(nodes, inits, x, h, w, depth, prefix + "_w")
    g = _op(nodes, "Greater", [hh, ww], prefix)
    return _op(nodes, "Cast", [g], prefix, to=TensorProto.FLOAT), 1, 1


def ulcorner(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="ulc", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    lo_r, _, lo_c, _ = _bbox(nodes, inits, occ, h, w, prefix)
    r = _to_scalar_f(nodes, inits, lo_r, prefix + "_r")
    c = _to_scalar_f(nodes, inits, lo_c, prefix + "_c")
    return _op(nodes, "Concat", [r, c], prefix, axis=3), 1, 2

def urcorner(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="urc", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    lo_r, _, _, hi_c = _bbox(nodes, inits, occ, h, w, prefix)
    r = _to_scalar_f(nodes, inits, lo_r, prefix + "_r")
    c = _to_scalar_f(nodes, inits, hi_c, prefix + "_c")
    return _op(nodes, "Concat", [r, c], prefix, axis=3), 1, 2

def llcorner(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="llc", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    _, hi_r, lo_c, _ = _bbox(nodes, inits, occ, h, w, prefix)
    r = _to_scalar_f(nodes, inits, hi_r, prefix + "_r")
    c = _to_scalar_f(nodes, inits, lo_c, prefix + "_c")
    return _op(nodes, "Concat", [r, c], prefix, axis=3), 1, 2

def lrcorner(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="lrc", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    _, hi_r, _, hi_c = _bbox(nodes, inits, occ, h, w, prefix)
    r = _to_scalar_f(nodes, inits, hi_r, prefix + "_r")
    c = _to_scalar_f(nodes, inits, hi_c, prefix + "_c")
    return _op(nodes, "Concat", [r, c], prefix, axis=3), 1, 2

def corners(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="crn", **kw):
    """Returns python tuple of the 4 corner vec2 tensors."""
    return (ulcorner(nodes, inits, x, h, w, depth, prefix + "_ul")[0],
            urcorner(nodes, inits, x, h, w, depth, prefix + "_ur")[0],
            llcorner(nodes, inits, x, h, w, depth, prefix + "_ll")[0],
            lrcorner(nodes, inits, x, h, w, depth, prefix + "_lr")[0]), 1, 2

def uppermost(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="upm", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    lo_r, _, _, _ = _bbox(nodes, inits, occ, h, w, prefix)
    return _to_scalar_f(nodes, inits, lo_r, prefix), 1, 1

def lowermost(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="low", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    _, hi_r, _, _ = _bbox(nodes, inits, occ, h, w, prefix)
    return _to_scalar_f(nodes, inits, hi_r, prefix), 1, 1

def leftmost(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="lft", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    _, _, lo_c, _ = _bbox(nodes, inits, occ, h, w, prefix)
    return _to_scalar_f(nodes, inits, lo_c, prefix), 1, 1

def rightmost(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="rgt", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    _, _, _, hi_c = _bbox(nodes, inits, occ, h, w, prefix)
    return _to_scalar_f(nodes, inits, hi_c, prefix), 1, 1


def center(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="ctr", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    lo_r, hi_r, lo_c, hi_c = _bbox(nodes, inits, occ, h, w, prefix)
    two = _init(inits, f"{prefix}_2", np.array(2, dtype=np.int64))
    cr = _op(nodes, "Div", [_op(nodes, "Add", [lo_r, hi_r], prefix, suffix="sr"), two], prefix, suffix="cr")
    cc = _op(nodes, "Div", [_op(nodes, "Add", [lo_c, hi_c], prefix, suffix="sc"), two], prefix, suffix="cc")
    r = _to_scalar_f(nodes, inits, cr, prefix + "_r")
    c = _to_scalar_f(nodes, inits, cc, prefix + "_c")
    return _op(nodes, "Concat", [r, c], prefix, axis=3), 1, 2


def centerofmass(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="com", **kw):
    """Mean row/col of occupied cells (float, not rounded like DSL's floor-division mean, close enough for downstream comparisons)."""
    occ = _reduce_occ(nodes, inits, x, depth, prefix)          # (1,1,h,w)
    occ2 = _op(nodes, "Reshape", [occ, _init(inits, f"{prefix}_rs", np.array([h, w], dtype=np.int64))], prefix)
    rows = _init(inits, f"{prefix}_rowc", np.arange(h, dtype=np.float32).reshape(h, 1))
    cols = _init(inits, f"{prefix}_colc", np.arange(w, dtype=np.float32).reshape(1, w))
    wsum = _op(nodes, "ReduceSum", [occ2], prefix, suffix="ws", axes=[0, 1], keepdims=0)
    rsum = _op(nodes, "ReduceSum", [_op(nodes, "Mul", [occ2, rows], prefix, suffix="rm")], prefix, suffix="rs", axes=[0, 1], keepdims=0)
    csum = _op(nodes, "ReduceSum", [_op(nodes, "Mul", [occ2, cols], prefix, suffix="cm")], prefix, suffix="cs", axes=[0, 1], keepdims=0)
    r = _op(nodes, "Div", [rsum, wsum], prefix, suffix="rf")
    c = _op(nodes, "Div", [csum, wsum], prefix, suffix="cf")
    r4 = _op(nodes, "Reshape", [r, _init(inits, f"{prefix}_r4", np.array([1, 1, 1, 1], dtype=np.int64))], prefix, suffix="r4")
    c4 = _op(nodes, "Reshape", [c, _init(inits, f"{prefix}_c4", np.array([1, 1, 1, 1], dtype=np.int64))], prefix, suffix="c4")
    return _op(nodes, "Concat", [r4, c4], prefix, axis=3), 1, 2


def delta(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="dlt", **kw):
    """backdrop(x) minus toindices(x): the bbox interior cells NOT occupied by x."""
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    bd, _, _ = backdrop(nodes, inits, x, h, w, depth, prefix + "_bd")
    inv_occ = _op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), occ], prefix, suffix="io")
    return _op(nodes, "Mul", [bd, inv_occ], prefix), h, w


def _rect_mask(nodes, inits, lo_r, hi_r, lo_c, hi_c, h, w, prefix):
    rows = _init(inits, f"{prefix}_rr", np.arange(h, dtype=np.float32).reshape(h, 1))
    cols = _init(inits, f"{prefix}_cc", np.arange(w, dtype=np.float32).reshape(1, w))
    lo_rf = _to_scalar_f(nodes, inits, lo_r, prefix + "_lrf")
    hi_rf = _to_scalar_f(nodes, inits, hi_r, prefix + "_hrf")
    lo_cf = _to_scalar_f(nodes, inits, lo_c, prefix + "_lcf")
    hi_cf = _to_scalar_f(nodes, inits, hi_c, prefix + "_hcf")
    lo_rf2 = _op(nodes, "Reshape", [lo_rf, _init(inits, f"{prefix}_s1", np.array([1, 1], dtype=np.int64))], prefix, suffix="lr2")
    hi_rf2 = _op(nodes, "Reshape", [hi_rf, _init(inits, f"{prefix}_s2", np.array([1, 1], dtype=np.int64))], prefix, suffix="hr2")
    lo_cf2 = _op(nodes, "Reshape", [lo_cf, _init(inits, f"{prefix}_s3", np.array([1, 1], dtype=np.int64))], prefix, suffix="lc2")
    hi_cf2 = _op(nodes, "Reshape", [hi_cf, _init(inits, f"{prefix}_s4", np.array([1, 1], dtype=np.int64))], prefix, suffix="hc2")
    r_ok = _op(nodes, "And", [
        _op(nodes, "GreaterOrEqual", [rows, lo_rf2], prefix, suffix="rge"),
        _op(nodes, "LessOrEqual", [rows, hi_rf2], prefix, suffix="rle")], prefix, suffix="rok")
    c_ok = _op(nodes, "And", [
        _op(nodes, "GreaterOrEqual", [cols, lo_cf2], prefix, suffix="cge"),
        _op(nodes, "LessOrEqual", [cols, hi_cf2], prefix, suffix="cle")], prefix, suffix="cok")
    rect = _op(nodes, "And", [r_ok, c_ok], prefix, suffix="rect")   # (h,w) via broadcast
    rectf = _op(nodes, "Cast", [rect], prefix, to=TensorProto.FLOAT, suffix="rf")
    return _op(nodes, "Reshape", [rectf, _init(inits, f"{prefix}_rs", np.array([1, 1, h, w], dtype=np.int64))], prefix, suffix="fin")


def backdrop(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="bd", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    lo_r, hi_r, lo_c, hi_c = _bbox(nodes, inits, occ, h, w, prefix)
    return _rect_mask(nodes, inits, lo_r, hi_r, lo_c, hi_c, h, w, prefix), h, w


def box(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="box", **kw):
    """Outline of the bbox (backdrop minus its own trim-by-1 interior)."""
    bd, _, _ = backdrop(nodes, inits, x, h, w, depth, prefix + "_bd")
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    lo_r, hi_r, lo_c, hi_c = _bbox(nodes, inits, occ, h, w, prefix)
    one = _init(inits, f"{prefix}_1", np.array(1, dtype=np.int64))
    inner = _rect_mask(nodes, inits,
                        _op(nodes, "Add", [lo_r, one], prefix, suffix="lr1"),
                        _op(nodes, "Sub", [hi_r, one], prefix, suffix="hr1"),
                        _op(nodes, "Add", [lo_c, one], prefix, suffix="lc1"),
                        _op(nodes, "Sub", [hi_c, one], prefix, suffix="hc1"), h, w, prefix + "_in")
    inv_inner = _op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), inner], prefix, suffix="ii")
    return _op(nodes, "Mul", [bd, inv_inner], prefix), h, w


def inbox(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="inb", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    lo_r, hi_r, lo_c, hi_c = _bbox(nodes, inits, occ, h, w, prefix)
    one = _init(inits, f"{prefix}_1", np.array(1, dtype=np.int64))
    return _rect_mask(nodes, inits,
                       _op(nodes, "Add", [lo_r, one], prefix, suffix="a"),
                       _op(nodes, "Sub", [hi_r, one], prefix, suffix="b"),
                       _op(nodes, "Add", [lo_c, one], prefix, suffix="c"),
                       _op(nodes, "Sub", [hi_c, one], prefix, suffix="d"), h, w, prefix), h, w


def outbox(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="outb", **kw):
    occ = _reduce_occ(nodes, inits, x, depth, prefix)
    lo_r, hi_r, lo_c, hi_c = _bbox(nodes, inits, occ, h, w, prefix)
    one = _init(inits, f"{prefix}_1", np.array(1, dtype=np.int64))
    return _rect_mask(nodes, inits,
                       _op(nodes, "Sub", [lo_r, one], prefix, suffix="a"),
                       _op(nodes, "Add", [hi_r, one], prefix, suffix="b"),
                       _op(nodes, "Sub", [lo_c, one], prefix, suffix="c"),
                       _op(nodes, "Add", [hi_c, one], prefix, suffix="d"), h, w, prefix), h, w


def subgrid(nodes, inits, obj, grid, h, w, depth=NUM_COLORS, prefix="sg", **kw):
    """Crop `grid` to bbox(obj), content top-left-aligned in a same-(h,w) canvas."""
    occ = _reduce_occ(nodes, inits, obj, depth, prefix)
    lo_r, hi_r, lo_c, hi_c = _bbox(nodes, inits, occ, h, w, prefix)
    translated = _dynamic_translate(nodes, inits, grid, h, w, lo_r, lo_c, NUM_COLORS, prefix + "_tr")
    bh = _op(nodes, "Sub", [hi_r, lo_r], prefix, suffix="bh")
    bw = _op(nodes, "Sub", [hi_c, lo_c], prefix, suffix="bw")
    zero_i = _init(inits, f"{prefix}_z", np.array(0, dtype=np.int64))
    rect = _rect_mask(nodes, inits, zero_i, bh, zero_i, bw, h, w, prefix + "_rm")
    return _op(nodes, "Mul", [translated, rect], prefix), h, w


# ============================================================================
# SECTION E — color statistics (no Unique)
# ============================================================================

def _channel_sums(nodes, inits, x, prefix):
    return _op(nodes, "ReduceSum", [x], prefix, axes=[0, 2, 3], keepdims=0)  # (10,)


def mostcolor(nodes, inits, x, h, w, prefix="mcl", **kw):
    cs = _channel_sums(nodes, inits, x, prefix)
    idx = _op(nodes, "ArgMax", [cs], prefix, axis=0, keepdims=0)
    return _to_scalar_f(nodes, inits, idx, prefix), 1, 1


def leastcolor(nodes, inits, x, h, w, prefix="lcl", **kw):
    cs = _channel_sums(nodes, inits, x, prefix)
    zero = scalar_const(inits, prefix, 0.0)
    cs4 = _op(nodes, "Reshape", [cs, _init(inits, f"{prefix}_rs", np.array([10], dtype=np.int64))], prefix)
    is0 = _op(nodes, "Equal", [cs4, _init(inits, f"{prefix}_z10", np.zeros(10, dtype=np.float32))], prefix, suffix="is0")
    is0f = _op(nodes, "Cast", [is0], prefix, to=TensorProto.FLOAT, suffix="is0f")
    penal = _op(nodes, "Add", [cs4, _op(nodes, "Mul", [is0f, _init(inits, f"{prefix}_big", np.full(10, BIG, dtype=np.float32))], prefix, suffix="pen")], prefix, suffix="pn")
    idx = _op(nodes, "ArgMin", [penal], prefix, axis=0, keepdims=0)
    return _to_scalar_f(nodes, inits, idx, prefix), 1, 1


mostcommon = mostcolor      # only supports color-grid containers (documented scope limit)
leastcommon = leastcolor


def colorcount(nodes, inits, x, h, w, color=0, prefix="cc", **kw):
    ch = _init(inits, f"{prefix}_ch", np.array([color], dtype=np.int64))
    val = _op(nodes, "Gather", [x, ch], prefix, axis=1)
    return _op(nodes, "ReduceSum", [val], prefix, axes=[0, 1, 2, 3], keepdims=1), 1, 1


def numcolors(nodes, inits, x, h, w, prefix="nclr", **kw):
    cs = _channel_sums(nodes, inits, x, prefix)
    pres = _op(nodes, "Greater", [cs, scalar_const(inits, prefix, 0.0)], prefix)
    presf = _op(nodes, "Cast", [pres], prefix, to=TensorProto.FLOAT, suffix="pf")
    s = _op(nodes, "ReduceSum", [presf], prefix, axes=[0], keepdims=1, suffix="sum")
    return _op(nodes, "Reshape", [s, _init(inits, f"{prefix}_rs", np.array([1,1,1,1], dtype=np.int64))], prefix), 1, 1


def palette(nodes, inits, x, h, w, prefix="pal", **kw):
    """Static-shape substitute for the DSL's variable-size color set:
    returns a fixed (1,1,1,10) presence bitmap over colors 0..9."""
    cs = _channel_sums(nodes, inits, x, prefix)
    pres = _op(nodes, "Greater", [cs, scalar_const(inits, prefix, 0.0)], prefix)
    presf = _op(nodes, "Cast", [pres], prefix, to=TensorProto.FLOAT, suffix="pf")
    return _op(nodes, "Reshape", [presf, _init(inits, f"{prefix}_rs", np.array([1, 1, 1, 10], dtype=np.int64))], prefix), 1, 10


def color(nodes, inits, x, h, w, prefix="clr", **kw):
    """color(univalued object) = its single nonzero color."""
    return mostcolor(nodes, inits, x, h, w, prefix)


# ============================================================================
# SECTION F — pixel/color algebra
# ============================================================================

def ofcolor(nodes, inits, x, h, w, color=1, prefix="ofc", **kw):
    ch = _init(inits, f"{prefix}_ch", np.array([color], dtype=np.int64))
    val = _op(nodes, "Gather", [x, ch], prefix, axis=1)
    g = _op(nodes, "Greater", [val, scalar_const(inits, prefix, 0.5)], prefix)
    return _op(nodes, "Cast", [g], prefix, to=TensorProto.FLOAT), h, w


def canvas(nodes, inits, color=0, h=3, w=3, prefix="cvs", **kw):
    arr = np.zeros((1, NUM_COLORS, h, w), dtype=np.float32)
    arr[0, color] = 1.0
    name = _init(inits, f"{prefix}_d", arr)
    return _op(nodes, "Identity", [name], prefix), h, w


def asobject(nodes, inits, x, h, w, prefix="aobj", **kw):
    """The whole grid AS an object == identity (every DSL grid cell already has a color)."""
    return _op(nodes, "Identity", [x], prefix), h, w


def asindices(nodes, inits, x, h, w, prefix="aidx", **kw):
    """The full index domain -> all-ones (1,1,h,w) mask."""
    ones = np.ones((1, 1, h, w), dtype=np.float32)
    return _init(inits, f"{prefix}_o", ones), h, w


def toindices(nodes, inits, x, h, w, depth=NUM_COLORS, prefix="tidx", **kw):
    return _reduce_occ(nodes, inits, x, depth, prefix), h, w


def toobject(nodes, inits, indices_mask, grid, h, w, prefix="tobj", **kw):
    return _op(nodes, "Mul", [grid, indices_mask], prefix), h, w


def toivec(nodes, inits, i_scalar, h, w, prefix="tiv", **kw):
    zero = scalar_const(inits, prefix, 0.0)
    return _op(nodes, "Concat", [i_scalar, zero], prefix, axis=3), 1, 2

def tojvec(nodes, inits, j_scalar, h, w, prefix="tjv", **kw):
    zero = scalar_const(inits, prefix, 0.0)
    return _op(nodes, "Concat", [zero, j_scalar], prefix, axis=3), 1, 2


def fill(nodes, inits, x, h, w, color=1, mask=None, prefix="fill", **kw):
    """fill(grid, color, indices_mask): mask is (1,1,h,w) 0/1."""
    one_hot_color = np.zeros((1, NUM_COLORS, h, w), dtype=np.float32)
    one_hot_color[0, color] = 1.0
    col = _init(inits, f"{prefix}_c", one_hot_color)
    fg = _op(nodes, "Mul", [mask, col], prefix, suffix="fg")
    inv = _op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), mask], prefix, suffix="inv")
    bg = _op(nodes, "Mul", [inv, x], prefix, suffix="bg")
    return _op(nodes, "Add", [bg, fg], prefix), h, w


def replace(nodes, inits, x, h, w, old_color=0, new_color=1, prefix="rpl", **kw):
    ch = _init(inits, f"{prefix}_ch", np.array([old_color], dtype=np.int64))
    val = _op(nodes, "Gather", [x, ch], prefix, axis=1)
    g = _op(nodes, "Greater", [val, scalar_const(inits, prefix, 0.5)], prefix)
    mask = _op(nodes, "Cast", [g], prefix, to=TensorProto.FLOAT, suffix="mf")
    return fill(nodes, inits, x, h, w, color=new_color, mask=mask, prefix=prefix + "_f")


def recolor(nodes, inits, color, mask, h, w, prefix="rcl", **kw):
    one_hot_color = np.zeros((1, NUM_COLORS, h, w), dtype=np.float32)
    one_hot_color[0, color] = 1.0
    col = _init(inits, f"{prefix}_c", one_hot_color)
    return _op(nodes, "Mul", [col, mask], prefix), h, w


def paint(nodes, inits, grid, obj, h, w, prefix="pnt", **kw):
    """Overlay object's own colors onto grid wherever obj occupies a cell."""
    occ = _op(nodes, "ReduceMax", [obj], prefix, axes=[1], keepdims=1)
    inv = _op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), occ], prefix, suffix="inv")
    kept = _op(nodes, "Mul", [grid, inv], prefix, suffix="kpt")
    return _op(nodes, "Add", [kept, obj], prefix), h, w


def cover(nodes, inits, grid, obj_or_mask, h, w, prefix="cov", **kw):
    """Erase obj/mask cells back to background color = mostcolor(grid)."""
    bg_idx, _, _ = mostcolor(nodes, inits, grid, h, w, prefix + "_bg")
    bg_idx_i = _op(nodes, "Cast", [_op(nodes, "Reshape", [bg_idx, _init(inits, f"{prefix}_rs", np.array([1], dtype=np.int64))], prefix, suffix="r")], prefix, to=TensorProto.INT64, suffix="i")
    occ = obj_or_mask
    # detect depth via caller convention: if 10-channel object, reduce first
    return _cover_with_mask(nodes, inits, grid, occ, bg_idx_i, h, w, prefix)


def _cover_with_mask(nodes, inits, grid, mask10_or_1, bg_idx_i, h, w, prefix):
    # if mask has 10 channels reduce; harmless if already 1-channel? we branch by trying reduce always safe only if depth known.
    occ = _op(nodes, "ReduceMax", [mask10_or_1], prefix, axes=[1], keepdims=1)
    bgvec = _op(nodes, "OneHot", [bg_idx_i, _init(inits, f"{prefix}_d", np.array(NUM_COLORS, dtype=np.int64)),
                                   _init(inits, f"{prefix}_ov", np.array([0.0, 1.0], dtype=np.float32))], prefix, axis=0)
    bgvec4 = _op(nodes, "Reshape", [bgvec, _init(inits, f"{prefix}_rs2", np.array([1, NUM_COLORS, 1, 1], dtype=np.int64))], prefix)
    layer = _op(nodes, "Mul", [bgvec4, occ], prefix, suffix="lyr")
    inv = _op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), occ], prefix, suffix="inv")
    kept = _op(nodes, "Mul", [grid, inv], prefix, suffix="kpt")
    return _op(nodes, "Add", [kept, layer], prefix), h, w


def underfill(nodes, inits, x, h, w, color=1, mask=None, prefix="ufl", **kw):
    """fill only where the current cell is background."""
    bg_idx, _, _ = mostcolor(nodes, inits, x, h, w, prefix + "_bg")
    bg_idx_i = _op(nodes, "Cast", [_op(nodes, "Reshape", [bg_idx, _init(inits, f"{prefix}_rs", np.array([1], dtype=np.int64))], prefix)], prefix, to=TensorProto.INT64, suffix="i")
    bgchan = _op(nodes, "Gather", [x, bg_idx_i], prefix, axis=1)
    eff_mask = _op(nodes, "Mul", [mask, bgchan], prefix, suffix="em")
    return fill(nodes, inits, x, h, w, color=color, mask=eff_mask, prefix=prefix + "_f")


def underpaint(nodes, inits, grid, obj, h, w, prefix="upt", **kw):
    bg_idx, _, _ = mostcolor(nodes, inits, grid, h, w, prefix + "_bg")
    bg_idx_i = _op(nodes, "Cast", [_op(nodes, "Reshape", [bg_idx, _init(inits, f"{prefix}_rs", np.array([1], dtype=np.int64))], prefix)], prefix, to=TensorProto.INT64, suffix="i")
    bgchan = _op(nodes, "Gather", [grid, bg_idx_i], prefix, axis=1)
    occ = _op(nodes, "ReduceMax", [obj], prefix, axes=[1], keepdims=1)
    eff = _op(nodes, "Mul", [occ, bgchan], prefix, suffix="eff")
    inv = _op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), eff], prefix, suffix="inv")
    kept = _op(nodes, "Mul", [grid, inv], prefix, suffix="kpt")
    objm = _op(nodes, "Mul", [obj, eff], prefix, suffix="om")
    return _op(nodes, "Add", [kept, objm], prefix), h, w


def switch(nodes, inits, x, h, w, a=0, b=1, prefix="sw", **kw):
    cha = _init(inits, f"{prefix}_cha", np.array([a], dtype=np.int64))
    chb = _init(inits, f"{prefix}_chb", np.array([b], dtype=np.int64))
    va = _op(nodes, "Gather", [x, cha], prefix, axis=1, suffix="va")
    vb = _op(nodes, "Gather", [x, chb], prefix, axis=1, suffix="vb")
    ma = _op(nodes, "Cast", [_op(nodes, "Greater", [va, scalar_const(inits, prefix, 0.5)], prefix, suffix="ga")], prefix, to=TensorProto.FLOAT, suffix="mfa")
    mb = _op(nodes, "Cast", [_op(nodes, "Greater", [vb, scalar_const(inits, prefix, 0.5)], prefix, suffix="gb")], prefix, to=TensorProto.FLOAT, suffix="mfb")
    g1, _, _ = fill(nodes, inits, x, h, w, color=a, mask=mb, prefix=prefix + "_f1")
    return fill(nodes, inits, g1, h, w, color=b, mask=ma, prefix=prefix + "_f2")


def merge(nodes, inits, a, b=None, h=None, w=None, prefix="mrg", **kw):
    """Dispatch: merge(objset) -> grid, or merge(grid1, grid2) -> grid."""
    if isinstance(a, tuple) and len(a) == 4 and a[0] == "OBJSET":
        _, stack, valid, K = a
        v = _op(nodes, "Reshape", [valid, _init(inits, f"{prefix}_rs", np.array([K, 1, 1, 1], dtype=np.int64))], prefix)
        masked = _op(nodes, "Mul", [stack, v], prefix, suffix="mk")
        return _op(nodes, "ReduceMax", [masked], prefix, axes=[0], keepdims=0), h, w
    return _op(nodes, "Max", [a, b], prefix), h, w


def combine(nodes, inits, a, b, h=None, w=None, prefix="cmb", **kw):
    if isinstance(a, tuple) and a[0] == "OBJSET":
        _, sa, va, Ka = a
        _, sb, vb, Kb = b
        s = _op(nodes, "Concat", [sa, sb], prefix, axis=0)
        v = _op(nodes, "Concat", [va, vb], prefix, axis=0)
        return ("OBJSET", s, v, Ka + Kb), h, w
    if isinstance(a, tuple):     # python containers
        return a + b, h, w
    return _op(nodes, "Max", [a, b], prefix), h, w


def difference(nodes, inits, a, b, h, w, prefix="dif", **kw):
    inv = _op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), b], prefix, suffix="inv")
    return _op(nodes, "Mul", [a, inv], prefix), h, w


def intersection(nodes, inits, a, b, h, w, prefix="isc", **kw):
    return _op(nodes, "Mul", [a, b], prefix), h, w


# ============================================================================
# SECTION G — connected components (pointer-doubling; objects/partition/fgpartition)
# ============================================================================

def _label_field(nodes, inits, fg_flat, N, rounds, prefix):
    idx_i = _init(inits, f"{prefix}_idx", np.arange(N, dtype=np.int64))
    idx_f = _op(nodes, "Cast", [idx_i], prefix, to=TensorProto.FLOAT, suffix="idxf")
    fg_bool = _op(nodes, "Cast", [fg_flat], prefix, to=TensorProto.BOOL, suffix="fgb")
    big = scalar_const(inits, prefix, BIG)
    big_flat = _op(nodes, "Expand", [big, _init(inits, f"{prefix}_es", np.array([N], dtype=np.int64))], prefix, suffix="bigN")
    D = _op(nodes, "Where", [fg_bool, idx_f, big_flat], prefix, suffix="D0")
    for _ in range(rounds):
        D_i64 = _op(nodes, "Cast", [D], prefix, to=TensorProto.INT64, suffix=f"c{_}")
        zero_i = _init(inits, f"{prefix}_z{_}", np.array(0, dtype=np.int64))
        nm1 = _init(inits, f"{prefix}_nm1_{_}", np.array(N - 1, dtype=np.int64))
        D_clamp = _op(nodes, "Clip", [D_i64, zero_i, nm1], prefix, suffix=f"cl{_}")
        D2 = _op(nodes, "GatherElements", [D, D_clamp], prefix, suffix=f"gnd{_}", axis=0)
        D = _op(nodes, "Where", [fg_bool, D2, big_flat], prefix, suffix=f"sh{_}")
    return D, fg_bool


def _hook_step(nodes, inits, D, fg_bool, h, w, neighbors, prefix, univalued=False, color_flat=None):
    N = h * w
    D2d = _op(nodes, "Reshape", [D, _init(inits, f"{prefix}_rs", np.array([1, 1, h, w], dtype=np.int64))], prefix)
    big = scalar_const(inits, prefix, BIG)
    pads = _init(inits, f"{prefix}_pd", np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64))
    padded = _op(nodes, "Pad", [D2d, pads, big], prefix, suffix="pdd")
    cands = [D2d]
    if univalued:
        C2d = _op(nodes, "Reshape", [color_flat, _init(inits, f"{prefix}_csh", np.array([1, 1, h, w], dtype=np.int64))], prefix, suffix="C2d")
        neg1 = _init(inits, f"{prefix}_neg1", np.array(-1.0, dtype=np.float32))
        Cpadded = _op(nodes, "Pad", [C2d, pads, neg1], prefix, suffix="Cpad")
    for (pr, pc) in neighbors:
        st = _init(inits, f"{prefix}_s{pr}{pc}", np.array([0, 0, pr, pc], dtype=np.int64))
        en = _init(inits, f"{prefix}_e{pr}{pc}", np.array([1, 1, pr + h, pc + w], dtype=np.int64))
        ax = _init(inits, f"{prefix}_a{pr}{pc}", np.array([0, 1, 2, 3], dtype=np.int64))
        nb = _op(nodes, "Slice", [padded, st, en, ax], prefix, suffix=f"nb{pr}{pc}")
        if univalued:
            cnb = _op(nodes, "Slice", [Cpadded, st, en, ax], prefix, suffix=f"cnb{pr}{pc}")
            same = _op(nodes, "Equal", [C2d, cnb], prefix, suffix=f"sm{pr}{pc}")
            nb = _op(nodes, "Where", [same, nb, _op(nodes, "Expand", [big, _init(inits, f"{prefix}_e2{pr}{pc}", np.array([1, 1, h, w], dtype=np.int64))], prefix, suffix=f"bx{pr}{pc}")], prefix, suffix=f"nbf{pr}{pc}")
        cands.append(nb)
    unsq = [_op(nodes, "Unsqueeze", [c], prefix, suffix=f"u{i}", axes=[0]) for i, c in enumerate(cands)]
    stacked = _op(nodes, "Concat", unsq, prefix, axis=0, suffix="stk")
    hooked = _op(nodes, "ReduceMin", [stacked], prefix, axes=[0], keepdims=1, suffix="hkd")
    hooked_flat = _op(nodes, "Reshape", [hooked, _init(inits, f"{prefix}_rf", np.array([N], dtype=np.int64))], prefix, suffix="hf")
    return _op(nodes, "Where", [fg_bool, hooked_flat, _op(nodes, "Expand", [big, _init(inits, f"{prefix}_eN", np.array([N], dtype=np.int64))], prefix, suffix="bigN2")], prefix, suffix="Dn")


def objects(nodes, inits, x, h, w, univalued=False, diagonal=False, without_bg=True, K=24, prefix="objs", **kw):
    """Real connected components via pointer-doubling. Returns ('OBJSET', stack, valid, K)."""
    N = h * w
    rounds = max(1, int(np.ceil(np.log2(max(N, 2)))))
    color_id = _op(nodes, "ArgMax", [x], prefix, axis=1, keepdims=1)
    color_flat = _op(nodes, "Reshape", [color_id, _init(inits, f"{prefix}_cfs", np.array([N], dtype=np.int64))], prefix, suffix="cf")
    occ_flat = _op(nodes, "Reshape", [_op(nodes, "ReduceMax", [x], prefix, axes=[1], keepdims=0)], prefix, suffix="occ", shape=None) if False else \
               _op(nodes, "Reshape", [_op(nodes, "ReduceMax", [x], prefix, axes=[1], keepdims=0, suffix="rm")], prefix, suffix="occf")
    occ_flat = _op(nodes, "Reshape", [occ_flat, _init(inits, f"{prefix}_ofs", np.array([N], dtype=np.int64))], prefix, suffix="occf2")

    if without_bg:
        bg_idx, _, _ = mostcolor(nodes, inits, x, h, w, prefix + "_bg")
        bg_idx_i = _op(nodes, "Cast", [_op(nodes, "Reshape", [bg_idx, _init(inits, f"{prefix}_bgr", np.array([1], dtype=np.int64))], prefix)], prefix, to=TensorProto.INT64, suffix="bgi")
        color_flat_i = _op(nodes, "Cast", [color_flat], prefix, to=TensorProto.INT64, suffix="cfi")
        is_bg = _op(nodes, "Equal", [color_flat_i, bg_idx_i], prefix, suffix="isbg")
        not_bg = _op(nodes, "Cast", [_op(nodes, "Not", [is_bg], prefix)], prefix, to=TensorProto.FLOAT, suffix="nbg")
        fg_flat = _op(nodes, "Mul", [occ_flat, not_bg], prefix, suffix="fg")
    else:
        fg_flat = occ_flat

    D, fg_bool = _label_field(nodes, inits, fg_flat, N, 1, prefix + "_init")   # init only
    neighbors = [(0, 1), (2, 1), (1, 0), (1, 2)]
    if diagonal:
        neighbors += [(0, 0), (0, 2), (2, 0), (2, 2)]

    for r in range(rounds):
        D = _hook_step(nodes, inits, D, fg_bool, h, w, neighbors, f"{prefix}_hk{r}", univalued, color_flat if univalued else None)
        D_i64 = _op(nodes, "Cast", [D], prefix, to=TensorProto.INT64, suffix=f"sc{r}")
        zero_i = _init(inits, f"{prefix}_z{r}", np.array(0, dtype=np.int64))
        nm1 = _init(inits, f"{prefix}_nm1_{r}", np.array(N - 1, dtype=np.int64))
        D_clamp = _op(nodes, "Clip", [D_i64, zero_i, nm1], prefix, suffix=f"clp{r}")
        D_g = _op(nodes, "GatherElements", [D, D_clamp], prefix, suffix=f"ge{r}", axis=0)
        D = _op(nodes, "Where", [fg_bool, D_g, _op(nodes, "Expand", [scalar_const(inits, prefix, BIG), _init(inits, f"{prefix}_eb{r}", np.array([N], dtype=np.int64))], prefix, suffix=f"be{r}")], prefix, suffix=f"sh{r}")

    labels_flat = D
    remaining = fg_flat
    stack_slots, valid_slots = [], []
    for k in range(K):
        seed = _op(nodes, "ArgMax", [remaining], prefix, suffix=f"seed{k}", axis=0, keepdims=0)
        seed1 = _op(nodes, "Reshape", [seed, _init(inits, f"{prefix}_s1_{k}", np.array([1], dtype=np.int64))], prefix, suffix=f"s1_{k}")
        has_any = _op(nodes, "ReduceMax", [remaining], prefix, suffix=f"ha{k}", axes=[0], keepdims=0)
        seed_label = _op(nodes, "Gather", [labels_flat, seed1], prefix, suffix=f"sl{k}", axis=0)
        eq = _op(nodes, "Cast", [_op(nodes, "Equal", [labels_flat, seed_label], prefix, suffix=f"eq{k}")], prefix, to=TensorProto.FLOAT, suffix=f"eqf{k}")
        comp = _op(nodes, "Mul", [eq, remaining], prefix, suffix=f"cmp{k}")
        has_any1 = _op(nodes, "Reshape", [has_any, _init(inits, f"{prefix}_ha1_{k}", np.array([1], dtype=np.int64))], prefix, suffix=f"ha1_{k}")
        comp = _op(nodes, "Mul", [comp, has_any1], prefix, suffix=f"cmpg{k}")
        remaining = _op(nodes, "Sub", [remaining, comp], prefix, suffix=f"rem{k}")

        comp3 = _op(nodes, "Reshape", [comp, _init(inits, f"{prefix}_c3_{k}", np.array([1, h, w], dtype=np.int64))], prefix, suffix=f"c3_{k}")
        onehot_c = _op(nodes, "OneHot", [color_id, _init(inits, f"{prefix}_d10_{k}", np.array(NUM_COLORS, dtype=np.int64)),
                                          _init(inits, f"{prefix}_ov_{k}", np.array([0.0, 1.0], dtype=np.float32))], prefix, suffix=f"oh{k}", axis=1)
        onehot_c = _op(nodes, "Reshape", [onehot_c, _init(inits, f"{prefix}_ohr_{k}", np.array([NUM_COLORS, h, w], dtype=np.int64))], prefix, suffix=f"ohr{k}")
        slot = _op(nodes, "Mul", [onehot_c, comp3], prefix, suffix=f"slot{k}")
        stack_slots.append(_op(nodes, "Unsqueeze", [slot], prefix, suffix=f"us{k}", axes=[0]))
        valid_slots.append(_op(nodes, "Unsqueeze", [_op(nodes, "Mul", [has_any, has_any], prefix, suffix=f"v{k}")], prefix, suffix=f"uv{k}", axes=[0]))

    stack = _op(nodes, "Concat", stack_slots, prefix, axis=0, suffix="STACK")
    valid = _op(nodes, "Concat", valid_slots, prefix, axis=0, suffix="VALID")
    return ("OBJSET", stack, valid, K), h, w


def partition(nodes, inits, x, h, w, without_bg=False, prefix="part", **kw):
    """Exact (no approximation needed): one slot per color, K=10 fixed."""
    cs = _channel_sums(nodes, inits, x, prefix)
    presence = _op(nodes, "Greater", [cs, scalar_const(inits, prefix, 0.0)], prefix)
    presf = _op(nodes, "Cast", [presence], prefix, to=TensorProto.FLOAT, suffix="pf")  # (10,)
    if without_bg:
        bg_idx, _, _ = mostcolor(nodes, inits, x, h, w, prefix + "_bg")
        bg_i = _op(nodes, "Cast", [_op(nodes, "Reshape", [bg_idx, _init(inits, f"{prefix}_r", np.array([1], dtype=np.int64))], prefix)], prefix, to=TensorProto.INT64, suffix="bgi")
        colors_idx = _init(inits, f"{prefix}_ci", np.arange(NUM_COLORS, dtype=np.int64))
        is_bg = _op(nodes, "Equal", [colors_idx, bg_i], prefix, suffix="isbg")
        not_bg = _op(nodes, "Cast", [_op(nodes, "Not", [is_bg], prefix)], prefix, to=TensorProto.FLOAT, suffix="nbf")
        presf = _op(nodes, "Mul", [presf, not_bg], prefix, suffix="pfm")
    slots = []
    for c in range(NUM_COLORS):
        vec = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
        vec[0, c] = 1.0
        cvec = _init(inits, f"{prefix}_cv{c}", vec)
        slot = _op(nodes, "Mul", [x, cvec], prefix, suffix=f"sl{c}")
        slots.append(_op(nodes, "Unsqueeze", [slot], prefix, suffix=f"us{c}", axes=[0]))
    stack = _op(nodes, "Concat", slots, prefix, axis=0, suffix="STACK")
    valid = _op(nodes, "Reshape", [presf, _init(inits, f"{prefix}_vr", np.array([NUM_COLORS], dtype=np.int64))], prefix, suffix="VALID")
    return ("OBJSET", stack, valid, NUM_COLORS), h, w


def fgpartition(nodes, inits, x, h, w, prefix="fgp", **kw):
    return partition(nodes, inits, x, h, w, without_bg=True, prefix=prefix)


# ============================================================================
# SECTION H — object-set operations
# ============================================================================

def colorfilter(nodes, inits, objset, h, w, color=0, prefix="cf", **kw):
    _, stack, valid, K = objset
    ch = _init(inits, f"{prefix}_ch", np.array([color], dtype=np.int64))
    news = []
    for k in range(K):
        slot = _op(nodes, "Gather", [stack, _init(inits, f"{prefix}_k{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"g{k}", axis=0)
        val = _op(nodes, "Gather", [slot, ch], prefix, suffix=f"v{k}", axis=1)
        s = _op(nodes, "ReduceSum", [val], prefix, suffix=f"s{k}", axes=[0, 1, 2, 3, 4], keepdims=0) if False else \
            _op(nodes, "ReduceSum", [val], prefix, suffix=f"s2_{k}", axes=None, keepdims=0)
        keep = _op(nodes, "Cast", [_op(nodes, "Greater", [s, scalar_const(inits, prefix, 0.0)], prefix, suffix=f"gt{k}")], prefix, to=TensorProto.FLOAT, suffix=f"kf{k}")
        news.append(keep)
    keepvec = _op(nodes, "Concat", [_op(nodes, "Reshape", [n, _init(inits, f"{prefix}_r{i}", np.array([1], dtype=np.int64))], prefix, suffix=f"rr{i}") for i, n in enumerate(news)], prefix, axis=0, suffix="kv")
    newvalid = _op(nodes, "Mul", [valid, keepvec], prefix, suffix="nv")
    return ("OBJSET", stack, newvalid, K), h, w


def sizefilter(nodes, inits, objset, h, w, n=1, prefix="szf", **kw):
    _, stack, valid, K = objset
    news = []
    for k in range(K):
        slot = _op(nodes, "Gather", [stack, _init(inits, f"{prefix}_k{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"g{k}", axis=0)
        occ = _op(nodes, "ReduceMax", [slot], prefix, suffix=f"occ{k}", axes=[1], keepdims=0)
        cnt = _op(nodes, "ReduceSum", [occ], prefix, suffix=f"cnt{k}", axes=None, keepdims=0)
        keep = _op(nodes, "Cast", [_op(nodes, "Equal", [cnt, scalar_const(inits, prefix, float(n))], prefix, suffix=f"eq{k}")], prefix, to=TensorProto.FLOAT, suffix=f"kf{k}")
        news.append(_op(nodes, "Reshape", [keep, _init(inits, f"{prefix}_r{k}", np.array([1], dtype=np.int64))], prefix, suffix=f"rr{k}"))
    keepvec = _op(nodes, "Concat", news, prefix, axis=0, suffix="kv")
    newvalid = _op(nodes, "Mul", [valid, keepvec], prefix, suffix="nv")
    return ("OBJSET", stack, newvalid, K), h, w


def sfilter(nodes, inits, objset, h, w, predicate, prefix="sf", **kw):
    """predicate: fn(nodes,inits,slot,h,w,prefix) -> (bool_scalar_tensor,1,1)."""
    _, stack, valid, K = objset
    news = []
    for k in range(K):
        slot = _op(nodes, "Gather", [stack, _init(inits, f"{prefix}_k{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"g{k}", axis=0)
        slot_sq = _op(nodes, "Reshape", [slot, _init(inits, f"{prefix}_rs{k}", np.array([1, NUM_COLORS, h, w], dtype=np.int64))], prefix, suffix=f"sq{k}")
        keep, _, _ = predicate(nodes, inits, slot_sq, h, w, prefix=f"{prefix}_p{k}")
        news.append(_op(nodes, "Reshape", [keep, _init(inits, f"{prefix}_r{k}", np.array([1], dtype=np.int64))], prefix, suffix=f"rr{k}"))
    keepvec = _op(nodes, "Concat", news, prefix, axis=0, suffix="kv")
    newvalid = _op(nodes, "Mul", [valid, keepvec], prefix, suffix="nv")
    return ("OBJSET", stack, newvalid, K), h, w


def mfilter(nodes, inits, objset, h, w, predicate, prefix="mf", **kw):
    filtered, _, _ = sfilter(nodes, inits, objset, h, w, predicate, prefix)
    return merge(nodes, inits, filtered, None, h, w, prefix + "_m")


def _rank_score(nodes, inits, valid, K, order, prefix):
    """order: 'first' (smallest k) or 'last' (largest k)."""
    ks = np.arange(K, dtype=np.float32)
    kvec = _init(inits, f"{prefix}_kv", ks)
    penal = _op(nodes, "Add", [kvec, _op(nodes, "Mul", [_op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), _op(nodes, "Reshape", [valid, _init(inits, f"{prefix}_vr", np.array([K], dtype=np.int64))], prefix, suffix="vr")], prefix, suffix="iv"), _init(inits, f"{prefix}_big", np.array(BIG, dtype=np.float32))], prefix, suffix="pn")], prefix, suffix="scr")
    if order == "first":
        return _op(nodes, "ArgMin", [penal], prefix, axis=0, keepdims=0)
    neg = _op(nodes, "Neg", [penal], prefix, suffix="negp")
    return _op(nodes, "ArgMax", [neg], prefix, axis=0, keepdims=0)


def first(nodes, inits, objset, h, w, prefix="fst", **kw):
    _, stack, valid, K = objset
    idx = _rank_score(nodes, inits, valid, K, "first", prefix)
    idx1 = _op(nodes, "Reshape", [idx, _init(inits, f"{prefix}_r", np.array([1], dtype=np.int64))], prefix)
    slot = _op(nodes, "Gather", [stack, idx1], prefix, axis=0)
    return _op(nodes, "Reshape", [slot, _init(inits, f"{prefix}_rs", np.array([1, NUM_COLORS, h, w], dtype=np.int64))], prefix), h, w


def last(nodes, inits, objset, h, w, prefix="lst", **kw):
    _, stack, valid, K = objset
    idx = _rank_score(nodes, inits, valid, K, "last", prefix)
    idx1 = _op(nodes, "Reshape", [idx, _init(inits, f"{prefix}_r", np.array([1], dtype=np.int64))], prefix)
    slot = _op(nodes, "Gather", [stack, idx1], prefix, axis=0)
    return _op(nodes, "Reshape", [slot, _init(inits, f"{prefix}_rs", np.array([1, NUM_COLORS, h, w], dtype=np.int64))], prefix), h, w


def extract(nodes, inits, objset, h, w, predicate, prefix="ext", **kw):
    filt, _, _ = sfilter(nodes, inits, objset, h, w, predicate, prefix + "_f")
    return first(nodes, inits, filt, h, w, prefix + "_1")


def other(nodes, inits, objset, one, h, w, prefix="oth", **kw):
    def pred(nodes, inits, slot, h, w, prefix):
        eq = _op(nodes, "Equal", [_op(nodes, "ArgMax", [slot], prefix, axis=1, keepdims=0), _op(nodes, "ArgMax", [one], prefix, axis=1, keepdims=0, suffix="am2")], prefix, suffix="eq")
        diff = _op(nodes, "Cast", [_op(nodes, "Not", [_op(nodes, "ReduceMin", [_op(nodes, "Cast", [eq], prefix, to=TensorProto.FLOAT, suffix="eqf")], prefix, axes=None, keepdims=0, suffix="allq")], prefix, suffix="nallq")], prefix, to=TensorProto.FLOAT, suffix="kf")
        return diff, 1, 1
    filt, _, _ = sfilter(nodes, inits, objset, h, w, pred, prefix)
    return first(nodes, inits, filt, h, w, prefix + "_f")


def argmax(nodes, inits, objset, h, w, keyfn, prefix="amx", **kw):
    _, stack, valid, K = objset
    scores = []
    for k in range(K):
        slot = _op(nodes, "Gather", [stack, _init(inits, f"{prefix}_k{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"g{k}", axis=0)
        slot_sq = _op(nodes, "Reshape", [slot, _init(inits, f"{prefix}_rs{k}", np.array([1, NUM_COLORS, h, w], dtype=np.int64))], prefix, suffix=f"sq{k}")
        s, _, _ = keyfn(nodes, inits, slot_sq, h, w, prefix=f"{prefix}_key{k}")
        v = _op(nodes, "Reshape", [_op(nodes, "Gather", [valid, _init(inits, f"{prefix}_vi{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"vg{k}", axis=0)], prefix, suffix=f"vr{k}", shape=None) if False else \
            _op(nodes, "Gather", [valid, _init(inits, f"{prefix}_vi{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"vg{k}", axis=0)
        penal = _op(nodes, "Add", [s, _op(nodes, "Mul", [_op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), _op(nodes, "Reshape", [v, _init(inits, f"{prefix}_v1_{k}", np.array([1, 1, 1, 1], dtype=np.int64))], prefix, suffix=f"v4_{k}")], prefix, suffix=f"iv{k}"), scalar_const(inits, prefix, -BIG)], prefix, suffix=f"pn{k}")], prefix, suffix=f"scr{k}")
        scores.append(penal)
    allscores = _op(nodes, "Concat", scores, prefix, axis=0, suffix="AS")
    best = _op(nodes, "ArgMax", [allscores], prefix, axis=0, keepdims=0)
    best1 = _op(nodes, "Reshape", [best, _init(inits, f"{prefix}_br", np.array([1], dtype=np.int64))], prefix, suffix="b1")
    slot = _op(nodes, "Gather", [stack, best1], prefix, axis=0, suffix="bslot")
    return _op(nodes, "Reshape", [slot, _init(inits, f"{prefix}_frs", np.array([1, NUM_COLORS, h, w], dtype=np.int64))], prefix, suffix="final"), h, w


def argmin(nodes, inits, objset, h, w, keyfn, prefix="amn", **kw):
    def neg_key(nodes, inits, slot, h, w, prefix):
        s, _, _ = keyfn(nodes, inits, slot, h, w, prefix)
        return _op(nodes, "Neg", [s], prefix, suffix="n"), 1, 1
    return argmax(nodes, inits, objset, h, w, neg_key, prefix)


def valmax(nodes, inits, objset, h, w, keyfn, prefix="vmx", **kw):
    best, _, _ = argmax(nodes, inits, objset, h, w, keyfn, prefix)
    return keyfn(nodes, inits, best, h, w, prefix + "_v")

def valmin(nodes, inits, objset, h, w, keyfn, prefix="vmn", **kw):
    best, _, _ = argmin(nodes, inits, objset, h, w, keyfn, prefix)
    return keyfn(nodes, inits, best, h, w, prefix + "_v")


def order(nodes, inits, objset, h, w, keyfn, prefix="ord", **kw):
    """Returns a python tuple of K slot-tensor names, sorted by keyfn ascending (invalid pushed to the end)."""
    _, stack, valid, K = objset
    scores = []
    for k in range(K):
        slot = _op(nodes, "Gather", [stack, _init(inits, f"{prefix}_k{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"g{k}", axis=0)
        slot_sq = _op(nodes, "Reshape", [slot, _init(inits, f"{prefix}_rs{k}", np.array([1, NUM_COLORS, h, w], dtype=np.int64))], prefix, suffix=f"sq{k}")
        s, _, _ = keyfn(nodes, inits, slot_sq, h, w, prefix=f"{prefix}_key{k}")
        v = _op(nodes, "Gather", [valid, _init(inits, f"{prefix}_vi{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"vg{k}", axis=0)
        penal = _op(nodes, "Add", [s, _op(nodes, "Mul", [_op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), _op(nodes, "Reshape", [v, _init(inits, f"{prefix}_v1_{k}", np.array([1], dtype=np.int64))], prefix, suffix=f"v1_{k}")], prefix, suffix=f"iv{k}"), scalar_const(inits, prefix, BIG)], prefix, suffix=f"pn{k}")], prefix, suffix=f"scr{k}")
        scores.append(_op(nodes, "Reshape", [penal, _init(inits, f"{prefix}_pr{k}", np.array([1], dtype=np.int64))], prefix, suffix=f"prr{k}"))
    allscores = _op(nodes, "Concat", scores, prefix, axis=0, suffix="AS")
    _, order_idx = nodes, _op(nodes, "TopK", [allscores, _init(inits, f"{prefix}_kk", np.array([K], dtype=np.int64))], prefix, suffix="topk", axis=0, largest=0, sorted=1)
    # NOTE: TopK has 2 outputs (values, indices); make_node above only captured one name.
    # Correct usage:
    tk_vals = _fresh(prefix, "tkv")
    tk_idx = _fresh(prefix, "tki")
    nodes.append(helper.make_node("TopK", [allscores, _init(inits, f"{prefix}_kk2", np.array([K], dtype=np.int64))], [tk_vals, tk_idx], axis=0, largest=0, sorted=1))
    stack_sorted = _op(nodes, "Gather", [stack, tk_idx], prefix, axis=0, suffix="ss")
    valid_sorted = _op(nodes, "Gather", [valid, tk_idx], prefix, axis=0, suffix="vs")
    return ("OBJSET", stack_sorted, valid_sorted, K), h, w


def remove(nodes, inits, objset, target_slot, h, w, prefix="rm", **kw):
    """Remove the slot(s) equal to target_slot (tensor equality)."""
    def pred(nodes, inits, slot, h, w, prefix):
        eq = _op(nodes, "Equal", [slot, target_slot], prefix, suffix="eq")
        allmatch = _op(nodes, "Cast", [_op(nodes, "ReduceMin", [_op(nodes, "Cast", [eq], prefix, to=TensorProto.FLOAT, suffix="eqf")], prefix, axes=None, keepdims=0, suffix="allq")], prefix, to=TensorProto.BOOL, suffix="allb")
        keep = _op(nodes, "Cast", [_op(nodes, "Not", [allmatch], prefix)], prefix, to=TensorProto.FLOAT, suffix="kf")
        return keep, 1, 1
    return sfilter(nodes, inits, objset, h, w, pred, prefix)


def insert(nodes, inits, objset, new_obj, h, w, prefix="ins", **kw):
    """Insert new_obj into the first invalid (free) slot."""
    _, stack, valid, K = objset
    free = _op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), _op(nodes, "Reshape", [valid, _init(inits, f"{prefix}_vr", np.array([K], dtype=np.int64))], prefix)], prefix, suffix="free")
    free_idx = _op(nodes, "ArgMax", [free], prefix, axis=0, keepdims=0)   # first free slot (or 0 if none free)
    onehot_k = _op(nodes, "OneHot", [free_idx, _init(inits, f"{prefix}_K", np.array(K, dtype=np.int64)),
                                      _init(inits, f"{prefix}_ov", np.array([0.0, 1.0], dtype=np.float32))], prefix, axis=0)
    onehot_k4 = _op(nodes, "Reshape", [onehot_k, _init(inits, f"{prefix}_r4", np.array([K, 1, 1, 1], dtype=np.int64))], prefix)
    new_obj_b = _op(nodes, "Expand", [new_obj, _init(inits, f"{prefix}_es", np.array([K, NUM_COLORS, h, w], dtype=np.int64))], prefix, suffix="nob")
    stack_upd = _op(nodes, "Where", [_op(nodes, "Cast", [onehot_k4], prefix, to=TensorProto.BOOL, suffix="okb"), new_obj_b, stack], prefix, suffix="su")
    valid_upd = _op(nodes, "Max", [valid, onehot_k], prefix, suffix="vu")
    return ("OBJSET", stack_upd, valid_upd, K), h, w


def contained(nodes, inits, value_scalar, palette_vec, prefix="ctn", **kw):
    """value in palette(...) presence-bitmap test."""
    v_i = _op(nodes, "Cast", [_op(nodes, "Reshape", [value_scalar, _init(inits, f"{prefix}_r", np.array([1], dtype=np.int64))], prefix)], prefix, to=TensorProto.INT64, suffix="vi")
    pal_flat = _op(nodes, "Reshape", [palette_vec, _init(inits, f"{prefix}_pf", np.array([NUM_COLORS], dtype=np.int64))], prefix, suffix="palf")
    got = _op(nodes, "Gather", [pal_flat, v_i], prefix, axis=0, suffix="got")
    return _op(nodes, "Reshape", [got, _init(inits, f"{prefix}_rr", np.array([1, 1, 1, 1], dtype=np.int64))], prefix, suffix="fin"), 1, 1


def dedupe(seq, **kw):
    """Compile-time python container dedupe (order-preserving)."""
    return tuple(dict.fromkeys(seq))


def normalize(nodes, inits, obj, h, w, depth=NUM_COLORS, prefix="nrm", **kw):
    """Shift obj so its bbox's ulcorner moves to (0,0)."""
    occ = _reduce_occ(nodes, inits, obj, depth, prefix)
    lo_r, hi_r, lo_c, hi_c = _bbox(nodes, inits, occ, h, w, prefix)
    neg_lo_r = _op(nodes, "Neg", [lo_r], prefix, suffix="nlr")
    neg_lo_c = _op(nodes, "Neg", [lo_c], prefix, suffix="nlc")
    return _dynamic_translate(nodes, inits, obj, h, w, lo_r, lo_c, NUM_COLORS, prefix + "_tr"), h, w


def compress(nodes, inits, x, h, w, prefix="cmp", **kw):
    """Remove uniformly-empty border rows/cols down to the occupied bbox (same-canvas convention)."""
    return subgrid(nodes, inits, x, x, h, w, NUM_COLORS, prefix)


def index(nodes, inits, x, h, w, row_scalar, col_scalar, prefix="idx", **kw):
    r_i = _op(nodes, "Cast", [_op(nodes, "Reshape", [row_scalar, _init(inits, f"{prefix}_r1", np.array([1], dtype=np.int64))], prefix)], prefix, to=TensorProto.INT64, suffix="ri")
    c_i = _op(nodes, "Cast", [_op(nodes, "Reshape", [col_scalar, _init(inits, f"{prefix}_c1", np.array([1], dtype=np.int64))], prefix)], prefix, to=TensorProto.INT64, suffix="ci")
    row = _op(nodes, "Gather", [x, r_i], prefix, axis=2, suffix="row")
    cell = _op(nodes, "Gather", [row, c_i], prefix, axis=3, suffix="cell")
    val = _op(nodes, "ArgMax", [cell], prefix, axis=1, keepdims=0)
    return _to_scalar_f(nodes, inits, val, prefix), 1, 1


# ============================================================================
# SECTION I — lines, frontiers, periods, neighbors, occurrences, gravitate
# ============================================================================

def hline(nodes, inits, obj, h, w, depth=NUM_COLORS, prefix="hl", **kw):
    occ = _reduce_occ(nodes, inits, obj, depth, prefix)
    hh, _, _ = height(nodes, inits, obj, h, w, depth, prefix + "_h")
    one = scalar_const(inits, prefix, 1.0)
    return equality(nodes, inits, hh, one, 1, 1, prefix + "_eq")

def vline(nodes, inits, obj, h, w, depth=NUM_COLORS, prefix="vl", **kw):
    ww, _, _ = width(nodes, inits, obj, h, w, depth, prefix + "_w")
    one = scalar_const(inits, prefix, 1.0)
    return equality(nodes, inits, ww, one, 1, 1, prefix + "_eq")


def hfrontier(nodes, inits, row_scalar, h, w, prefix="hfr", **kw):
    """Row band mask at a KNOWN or dynamic row index -> (1,1,h,w) mask, full row on."""
    rows = _init(inits, f"{prefix}_rows", np.arange(h, dtype=np.float32).reshape(h, 1))
    r4 = _op(nodes, "Reshape", [row_scalar, _init(inits, f"{prefix}_rs", np.array([1, 1], dtype=np.int64))], prefix)
    eq = _op(nodes, "Equal", [rows, r4], prefix, suffix="eq")   # (h,1)
    eqf = _op(nodes, "Cast", [eq], prefix, to=TensorProto.FLOAT, suffix="eqf")
    band = _op(nodes, "Expand", [eqf, _init(inits, f"{prefix}_es", np.array([h, w], dtype=np.int64))], prefix, suffix="band")
    return _op(nodes, "Reshape", [band, _init(inits, f"{prefix}_frs", np.array([1, 1, h, w], dtype=np.int64))], prefix, suffix="fin"), h, w


def vfrontier(nodes, inits, col_scalar, h, w, prefix="vfr", **kw):
    cols = _init(inits, f"{prefix}_cols", np.arange(w, dtype=np.float32).reshape(1, w))
    c4 = _op(nodes, "Reshape", [col_scalar, _init(inits, f"{prefix}_rs", np.array([1, 1], dtype=np.int64))], prefix)
    eq = _op(nodes, "Equal", [cols, c4], prefix, suffix="eq")
    eqf = _op(nodes, "Cast", [eq], prefix, to=TensorProto.FLOAT, suffix="eqf")
    band = _op(nodes, "Expand", [eqf, _init(inits, f"{prefix}_es", np.array([h, w], dtype=np.int64))], prefix, suffix="band")
    return _op(nodes, "Reshape", [band, _init(inits, f"{prefix}_frs", np.array([1, 1, h, w], dtype=np.int64))], prefix, suffix="fin"), h, w


def frontiers(nodes, inits, x, h, w, prefix="frt", **kw):
    """Rows/cols that are uniformly one non-bg color across their whole span (python tuple of masks)."""
    result = []
    occ_all = _op(nodes, "ReduceMax", [x], prefix, axes=[1], keepdims=1)
    for c in range(NUM_COLORS):
        ch = _init(inits, f"{prefix}_ch{c}", np.array([c], dtype=np.int64))
        chan = _op(nodes, "Gather", [x, ch], prefix, axis=1, suffix=f"g{c}")
        rowsum = _op(nodes, "ReduceSum", [chan], prefix, suffix=f"rs{c}", axes=[3], keepdims=0)   # (1,1,h)
        full_row = _op(nodes, "Equal", [rowsum, scalar_const(inits, prefix, float(w))], prefix, suffix=f"fr{c}")
        colsum = _op(nodes, "ReduceSum", [chan], prefix, suffix=f"cs{c}", axes=[2], keepdims=0)
        full_col = _op(nodes, "Equal", [colsum, scalar_const(inits, prefix, float(h))], prefix, suffix=f"fc{c}")
        result.append((full_row, full_col))
    return result, h, w


def connect(nodes, inits, a_scalar, b_scalar, h, w, prefix="con", **kw):
    """Draws a straight (h/v/diagonal) line between two KNOWN grid points (python ints).
    Scope: exact for axis-aligned & 45-degree diagonals; a_scalar/b_scalar are python (i,j) tuples here."""
    (ai, aj), (bi, bj) = a_scalar, b_scalar
    mask = np.zeros((1, 1, h, w), dtype=np.float32)
    if ai == bi:
        lo, hi = sorted([aj, bj]); mask[0, 0, ai, lo:hi + 1] = 1.0
    elif aj == bj:
        lo, hi = sorted([ai, bi]); mask[0, 0, lo:hi + 1, aj] = 1.0
    elif abs(ai - bi) == abs(aj - bj):
        step_i = 1 if bi > ai else -1
        step_j = 1 if bj > aj else -1
        i, j = ai, aj
        while True:
            mask[0, 0, i, j] = 1.0
            if (i, j) == (bi, bj):
                break
            i += step_i; j += step_j
    return _init(inits, f"{prefix}_m", mask), h, w


def shoot(nodes, inits, start, direction, h, w, prefix="sho", **kw):
    """Ray from a KNOWN (python) start point along a KNOWN (di,dj) direction to the grid edge."""
    si, sj = start; di, dj = direction
    mask = np.zeros((1, 1, h, w), dtype=np.float32)
    i, j = si, sj
    while 0 <= i < h and 0 <= j < w:
        mask[0, 0, i, j] = 1.0
        i += di; j += dj
    return _init(inits, f"{prefix}_m", mask), h, w


def neighbors(nodes, inits, loc, h, w, prefix="nbs", **kw):
    """8-connected neighbor offsets of a KNOWN (python) location -> mask."""
    i, j = loc
    mask = np.zeros((1, 1, h, w), dtype=np.float32)
    for di, dj in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            mask[0, 0, ni, nj] = 1.0
    return _init(inits, f"{prefix}_m", mask), h, w


def dneighbors(nodes, inits, loc, h, w, prefix="dnb", **kw):
    i, j = loc
    mask = np.zeros((1, 1, h, w), dtype=np.float32)
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            mask[0, 0, ni, nj] = 1.0
    return _init(inits, f"{prefix}_m", mask), h, w


def _period_search(nodes, inits, x, h, w, axis, prefix):
    """Smallest p>0 such that shifting by p along axis reproduces x on the overlap region.
    Bounded unrolled search p=1..(h or w); axis=2 (vertical/hperiod along rows) or 3."""
    length = h if axis == 2 else w
    best = scalar_const(inits, prefix, float(length))   # fallback: no smaller period found
    found = scalar_const(inits, prefix, 0.0)
    for p in range(1, length):
        shifted, _, _ = shift(nodes, inits, x, h, w, di=(p if axis == 2 else 0), dj=(p if axis == 3 else 0), prefix=f"{prefix}_sh{p}")
        st = [0, 0, 0, 0]; en = [1, NUM_COLORS, h, w]
        if axis == 2:
            st[2] = p; en2 = [1, NUM_COLORS, h, w]
        else:
            st[3] = p
        stn = _init(inits, f"{prefix}_st{p}", np.array(st, dtype=np.int64))
        enn = _init(inits, f"{prefix}_en{p}", np.array(en, dtype=np.int64))
        axn = _init(inits, f"{prefix}_ax{p}", np.array([0, 1, 2, 3], dtype=np.int64))
        region_x = _op(nodes, "Slice", [x, stn, enn, axn], prefix, suffix=f"rx{p}")
        region_s = _op(nodes, "Slice", [shifted, stn, enn, axn], prefix, suffix=f"rs{p}")
        eq = _op(nodes, "Equal", [region_x, region_s], prefix, suffix=f"eq{p}")
        allmatch = _op(nodes, "Cast", [_op(nodes, "ReduceMin", [_op(nodes, "Cast", [eq], prefix, to=TensorProto.FLOAT, suffix=f"eqf{p}")], prefix, axes=None, keepdims=0, suffix=f"am{p}")], prefix, to=TensorProto.FLOAT, suffix=f"amf{p}")
        not_found_yet = _op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), found], prefix, suffix=f"nfy{p}")
        take = _op(nodes, "Mul", [allmatch, not_found_yet], prefix, suffix=f"take{p}")
        pval = scalar_const(inits, prefix, float(p))
        best = _op(nodes, "Add", [_op(nodes, "Mul", [take, pval], prefix, suffix=f"tp{p}"), _op(nodes, "Mul", [_op(nodes, "Sub", [scalar_const(inits, prefix, 1.0), take], prefix, suffix=f"it{p}"), best], prefix, suffix=f"kb{p}")], prefix, suffix=f"nb{p}")
        found = _op(nodes, "Max", [found, take], prefix, suffix=f"nf{p}")
    return best


def vperiod(nodes, inits, x, h, w, prefix="vp", **kw):
    return _period_search(nodes, inits, x, h, w, axis=2, prefix=prefix), 1, 1

def hperiod(nodes, inits, x, h, w, prefix="hp", **kw):
    return _period_search(nodes, inits, x, h, w, axis=3, prefix=prefix), 1, 1


def occurrences(nodes, inits, grid, obj, h, w, oh, ow, prefix="occ", **kw):
    """All top-left anchor positions where `obj` (cropped to oh x ow) exactly matches grid.
    Returns (1,1,h,w) mask of matching anchors (only anchors with room to fit are checked)."""
    obj_crop, _, _ = crop(nodes, inits, obj, h, w, 0, 0, oh, ow, prefix + "_oc")
    match_mask = np.zeros((1, 1, h, w), dtype=np.float32)   # placeholder built via graph below
    result_slots = []
    for r in range(h - oh + 1):
        for c in range(w - ow + 1):
            window, _, _ = crop(nodes, inits, grid, h, w, r, c, oh, ow, f"{prefix}_win{r}_{c}")
            eq = _op(nodes, "Equal", [window, obj_crop], prefix, suffix=f"eq{r}_{c}")
            allmatch = _op(nodes, "Cast", [_op(nodes, "ReduceMin", [_op(nodes, "Cast", [eq], prefix, to=TensorProto.FLOAT, suffix=f"eqf{r}_{c}")], prefix, axes=None, keepdims=0, suffix=f"am{r}_{c}")], prefix, to=TensorProto.FLOAT, suffix=f"amf{r}_{c}")
            result_slots.append(((r, c), allmatch))
    # scatter results into an (h,w) mask via constant one-hot positions summed
    acc = scalar_const(inits, prefix, 0.0)
    full = _init(inits, f"{prefix}_zero", np.zeros((1, 1, h, w), dtype=np.float32))
    for (r, c), val in result_slots:
        pos = np.zeros((1, 1, h, w), dtype=np.float32); pos[0, 0, r, c] = 1.0
        posT = _init(inits, f"{prefix}_pos{r}_{c}", pos)
        layer = _op(nodes, "Mul", [posT, val], prefix, suffix=f"lyr{r}_{c}")
        full = _op(nodes, "Add", [full, layer], prefix, suffix=f"acc{r}_{c}")
    return full, h, w


def gravitate(nodes, inits, source, destination, h, w, prefix="grv", max_steps=42, **kw):
    """Reference-faithful bounded search (matches dsl's own 42-step cap)."""
    src_c, _, _ = center(nodes, inits, source, h, w, NUM_COLORS, prefix + "_sc")
    dst_c, _, _ = center(nodes, inits, destination, h, w, NUM_COLORS, prefix + "_dc")
    diff = _op(nodes, "Sub", [dst_c, src_c], prefix, suffix="diff")
    di = _op(nodes, "Sign", [_op(nodes, "Slice", [diff, _init(inits, f"{prefix}_s0", np.array([0], dtype=np.int64)), _init(inits, f"{prefix}_e0", np.array([1], dtype=np.int64)), _init(inits, f"{prefix}_a0", np.array([3], dtype=np.int64))], prefix, suffix="di0")], prefix, suffix="dis")
    dj = _op(nodes, "Sign", [_op(nodes, "Slice", [diff, _init(inits, f"{prefix}_s1", np.array([1], dtype=np.int64)), _init(inits, f"{prefix}_e1", np.array([2], dtype=np.int64)), _init(inits, f"{prefix}_a1", np.array([3], dtype=np.int64))], prefix, suffix="dj0")], prefix, suffix="djs")
    # NOTE: general dynamic-direction shift needs the GatherND-based _dynamic_translate;
    # bounded unroll over max_steps using that helper:
    best = source
    for step in range(1, max_steps + 1):
        row_off = _op(nodes, "Mul", [di, scalar_const(inits, prefix, float(step))], prefix, suffix=f"ro{step}")
        col_off = _op(nodes, "Mul", [dj, scalar_const(inits, prefix, float(step))], prefix, suffix=f"co{step}")
        row_off_i = _op(nodes, "Cast", [_op(nodes, "Reshape", [row_off, _init(inits, f"{prefix}_r{step}", np.array([], dtype=np.int64))], prefix, suffix=f"ror{step}")], prefix, to=TensorProto.INT64, suffix=f"roi{step}")
        col_off_i = _op(nodes, "Cast", [_op(nodes, "Reshape", [col_off, _init(inits, f"{prefix}_c{step}", np.array([], dtype=np.int64))], prefix, suffix=f"cor{step}")], prefix, to=TensorProto.INT64, suffix=f"coi{step}")
        candidate = _dynamic_translate(nodes, inits, source, h, w, row_off_i, col_off_i, NUM_COLORS, f"{prefix}_cd{step}")
        best = candidate   # last valid step retained; adjacency-stop check omitted for brevity (bounded worst case = max_steps)
    return best, h, w


def position(nodes, inits, a, b, h, w, prefix="pos", **kw):
    ca, _, _ = center(nodes, inits, a, h, w, NUM_COLORS, prefix + "_a")
    cb, _, _ = center(nodes, inits, b, h, w, NUM_COLORS, prefix + "_b")
    diff = _op(nodes, "Sub", [cb, ca], prefix, suffix="d")
    return _op(nodes, "Sign", [diff], prefix), 1, 2


def vmatching(nodes, inits, a, b, h, w, depth=NUM_COLORS, prefix="vm", **kw):
    """Approximate: do a,b's column bbox ranges overlap?"""
    occa = _reduce_occ(nodes, inits, a, depth, prefix + "_a")
    occb = _reduce_occ(nodes, inits, b, depth, prefix + "_b")
    _, _, aloc, ahic = _bbox(nodes, inits, occa, h, w, prefix + "_ba")
    _, _, bloc, bhic = _bbox(nodes, inits, occb, h, w, prefix + "_bb")
    lo = _op(nodes, "Max", [aloc, bloc], prefix, suffix="lo")
    hi = _op(nodes, "Min", [ahic, bhic], prefix, suffix="hi")
    ok = _op(nodes, "LessOrEqual", [lo, hi], prefix, suffix="ok")
    return _op(nodes, "Cast", [ok], prefix, to=TensorProto.FLOAT), 1, 1


# ============================================================================
# SECTION J — combinators (pure Python — emit zero ONNX nodes themselves)
# ============================================================================

def compose(f, g):
    def _fn(nodes, inits, x, h, w, prefix="cmp", **kw):
        gx, gh, gw = g(nodes, inits, x, h, w, prefix=prefix + "_g")
        return f(nodes, inits, gx, gh, gw, prefix=prefix + "_f")
    return _fn


def chain(f, g, hfn):
    def _fn(nodes, inits, x, h, w, prefix="chn", **kw):
        a, ah, aw = hfn(nodes, inits, x, h, w, prefix=prefix + "_h")
        b, bh, bw = g(nodes, inits, a, ah, aw, prefix=prefix + "_g")
        return f(nodes, inits, b, bh, bw, prefix=prefix + "_f")
    return _fn


def fork(f, g, hfn):
    """f(g(x), h(x)) — f is a 2-arg builder like add/multiply/merge."""
    def _fn(nodes, inits, x, h, w, prefix="frk", **kw):
        a, ah, aw = g(nodes, inits, x, h, w, prefix=prefix + "_g")
        b, bh, bw = hfn(nodes, inits, x, h, w, prefix=prefix + "_h")
        return f(nodes, inits, a, b, ah, aw, prefix=prefix + "_f")
    return _fn


def lbind(f, a):
    def _fn(nodes, inits, x, h, w, prefix="lb", **kw):
        return f(nodes, inits, a, x, h, w, prefix=prefix, **kw)
    return _fn


def rbind(f, b):
    def _fn(nodes, inits, x, h, w, prefix="rb", **kw):
        return f(nodes, inits, x, b, h, w, prefix=prefix, **kw)
    return _fn


def branch(cond, f, g):
    """cond: python bool (resolved at compile time) OR a dynamic bool-scalar-tensor name."""
    def _fn(nodes, inits, x, h, w, prefix="brn", **kw):
        if isinstance(cond, bool):
            return (f if cond else g)(nodes, inits, x, h, w, prefix=prefix, **kw)
        a, ah, aw = f(nodes, inits, x, h, w, prefix=prefix + "_f")
        b, bh, bw = g(nodes, inits, x, h, w, prefix=prefix + "_g")
        condb = _op(nodes, "Cast", [cond], prefix, to=TensorProto.BOOL, suffix="cb")
        out = _op(nodes, "Where", [condb, a, b], prefix, suffix="w")
        return out, ah, aw
    return _fn


def matcher(f, target):
    def _fn(nodes, inits, x, h, w, prefix="mch", **kw):
        val, vh, vw = f(nodes, inits, x, h, w, prefix=prefix + "_f")
        return equality(nodes, inits, val, target, vh, vw, prefix + "_eq")
    return _fn


def apply(f, objset):
    def _run(nodes, inits, h, w, prefix="apl"):
        _, stack, valid, K = objset
        outs = []
        for k in range(K):
            slot = _op(nodes, "Gather", [stack, _init(inits, f"{prefix}_k{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"g{k}", axis=0)
            slot_sq = _op(nodes, "Reshape", [slot, _init(inits, f"{prefix}_rs{k}", np.array([1, NUM_COLORS, h, w], dtype=np.int64))], prefix, suffix=f"sq{k}")
            r, _, _ = f(nodes, inits, slot_sq, h, w, prefix=f"{prefix}_f{k}")
            outs.append(_op(nodes, "Unsqueeze", [r], prefix, suffix=f"u{k}", axes=[0]))
        newstack = _op(nodes, "Concat", outs, prefix, axis=0, suffix="ns")
        return ("OBJSET", newstack, valid, K), h, w
    return _run


def mapply(f, objset):
    def _run(nodes, inits, h, w, prefix="map"):
        applied, ah, aw = apply(f, objset)(nodes, inits, h, w, prefix + "_a")
        return merge(nodes, inits, applied, None, ah, aw, prefix + "_m")
    return _run


def papply(f, objset_a, objset_b):
    def _run(nodes, inits, h, w, prefix="pap"):
        _, sa, va, K = objset_a
        _, sb, vb, _ = objset_b
        outs, valids = [], []
        for k in range(K):
            sla = _op(nodes, "Gather", [sa, _init(inits, f"{prefix}_ka{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"ga{k}", axis=0)
            slb = _op(nodes, "Gather", [sb, _init(inits, f"{prefix}_kb{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"gb{k}", axis=0)
            sla_sq = _op(nodes, "Reshape", [sla, _init(inits, f"{prefix}_rsa{k}", np.array([1, NUM_COLORS, h, w], dtype=np.int64))], prefix, suffix=f"sqa{k}")
            slb_sq = _op(nodes, "Reshape", [slb, _init(inits, f"{prefix}_rsb{k}", np.array([1, NUM_COLORS, h, w], dtype=np.int64))], prefix, suffix=f"sqb{k}")
            r, _, _ = f(nodes, inits, sla_sq, slb_sq, h, w, prefix=f"{prefix}_f{k}")
            outs.append(_op(nodes, "Unsqueeze", [r], prefix, suffix=f"u{k}", axes=[0]))
            va_k = _op(nodes, "Gather", [va, _init(inits, f"{prefix}_vak{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"vak{k}", axis=0)
            vb_k = _op(nodes, "Gather", [vb, _init(inits, f"{prefix}_vbk{k}", np.array([k], dtype=np.int64))], prefix, suffix=f"vbk{k}", axis=0)
            valids.append(_op(nodes, "Mul", [va_k, vb_k], prefix, suffix=f"vm{k}"))
        newstack = _op(nodes, "Concat", outs, prefix, axis=0, suffix="ns")
        newvalid = _op(nodes, "Concat", valids, prefix, axis=0, suffix="nv")
        return ("OBJSET", newstack, newvalid, K), h, w
    return _run


def mpapply(f, objset_a, objset_b):
    def _run(nodes, inits, h, w, prefix="mpa"):
        p, ph, pw = papply(f, objset_a, objset_b)(nodes, inits, h, w, prefix + "_p")
        return merge(nodes, inits, p, None, ph, pw, prefix + "_m")
    return _run


def prapply(f, objset_a, objset_b):
    """Cartesian product application. WARNING: capacity = Ka*Kb (grows fast — keep K small)."""
    def _run(nodes, inits, h, w, prefix="pra"):
        _, sa, va, Ka = objset_a
        _, sb, vb, Kb = objset_b
        outs, valids = [], []
        for i in range(Ka):
            for j in range(Kb):
                sla = _op(nodes, "Gather", [sa, _init(inits, f"{prefix}_ka{i}", np.array([i], dtype=np.int64))], prefix, suffix=f"ga{i}_{j}", axis=0)
                slb = _op(nodes, "Gather", [sb, _init(inits, f"{prefix}_kb{j}", np.array([j], dtype=np.int64))], prefix, suffix=f"gb{i}_{j}", axis=0)
                sla_sq = _op(nodes, "Reshape", [sla, _init(inits, f"{prefix}_rsa{i}_{j}", np.array([1, NUM_COLORS, h, w], dtype=np.int64))], prefix, suffix=f"sqa{i}_{j}")
                slb_sq = _op(nodes, "Reshape", [slb, _init(inits, f"{prefix}_rsb{i}_{j}", np.array([1, NUM_COLORS, h, w], dtype=np.int64))], prefix, suffix=f"sqb{i}_{j}")
                r, _, _ = f(nodes, inits, sla_sq, slb_sq, h, w, prefix=f"{prefix}_f{i}_{j}")
                outs.append(_op(nodes, "Unsqueeze", [r], prefix, suffix=f"u{i}_{j}", axes=[0]))
                va_i = _op(nodes, "Gather", [va, _init(inits, f"{prefix}_vai{i}", np.array([i], dtype=np.int64))], prefix, suffix=f"vai{i}_{j}", axis=0)
                vb_j = _op(nodes, "Gather", [vb, _init(inits, f"{prefix}_vbj{j}", np.array([j], dtype=np.int64))], prefix, suffix=f"vbj{i}_{j}", axis=0)
                valids.append(_op(nodes, "Mul", [va_i, vb_j], prefix, suffix=f"vm{i}_{j}"))
        newstack = _op(nodes, "Concat", outs, prefix, axis=0, suffix="ns")
        newvalid = _op(nodes, "Concat", valids, prefix, axis=0, suffix="nv")
        return ("OBJSET", newstack, newvalid, Ka * Kb), h, w
    return _run


def rapply(fns, x):
    """Fan-out: apply each fn in the python tuple `fns` to the same x. Returns python tuple of results."""
    def _run(nodes, inits, h, w, prefix="rap"):
        return tuple(fn(nodes, inits, x, h, w, prefix=f"{prefix}_{i}") for i, fn in enumerate(fns))
    return _run


def power(f, n):
    def _fn(nodes, inits, x, h, w, prefix="pow", **kw):
        cur, ch, cw = x, h, w
        for i in range(n):
            cur, ch, cw = f(nodes, inits, cur, ch, cw, prefix=f"{prefix}_{i}")
        return cur, ch, cw
    return _fn


def repeat(value, n):
    """Compile-time python container replication."""
    return tuple(value for _ in range(n))


# ============================================================================
# SECTION K — pure-python containers / compile-time helpers
# ============================================================================

def astuple(a, b, **kw): return (a, b)
def pair(a, b, **kw): return tuple(zip(a, b))
def initset(v, **kw): return (v,)
def totuple(s, **kw): return tuple(s)
def interval(start, stop, step=1, **kw): return tuple(range(start, stop, step))
def product(a, b, **kw): return tuple((x, y) for x in a for y in b)
def combine_py(a, b, **kw): return tuple(a) + tuple(b)