"""
test_all_primitives.py — exhaustive registry-based test harness for arc_onnx_primitives.py

Strategy
--------
For every primitive we build a *tiny* ONNX graph exercising just that primitive
(or, for combinators/containers, a minimal concrete pipeline built out of it),
run it with onnxruntime, and compare against an INDEPENDENTLY computed
reference value (usually plain numpy/python, deliberately NOT re-using the
module's own logic) so a bug in the primitive can't hide behind a matching bug
in the test.

Encoding conventions used here
-------------------------------
* "raw grid" encoding  (encode_grid(g))       -> one-hot at EVERY cell,
  including background. Use for: geometric transforms, color-histogram stats
  (mostcolor/colorcount/...), fill/replace/switch/index, i.e. anything that
  is supposed to see the *whole* grid including its background color.

* "object" encoding    (encode_grid(g, bg=0)) -> cells equal to `bg` are left
  as an all-zero channel vector (nothing one-hot'd). Use for: height/width/
  bbox/corners/center/backdrop/box/... and anything from objects()/partition()
  — these primitives compute occupancy via ReduceMax over the channel axis,
  which only makes sense if "no color" is truly the zero vector.

Known, intentionally-probed limitations of the reference implementation
-------------------------------------------------------------------------
* gravitate(): the bounded unroll does not implement the "stop once adjacent"
  early-exit the DSL specifies, so we only test it at max_steps=1 (a single
  correct unit-step), not full convergence.
* order(): contains an extra malformed TopK node (built via the generic
  single-output `_op` helper) before the real, correctly-shaped TopK node.
  If your onnx/onnxruntime build's shape inference rejects the first
  malformed node, `order` will legitimately FAIL here — that's a real bug
  report, not a test bug.
* connect/shoot/neighbors/dneighbors/canvas: compile-time-constant-only
  (python ints), so they're tested via baked-in constants + dummy input.
* occurrences(): only exact for small grids (fully unrolled anchor search).
"""

import numpy as np
import onnxruntime as ort

from arc_onnx_primitives import *          # public primitives
from arc_onnx_primitives import _init, _op  # need these two internals for
                                             # building small hand-rolled
                                             # constant graphs in some tests

# ============================================================================
# infra
# ============================================================================

def encode_grid(g, bg=None):
    """bg=None -> raw one-hot-everywhere grid. bg=<int> -> 'object' encoding
    (cells equal to bg become the all-zero channel vector)."""
    g = np.array(g, dtype=np.int64)
    h, w = g.shape
    x = np.zeros((1, NUM_COLORS, h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            v = int(g[i, j])
            if bg is not None and v == bg:
                continue
            x[0, v, i, j] = 1.0
    return x, h, w


def decode_grid(t):
    return np.argmax(t[0], axis=0).astype(int)


def decode_mask(t):
    return (t[0, 0] > 0.5).astype(np.int64)


def decode_scalar(t):
    return float(np.array(t).reshape(-1)[0])


def decode_vec(t):
    return np.array(t).reshape(-1).tolist()


def run(build_fn, x_in, h, w, out_shape):
    """build_fn(nodes, inits, 'input', h, w) -> out_name"""
    nodes, inits = [], []
    out_name = build_fn(nodes, inits, "input", h, w)
    model = make_model(nodes, inits, in_shape=(1, NUM_COLORS, h, w),
                        out_name=out_name, out_shape=out_shape)
    sess = ort.InferenceSession(model.SerializeToString())
    return sess.run(["output"], {"input": x_in})[0]


def run_const(build_fn, out_shape):
    """build_fn(nodes, inits) -> out_name, uses only baked-in constants,
    a dummy (1,10,1,1) input is wired but never consumed."""
    nodes, inits = [], []
    out_name = build_fn(nodes, inits)
    dummy_shape = (1, NUM_COLORS, 1, 1)
    dummy = np.zeros(dummy_shape, dtype=np.float32)
    model = make_model(nodes, inits, in_shape=dummy_shape,
                        out_name=out_name, out_shape=out_shape)
    sess = ort.InferenceSession(model.SerializeToString())
    return sess.run(["output"], {"input": dummy})[0]


def const_grid(inits, name, grid, bg=None):
    x, h, w = encode_grid(grid, bg=bg)
    return _init(inits, name, x), h, w


REGISTRY = []

def T(name, fn):
    try:
        ok, info = fn()
        ok = bool(ok)
    except Exception as e:
        ok, info = False, f"{type(e).__name__}: {e}"
    REGISTRY.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"   -> {info}"))


# reference numpy transforms (independent of the module's own code)
def ref_hmirror(g):  return np.flipud(np.array(g))
def ref_vmirror(g):  return np.fliplr(np.array(g))
def ref_dmirror(g):  return np.array(g).T
def ref_cmirror(g):  return np.fliplr(np.flipud(np.array(g))).T
def ref_rot90(g):    return np.rot90(g, -1)
def ref_rot270(g):   return np.rot90(g, 1)
def ref_rot180(g):   return np.rot90(g, 2)


def check_grid_eq(name, build_fn, grid, expected, bg=None, out_shape=None):
    def _t():
        x, h, w = encode_grid(grid, bg=bg)
        oh, ow = np.array(expected).shape
        shp = out_shape or (1, NUM_COLORS, oh, ow)
        r = run(build_fn, x, h, w, shp)
        got = decode_grid(r)
        exp = np.array(expected)
        return (got.shape == exp.shape and np.array_equal(got, exp)), got
    T(name, _t)


def check_mask_eq(name, build_fn, grid, expected_mask, bg=None):
    def _t():
        x, h, w = encode_grid(grid, bg=bg)
        r = run(build_fn, x, h, w, (1, 1, h, w))
        got = decode_mask(r)
        exp = np.array(expected_mask, dtype=np.int64)
        return np.array_equal(got, exp), got
    T(name, _t)


def check_scalar_eq(name, build_fn, grid, expected, bg=None, tol=1e-4):
    def _t():
        x, h, w = encode_grid(grid, bg=bg)
        r = run(build_fn, x, h, w, (1, 1, 1, 1))
        got = decode_scalar(r)
        return abs(got - expected) < tol, got
    T(name, _t)


def check_vec_eq(name, build_fn, grid, expected, bg=None):
    def _t():
        x, h, w = encode_grid(grid, bg=bg)
        r = run(build_fn, x, h, w, (1, 1, 1, 2))
        got = decode_vec(r)
        return got == [float(v) for v in expected], got
    T(name, _t)


print("=" * 78)
print("SECTION A — scalar arithmetic & logic (21 primitives)")
print("=" * 78)

def test_binop(name, fn, a, b, expected, **kw):
    def _t():
        r = run_const(lambda n, i: fn(n, i, scalar_const(i, 'a', a),
                                       scalar_const(i, 'b', b), 1, 1,
                                       prefix='t', **kw)[0], (1, 1, 1, 1))
        got = decode_scalar(r)
        return abs(got - expected) < 1e-4, got
    T(name, _t)

def test_unop(name, fn, a, expected, **kw):
    def _t():
        r = run_const(lambda n, i: fn(n, i, scalar_const(i, 'a', a), 1, 1,
                                       prefix='t', **kw)[0], (1, 1, 1, 1))
        got = decode_scalar(r)
        return abs(got - expected) < 1e-4, got
    T(name, _t)

test_binop("add",        add,        3, 4, 7)
test_binop("subtract",   subtract,   7, 4, 3)
test_binop("multiply",   multiply,   3, 4, 12)
test_binop("divide",     divide,     12, 4, 3)
test_binop("minimum",    minimum,    3, 7, 3)
test_binop("maximum",    maximum,    3, 7, 7)
test_binop("cellwise(mul)",
           lambda n,i,x1,x2,h,w,prefix,**kw: cellwise(n,i,x1,x2,h,w,func='mul',prefix=prefix),
           2, 5, 10)
test_binop("equality(true)", equality, 3, 3, 1)
test_binop("equality(false)", equality, 3, 4, 0)
test_binop("greater",    greater,    5, 3, 1)
test_binop("both(0)",    both,       1, 0, 0)
test_binop("both(1)",    both,       1, 1, 1)

test_unop("increment", increment, 5, 6, delta=1)
test_unop("decrement", decrement, 5, 4, delta=1)
test_unop("crement(+)", crement, 5, 6)
test_unop("crement(-)", crement, -5, -6)
test_unop("double", double, 5, 10)
test_unop("halve", halve, 10, 5)
test_unop("negate", negate, 5, -5)
test_unop("sign", sign, -5, -1)
test_unop("positive(true)", positive, 5, 1)
test_unop("positive(false)", positive, -5, 0)
test_unop("invert", invert, 5, -5)
test_unop("even(true)", even, 4, 1)
test_unop("even(false)", even, 5, 0)
test_unop("flip", flip, 0, 1)


print("=" * 78)
print("SECTION B — geometric transforms (23 primitives)")
print("=" * 78)

GRID_B = [[0, 0, 0, 0],
          [0, 1, 1, 0],
          [0, 1, 1, 0],
          [0, 0, 2, 0]]

GRID_R = [[1, 2, 3],
          [4, 5, 0]]   # 2x3, non-square, exercises transpose-shape ops

check_grid_eq("hmirror", lambda n,i,x,h,w: hmirror(n,i,x,h,w)[0], GRID_B, ref_hmirror(GRID_B))
check_grid_eq("vmirror", lambda n,i,x,h,w: vmirror(n,i,x,h,w)[0], GRID_B, ref_vmirror(GRID_B))
check_grid_eq("dmirror", lambda n,i,x,h,w: dmirror(n,i,x,h,w)[0], GRID_R, ref_dmirror(GRID_R))
check_grid_eq("cmirror", lambda n,i,x,h,w: cmirror(n,i,x,h,w)[0], GRID_R, ref_cmirror(GRID_R))
check_grid_eq("rot90",   lambda n,i,x,h,w: rot90(n,i,x,h,w)[0],   GRID_R, ref_rot90(GRID_R))
check_grid_eq("rot270",  lambda n,i,x,h,w: rot270(n,i,x,h,w)[0],  GRID_R, ref_rot270(GRID_R))
check_grid_eq("rot180",  lambda n,i,x,h,w: rot180(n,i,x,h,w)[0],  GRID_R, ref_rot180(GRID_R))

check_grid_eq("tophalf",    lambda n,i,x,h,w: tophalf(n,i,x,h,w)[0],    GRID_B, np.array(GRID_B)[:2])
check_grid_eq("bottomhalf", lambda n,i,x,h,w: bottomhalf(n,i,x,h,w)[0], GRID_B, np.array(GRID_B)[2:])
check_grid_eq("lefthalf",   lambda n,i,x,h,w: lefthalf(n,i,x,h,w)[0],   GRID_B, np.array(GRID_B)[:, :2])
check_grid_eq("righthalf",  lambda n,i,x,h,w: righthalf(n,i,x,h,w)[0],  GRID_B, np.array(GRID_B)[:, 2:])

check_grid_eq(
    "hsplit (roundtrip via hconcat)",
    lambda n,i,x,h,w: hconcat(n, i, *hsplit(n, i, x, h, w, n=2)[0], h, w // 2, w // 2)[0],
    GRID_B, GRID_B,
)
check_grid_eq(
    "vsplit (roundtrip via vconcat)",
    lambda n,i,x,h,w: vconcat(n, i, *vsplit(n, i, x, h, w, n=2)[0], h // 2, h // 2, w)[0],
    GRID_B, GRID_B,
)

def test_vconcat():
    def _t():
        nodes, inits = [], []
        a_name, ah, aw = const_grid(inits, "vc_a", [[1, 1], [2, 2]])
        b_name, bh, bw = const_grid(inits, "vc_b", [[3, 3]])
        out_name, oh, ow = vconcat(nodes, inits, a_name, b_name, ah, bh, aw)
        r = run_const_model(nodes, inits, out_name, (1, NUM_COLORS, oh, ow))
        got = decode_grid(r)
        exp = np.array([[1, 1], [2, 2], [3, 3]])
        return np.array_equal(got, exp), got
    T("vconcat", _t)

def test_hconcat():
    def _t():
        nodes, inits = [], []
        a_name, ah, aw = const_grid(inits, "hc_a", [[1], [2]])
        b_name, bh, bw = const_grid(inits, "hc_b", [[3, 3], [4, 4]])
        out_name, oh, ow = hconcat(nodes, inits, a_name, b_name, ah, aw, bw)
        r = run_const_model(nodes, inits, out_name, (1, NUM_COLORS, oh, ow))
        got = decode_grid(r)
        exp = np.array([[1, 3, 3], [2, 4, 4]])
        return np.array_equal(got, exp), got
    T("hconcat", _t)

def run_const_model(nodes, inits, out_name, out_shape):
    dummy_shape = (1, NUM_COLORS, 1, 1)
    dummy = np.zeros(dummy_shape, dtype=np.float32)
    model = make_model(nodes, inits, in_shape=dummy_shape, out_name=out_name, out_shape=out_shape)
    sess = ort.InferenceSession(model.SerializeToString())
    return sess.run(["output"], {"input": dummy})[0]

test_vconcat()
test_hconcat()

check_grid_eq("hupscale(x2)", lambda n,i,x,h,w: hupscale(n,i,x,h,w,factor=2)[0],
              GRID_B, np.repeat(GRID_B, 2, axis=1))
check_grid_eq("vupscale(x2)", lambda n,i,x,h,w: vupscale(n,i,x,h,w,factor=2)[0],
              GRID_B, np.repeat(GRID_B, 2, axis=0))
check_grid_eq("upscale(x2)", lambda n,i,x,h,w: upscale(n,i,x,h,w,factor=2)[0],
              GRID_B, np.kron(GRID_B, np.ones((2, 2), dtype=int)))

UP_B = np.kron(GRID_B, np.ones((2, 2), dtype=int)).tolist()
check_grid_eq("downscale(x2, roundtrip)", lambda n,i,x,h,w: downscale(n,i,x,h,w,factor=2)[0],
              UP_B, GRID_B)

check_grid_eq("crop", lambda n,i,x,h,w: crop(n,i,x,h,w,top=1,left=1,height=2,width=2)[0],
              GRID_B, np.array(GRID_B)[1:3, 1:3])
check_grid_eq("trim", lambda n,i,x,h,w: trim(n,i,x,h,w)[0],
              GRID_B, np.array(GRID_B)[1:-1, 1:-1])

def ref_shift(grid, di, dj):
    g = np.array(grid); h, w = g.shape; out = np.zeros_like(g)
    for i in range(h):
        for j in range(w):
            si, sj = i - di, j - dj
            if 0 <= si < h and 0 <= sj < w:
                out[i, j] = g[si, sj]
    return out

check_grid_eq("shift(+1,-1)", lambda n,i,x,h,w: shift(n,i,x,h,w,di=1,dj=-1)[0],
              GRID_B, ref_shift(GRID_B, 1, -1))

def ref_move(grid, color, di, dj):
    g = np.array(grid); bg = np.bincount(g.flatten()).argmax()
    covered = np.where(g == color, bg, g)
    h, w = g.shape; out = covered.copy()
    for i in range(h):
        for j in range(w):
            if g[i, j] == color:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    out[ni, nj] = color
    return out

def _move_pipeline(n, i, x, h, w):
    mask, _, _ = ofcolor(n, i, x, h, w, color=1)
    obj, _, _ = toobject(n, i, mask, x, h, w)
    return move(n, i, x, obj, h, w, di=1, dj=0)[0]

check_grid_eq("move", _move_pipeline, GRID_B, ref_move(GRID_B, 1, 1, 0))


print("=" * 78)
print("SECTION D — bounding box / shape geometry (22 primitives, OBJECT encoding)")
print("=" * 78)

GRID_D = [[0, 0, 0, 0, 0],
          [0, 3, 3, 0, 0],
          [0, 3, 3, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]

check_scalar_eq("height", lambda n,i,x,h,w: height(n,i,x,h,w)[0], GRID_D, 2.0, bg=0)
check_scalar_eq("width",  lambda n,i,x,h,w: width(n,i,x,h,w)[0],  GRID_D, 2.0, bg=0)
check_vec_eq("shape",     lambda n,i,x,h,w: shape(n,i,x,h,w)[0],  GRID_D, [2, 2], bg=0)
check_scalar_eq("size",   lambda n,i,x,h,w: size(n,i,x,h,w)[0],   GRID_D, 4.0, bg=0)
check_scalar_eq("portrait(false)", lambda n,i,x,h,w: portrait(n,i,x,h,w)[0], GRID_D, 0.0, bg=0)

check_vec_eq("ulcorner", lambda n,i,x,h,w: ulcorner(n,i,x,h,w)[0], GRID_D, [1, 1], bg=0)
check_vec_eq("urcorner", lambda n,i,x,h,w: urcorner(n,i,x,h,w)[0], GRID_D, [1, 2], bg=0)
check_vec_eq("llcorner", lambda n,i,x,h,w: llcorner(n,i,x,h,w)[0], GRID_D, [2, 1], bg=0)
check_vec_eq("lrcorner", lambda n,i,x,h,w: lrcorner(n,i,x,h,w)[0], GRID_D, [2, 2], bg=0)

def test_corners():
    def _t():
        x, h, w = encode_grid(GRID_D, bg=0)
        nodes, inits = [], []
        c, ch, cw = corners(nodes, inits, "input", h, w)
        cat = _op(nodes, "Concat", list(c), "catc", axis=3)
        model = make_model(nodes, inits, in_shape=(1, NUM_COLORS, h, w), out_name=cat, out_shape=(1, 1, 1, 8))
        sess = ort.InferenceSession(model.SerializeToString())
        r = sess.run(["output"], {"input": x})[0]
        got = r.reshape(-1).tolist()
        exp = [1, 1, 1, 2, 2, 1, 2, 2]
        return got == [float(v) for v in exp], got
    T("corners", _t)
test_corners()

check_scalar_eq("uppermost", lambda n,i,x,h,w: uppermost(n,i,x,h,w)[0], GRID_D, 1.0, bg=0)
check_scalar_eq("lowermost", lambda n,i,x,h,w: lowermost(n,i,x,h,w)[0], GRID_D, 2.0, bg=0)
check_scalar_eq("leftmost",  lambda n,i,x,h,w: leftmost(n,i,x,h,w)[0],  GRID_D, 1.0, bg=0)
check_scalar_eq("rightmost", lambda n,i,x,h,w: rightmost(n,i,x,h,w)[0], GRID_D, 2.0, bg=0)

check_vec_eq("center", lambda n,i,x,h,w: center(n,i,x,h,w)[0], GRID_D, [1, 1], bg=0)
check_vec_eq("centerofmass", lambda n,i,x,h,w: centerofmass(n,i,x,h,w)[0], GRID_D, [1.5, 1.5], bg=0)

exp_delta = np.zeros((5, 5), dtype=int)
check_mask_eq("delta", lambda n,i,x,h,w: delta(n,i,x,h,w)[0], GRID_D, exp_delta, bg=0)

exp_backdrop = np.zeros((5, 5), dtype=int); exp_backdrop[1:3, 1:3] = 1
check_mask_eq("backdrop", lambda n,i,x,h,w: backdrop(n,i,x,h,w)[0], GRID_D, exp_backdrop, bg=0)
check_mask_eq("box", lambda n,i,x,h,w: box(n,i,x,h,w)[0], GRID_D, exp_backdrop, bg=0)  # 2x2 bbox: all border

exp_inbox = np.zeros((5, 5), dtype=int)
check_mask_eq("inbox", lambda n,i,x,h,w: inbox(n,i,x,h,w)[0], GRID_D, exp_inbox, bg=0)

exp_outbox = np.zeros((5, 5), dtype=int); exp_outbox[0:4, 0:4] = 1
check_mask_eq("outbox", lambda n,i,x,h,w: outbox(n,i,x,h,w)[0], GRID_D, exp_outbox, bg=0)

exp_subgrid = np.zeros((5, 5), dtype=int); exp_subgrid[0:2, 0:2] = 3
check_grid_eq("subgrid", lambda n,i,x,h,w: subgrid(n,i,x,x,h,w)[0], GRID_D, exp_subgrid, bg=0)


print("=" * 78)
print("SECTION E — color statistics (8 primitives, RAW encoding)")
print("=" * 78)

GRID_E = [[5, 5, 5],
          [5, 1, 1],
          [2, 2, 0]]

check_scalar_eq("mostcolor",   lambda n,i,x,h,w: mostcolor(n,i,x,h,w)[0],   GRID_E, 5.0)
check_scalar_eq("leastcolor",  lambda n,i,x,h,w: leastcolor(n,i,x,h,w)[0],  GRID_E, 0.0)
check_scalar_eq("mostcommon",  lambda n,i,x,h,w: mostcommon(n,i,x,h,w)[0],  GRID_E, 5.0)
check_scalar_eq("leastcommon", lambda n,i,x,h,w: leastcommon(n,i,x,h,w)[0], GRID_E, 0.0)
check_scalar_eq("colorcount(5)", lambda n,i,x,h,w: colorcount(n,i,x,h,w,color=5)[0], GRID_E, 4.0)
check_scalar_eq("numcolors",   lambda n,i,x,h,w: numcolors(n,i,x,h,w)[0],   GRID_E, 4.0)
check_scalar_eq("color (alias of mostcolor)", lambda n,i,x,h,w: color(n,i,x,h,w)[0], GRID_E, 5.0)

def test_palette():
    def _t():
        x, h, w = encode_grid(GRID_E)
        r = run(lambda n,i,xx,hh,ww: palette(n,i,xx,hh,ww)[0], x, h, w, (1, 1, 1, 10))
        got = r.reshape(-1).tolist()
        exp = [1.0, 1.0, 1.0, 0, 0, 1.0, 0, 0, 0, 0]  # colors present: 0,1,2,5
        return got == exp, got
    T("palette", _t)
test_palette()


print("=" * 78)
print("SECTION F — pixel/color algebra (20 primitives)")
print("=" * 78)

GRID_F = [[0, 1, 1],
          [0, 1, 0],
          [2, 2, 0]]

check_mask_eq("ofcolor(1)", lambda n,i,x,h,w: ofcolor(n,i,x,h,w,color=1)[0], GRID_F,
              [[0, 1, 1], [0, 1, 0], [0, 0, 0]])

def test_canvas():
    def _t():
        r = run_const(lambda n, i: canvas(n, i, color=4, h=2, w=3)[0], (1, NUM_COLORS, 2, 3))
        got = decode_grid(r)
        exp = np.full((2, 3), 4)
        return np.array_equal(got, exp), got
    T("canvas", _t)
test_canvas()

check_grid_eq("asobject (identity)", lambda n,i,x,h,w: asobject(n,i,x,h,w)[0], GRID_F, GRID_F)

def test_asindices():
    def _t():
        x, h, w = encode_grid(GRID_F)
        r = run(lambda n,i,xx,hh,ww: asindices(n,i,xx,hh,ww)[0], x, h, w, (1, 1, h, w))
        got = decode_mask(r)
        return np.array_equal(got, np.ones((h, w), dtype=np.int64)), got
    T("asindices", _t)
test_asindices()

def _toindices_pipeline(n, i, x, h, w):
    mask, _, _ = ofcolor(n, i, x, h, w, color=1)
    obj, _, _ = toobject(n, i, mask, x, h, w)
    return toindices(n, i, obj, h, w)[0]

check_mask_eq("toindices", _toindices_pipeline, GRID_F,
              [[0, 1, 1], [0, 1, 0], [0, 0, 0]])

def _toobject_pipeline(n, i, x, h, w):
    mask, _, _ = ofcolor(n, i, x, h, w, color=1)
    return toobject(n, i, mask, x, h, w)[0]

exp_toobject = np.where(np.array(GRID_F) == 1, 1, 0)
check_grid_eq("toobject", _toobject_pipeline, GRID_F, exp_toobject) # bg=0 coincides w/ true bg

def test_toivec():
    def _t():
        r = run_const(lambda n, i: toivec(n, i, scalar_const(i, 'v', 3.0), 1, 1)[0], (1, 1, 1, 2))
        return decode_vec(r) == [3.0, 0.0], decode_vec(r)
    T("toivec", _t)

def test_tojvec():
    def _t():
        r = run_const(lambda n, i: tojvec(n, i, scalar_const(i, 'v', 4.0), 1, 1)[0], (1, 1, 1, 2))
        return decode_vec(r) == [0.0, 4.0], decode_vec(r)
    T("tojvec", _t)

test_toivec()
test_tojvec()

def _fill_pipeline(n, i, x, h, w):
    mask, _, _ = ofcolor(n, i, x, h, w, color=1)
    return fill(n, i, x, h, w, color=9, mask=mask)[0]

check_grid_eq("fill", _fill_pipeline, GRID_F, np.where(np.array(GRID_F) == 1, 9, GRID_F))
check_grid_eq("replace", lambda n,i,x,h,w: replace(n,i,x,h,w,old_color=1,new_color=9)[0],
              GRID_F, np.where(np.array(GRID_F) == 1, 9, GRID_F))

def _recolor_pipeline(n, i, x, h, w):
    mask, _, _ = ofcolor(n, i, x, h, w, color=2)
    return recolor(n, i, 7, mask, h, w)[0]

check_grid_eq("recolor", _recolor_pipeline, GRID_F,
              [[0, 0, 0], [0, 0, 0], [7, 7, 0]])

def _paint_pipeline(n, i, x, h, w):
    mask, _, _ = ofcolor(n, i, x, h, w, color=2)
    obj, _, _ = recolor(n, i, 7, mask, h, w)
    return paint(n, i, x, obj, h, w)[0]

check_grid_eq("paint", _paint_pipeline, GRID_F,
              [[0, 1, 1], [0, 1, 0], [7, 7, 0]])

def _cover_pipeline(n, i, x, h, w):
    mask, _, _ = ofcolor(n, i, x, h, w, color=1)
    return cover(n, i, x, mask, h, w)[0]

check_grid_eq("cover", _cover_pipeline, GRID_F,
              [[0, 0, 0], [0, 0, 0], [2, 2, 0]])

def _underfill_pipeline(n, i, x, h, w):
    full, _, _ = asindices(n, i, x, h, w)
    return underfill(n, i, x, h, w, color=6, mask=full)[0]

check_grid_eq("underfill", _underfill_pipeline, GRID_F,
              [[6, 1, 1], [6, 1, 6], [2, 2, 6]])

def _underpaint_pipeline(n, i, x, h, w):
    mask, _, _ = ofcolor(n, i, x, h, w, color=2)
    obj, _, _ = recolor(n, i, 7, mask, h, w)
    return underpaint(n, i, x, obj, h, w)[0]

check_grid_eq("underpaint (no-op: target cells not bg)", _underpaint_pipeline, GRID_F, GRID_F)

check_grid_eq("switch(0,1)", lambda n,i,x,h,w: switch(n,i,x,h,w,a=0,b=1)[0], GRID_F,
              [[1, 0, 0], [1, 0, 1], [2, 2, 1]])

def _merge_2grid_pipeline(n, i, x, h, w):
    m, _, _ = vmirror(n, i, x, h, w)
    return merge(n, i, x, m, h, w)[0]

exp_merge = np.maximum(np.array(GRID_F) * (np.array(GRID_F) > 0).astype(int),
                        np.fliplr(GRID_F) * (np.fliplr(GRID_F) > 0).astype(int))
check_grid_eq("merge (2-grid form)", _merge_2grid_pipeline, GRID_F, GRID_F)  # identical to vmirror for this grid

def test_combine():
    def _t():
        GRID_G = [[0, 0, 0, 0], [0, 5, 0, 0], [0, 0, 0, 6], [0, 0, 6, 6]]
        x, h, w = encode_grid(GRID_G, bg=0)
        nodes, inits = [], []
        objs, _, _ = objects(nodes, inits, "input", h, w, without_bg=True, K=8)
        a, _, _ = colorfilter(nodes, inits, objs, h, w, color=5)
        b, _, _ = colorfilter(nodes, inits, objs, h, w, color=6)
        c, _, _ = combine(nodes, inits, a, b, h, w)
        merged, _, _ = merge(nodes, inits, c, None, h, w)
        model = make_model(nodes, inits, in_shape=(1, NUM_COLORS, h, w), out_name=merged, out_shape=(1, NUM_COLORS, h, w))
        sess = ort.InferenceSession(model.SerializeToString())
        r = sess.run(["output"], {"input": x})[0]
        got = decode_grid(r)
        return np.array_equal(got, np.array(GRID_G)), got
    T("combine (OBJSET concat)", _t)
test_combine()

def test_difference_intersection():
    def _t():
        nodes, inits = [], []
        a = _init(inits, "da", np.array([[[[1, 1, 0], [0, 1, 0]]]], dtype=np.float32))
        b = _init(inits, "db", np.array([[[[1, 0, 0], [0, 0, 1]]]], dtype=np.float32))
        diff, _, _ = difference(nodes, inits, a, b, 2, 3)
        model = make_model(nodes, inits, in_shape=(1, NUM_COLORS, 1, 1), out_name=diff, out_shape=(1, 1, 2, 3))
        dummy = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
        sess = ort.InferenceSession(model.SerializeToString())
        got = decode_mask(sess.run(["output"], {"input": dummy})[0])
        exp = np.array([[0, 1, 0], [0, 1, 0]])
        return np.array_equal(got, exp), got
    T("difference", _t)

    def _t2():
        nodes, inits = [], []
        a = _init(inits, "ia", np.array([[[[1, 1, 0], [0, 1, 0]]]], dtype=np.float32))
        b = _init(inits, "ib", np.array([[[[1, 0, 0], [0, 0, 1]]]], dtype=np.float32))
        inter, _, _ = intersection(nodes, inits, a, b, 2, 3)
        model = make_model(nodes, inits, in_shape=(1, NUM_COLORS, 1, 1), out_name=inter, out_shape=(1, 1, 2, 3))
        dummy = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
        sess = ort.InferenceSession(model.SerializeToString())
        got = decode_mask(sess.run(["output"], {"input": dummy})[0])
        exp = np.array([[1, 0, 0], [0, 0, 0]])
        return np.array_equal(got, exp), got
    T("intersection", _t2)

test_difference_intersection()


print("=" * 78)
print("SECTION G — connected components (3 primitives)")
print("=" * 78)

GRID_G = [[0, 0, 0, 0],
          [0, 5, 0, 0],
          [0, 0, 0, 6],
          [0, 0, 6, 6]]

def _objects_merge(n, i, x, h, w):
    objs, _, _ = objects(n, i, x, h, w, univalued=False, diagonal=False, without_bg=True, K=8)
    return merge(n, i, objs, None, h, w)[0]

check_grid_eq("objects()+merge() round-trip", _objects_merge, GRID_G, GRID_G)

def _partition_merge(n, i, x, h, w):
    p, _, _ = partition(n, i, x, h, w, without_bg=False)
    return merge(n, i, p, None, h, w)[0]

check_grid_eq("partition()+merge() round-trip", _partition_merge, GRID_G, GRID_G)

def _fgpartition_merge(n, i, x, h, w):
    p, _, _ = fgpartition(n, i, x, h, w)
    return merge(n, i, p, None, h, w)[0]

check_grid_eq("fgpartition()+merge() round-trip", _fgpartition_merge, GRID_G, GRID_G)


print("=" * 78)
print("SECTION H — object-set operations (20 primitives)")
print("=" * 78)

exp_6 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 6], [0, 0, 6, 6]])
exp_5 = np.array([[0, 0, 0, 0], [0, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

def _objs(n, i, x, h, w):
    return objects(n, i, x, h, w, without_bg=True, K=8)

def _colorfilter_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    f, _, _ = colorfilter(n, i, objs, h, w, color=6)
    return merge(n, i, f, None, h, w)[0]

check_grid_eq("colorfilter", _colorfilter_pipeline, GRID_G, exp_6)

def _sizefilter_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    f, _, _ = sizefilter(n, i, objs, h, w, n=1)
    return merge(n, i, f, None, h, w)[0]

check_grid_eq("sizefilter", _sizefilter_pipeline, GRID_G, exp_5)

def _color_eq6_pred(n, i, slot, h, w, prefix):
    c, _, _ = color(n, i, slot, h, w, prefix=prefix)
    return equality(n, i, c, scalar_const(i, prefix, 6.0), 1, 1, prefix)

def _sfilter_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    f, _, _ = sfilter(n, i, objs, h, w, _color_eq6_pred)
    return merge(n, i, f, None, h, w)[0]

check_grid_eq("sfilter", _sfilter_pipeline, GRID_G, exp_6)

def _mfilter_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    return mfilter(n, i, objs, h, w, _color_eq6_pred)[0]

check_grid_eq("mfilter", _mfilter_pipeline, GRID_G, exp_6)

def _order_by_size(n, i, slot, h, w, prefix):
    return size(n, i, slot, h, w, prefix=prefix)

def _first_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    ordered, _, _ = order(n, i, objs, h, w, _order_by_size)
    return first(n, i, ordered, h, w)[0]

check_grid_eq("first (after order-by-size)", _first_pipeline, GRID_G, exp_5)

def _last_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    ordered, _, _ = order(n, i, objs, h, w, _order_by_size)
    return last(n, i, ordered, h, w)[0]

check_grid_eq("last (after order-by-size)", _last_pipeline, GRID_G, exp_6)

def _extract_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    return extract(n, i, objs, h, w, _color_eq6_pred)[0]

check_grid_eq("extract", _extract_pipeline, GRID_G, exp_6)

def _other_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    six, _, _ = extract(n, i, objs, h, w, _color_eq6_pred)
    return other(n, i, objs, six, h, w)[0]

check_grid_eq("other", _other_pipeline, GRID_G, exp_5)

def _argmax_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    return argmax(n, i, objs, h, w, _order_by_size)[0]

check_grid_eq("argmax", _argmax_pipeline, GRID_G, exp_6)

def _argmin_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    return argmin(n, i, objs, h, w, _order_by_size)[0]

check_grid_eq("argmin", _argmin_pipeline, GRID_G, exp_5)

check_scalar_eq("valmax", lambda n,i,x,h,w: valmax(n, i, _objs(n,i,x,h,w)[0], h, w, _order_by_size),
                 GRID_G, 3.0, bg=None)
check_scalar_eq("valmin", lambda n,i,x,h,w: valmin(n, i, _objs(n,i,x,h,w)[0], h, w, _order_by_size),
                 GRID_G, 1.0, bg=None)

def _order_sanity_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    ordered, _, _ = order(n, i, objs, h, w, _order_by_size)
    return merge(n, i, ordered, None, h, w)[0]

check_grid_eq("order (content-preserving)", _order_sanity_pipeline, GRID_G, GRID_G)

def _remove_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    six, _, _ = extract(n, i, objs, h, w, _color_eq6_pred)
    remain, _, _ = remove(n, i, objs, six, h, w)
    return merge(n, i, remain, None, h, w)[0]

check_grid_eq("remove", _remove_pipeline, GRID_G, exp_5)

def _insert_pipeline(n, i, x, h, w):
    objs, _, _ = _objs(n, i, x, h, w)
    newobj = _init(i, "ins_obj", encode_grid([[7, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], bg=0)[0])
    upd, _, _ = insert(n, i, objs, newobj, h, w)
    return merge(n, i, upd, None, h, w)[0]

exp_insert = np.array(GRID_G).copy(); exp_insert[0, 0] = 7
check_grid_eq("insert", _insert_pipeline, GRID_G, exp_insert)

def test_contained():
    def _t():
        x, h, w = encode_grid(GRID_G)
        nodes, inits = [], []
        pal, _, _ = palette(nodes, inits, "input", h, w)
        v_in = scalar_const(inits, "cin", 6.0)
        v_out = scalar_const(inits, "cout", 9.0)
        r_in, _, _ = contained(nodes, inits, v_in, pal)
        r_out, _, _ = contained(nodes, inits, v_out, pal)
        cat = _op(nodes, "Concat", [r_in, r_out], "catcont", axis=3)
        model = make_model(nodes, inits, in_shape=(1, NUM_COLORS, h, w), out_name=cat, out_shape=(1, 1, 1, 2))
        sess = ort.InferenceSession(model.SerializeToString())
        got = sess.run(["output"], {"input": x})[0].reshape(-1).tolist()
        return got == [1.0, 0.0], got
    T("contained", _t)
test_contained()

def test_dedupe():
    got = dedupe((1, 2, 2, 3, 1))
    T("dedupe", lambda: (got == (1, 2, 3), got))

test_dedupe()

def test_normalize_and_compress():
    # stage 1: extract the standalone '6' object as a plain grid
    x, h, w = encode_grid(GRID_G, bg=0)
    nodes, inits = [], []
    objs, _, _ = objects(nodes, inits, "input", h, w, without_bg=True, K=8)
    six, _, _ = extract(nodes, inits, objs, h, w, _color_eq6_pred)
    model = make_model(nodes, inits, in_shape=(1, NUM_COLORS, h, w), out_name=six, out_shape=(1, NUM_COLORS, h, w))
    sess = ort.InferenceSession(model.SerializeToString())
    six_grid = decode_grid(sess.run(["output"], {"input": x})[0])

    exp = np.zeros((4, 4), dtype=int); exp[0, 1] = 6; exp[1, 0] = 6; exp[1, 1] = 6

    def _t_norm():
        xo, ho, wo = encode_grid(six_grid.tolist(), bg=0)
        r = run(lambda n,i,xx,hh,ww: normalize(n,i,xx,hh,ww)[0], xo, ho, wo, (1, NUM_COLORS, ho, wo))
        got = decode_grid(r)
        return np.array_equal(got, exp), got
    T("normalize", _t_norm)

    def _t_compress():
        xo, ho, wo = encode_grid(six_grid.tolist(), bg=0)
        r = run(lambda n,i,xx,hh,ww: compress(n,i,xx,hh,ww)[0], xo, ho, wo, (1, NUM_COLORS, ho, wo))
        got = decode_grid(r)
        return np.array_equal(got, exp), got
    T("compress", _t_compress)

test_normalize_and_compress()

def test_index():
    def _t():
        x, h, w = encode_grid(GRID_G)
        nodes, inits = [], []
        r_row = scalar_const(inits, "ir", 2.0)
        c_col = scalar_const(inits, "ic", 3.0)
        out, _, _ = index(nodes, inits, "input", h, w, r_row, c_col)
        model = make_model(nodes, inits, in_shape=(1, NUM_COLORS, h, w), out_name=out, out_shape=(1, 1, 1, 1))
        sess = ort.InferenceSession(model.SerializeToString())
        got = decode_scalar(sess.run(["output"], {"input": x})[0])
        return abs(got - 6.0) < 1e-4, got
    T("index", _t)
test_index()


print("=" * 78)
print("SECTION I — lines / frontiers / periods / neighbors / occurrences (15)")
print("=" * 78)

GRID_HLINE = [[0, 0, 0], [5, 5, 5], [0, 0, 0]]
GRID_VLINE = [[0, 5, 0], [0, 5, 0], [0, 5, 0]]

check_scalar_eq("hline", lambda n,i,x,h,w: hline(n,i,x,h,w)[0], GRID_HLINE, 1.0, bg=0)
check_scalar_eq("vline", lambda n,i,x,h,w: vline(n,i,x,h,w)[0], GRID_VLINE, 1.0, bg=0)

def test_hfrontier():
    def _t():
        nodes, inits = [], []
        row = scalar_const(inits, "row", 1.0)
        out, h, w = hfrontier(nodes, inits, row, 3, 3)
        r = run_const_model(nodes, inits, out, (1, 1, 3, 3))
        got = decode_mask(r)
        exp = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        return np.array_equal(got, exp), got
    T("hfrontier", _t)

def test_vfrontier():
    def _t():
        nodes, inits = [], []
        col = scalar_const(inits, "col", 1.0)
        out, h, w = vfrontier(nodes, inits, col, 3, 3)
        r = run_const_model(nodes, inits, out, (1, 1, 3, 3))
        got = decode_mask(r)
        exp = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        return np.array_equal(got, exp), got
    T("vfrontier", _t)

test_hfrontier()
test_vfrontier()

def test_frontiers():
    def _t():
        GRID_FR = [[1, 1, 1], [2, 3, 2], [1, 1, 1]]
        x, h, w = encode_grid(GRID_FR)
        nodes, inits = [], []
        result, _, _ = frontiers(nodes, inits, "input", h, w)
        full_row_c1 = result[1][0]
        f = _op(nodes, "Cast", [full_row_c1], "frc", to=1)  # FLOAT
        out = _op(nodes, "Reshape", [f, _init(inits, "frs", np.array([1, 1, 1, h], dtype=np.int64))], "frr")
        model = make_model(nodes, inits, in_shape=(1, NUM_COLORS, h, w), out_name=out, out_shape=(1, 1, 1, h))
        sess = ort.InferenceSession(model.SerializeToString())
        got = sess.run(["output"], {"input": x})[0].reshape(-1).tolist()
        return got == [1.0, 0.0, 1.0], got
    T("frontiers", _t)
test_frontiers()

def test_connect():
    def _t():
        r = run_const(lambda n, i: connect(n, i, (0, 0), (2, 2), 3, 3)[0], (1, 1, 3, 3))
        got = decode_mask(r)
        exp = np.eye(3, dtype=int)
        return np.array_equal(got, exp), got
    T("connect", _t)

def test_shoot():
    def _t():
        r = run_const(lambda n, i: shoot(n, i, (1, 1), (1, 0), 3, 3)[0], (1, 1, 3, 3))
        got = decode_mask(r)
        exp = np.zeros((3, 3), dtype=int); exp[1, 1] = 1; exp[2, 1] = 1
        return np.array_equal(got, exp), got
    T("shoot", _t)

def test_neighbors():
    def _t():
        r = run_const(lambda n, i: neighbors(n, i, (1, 1), 3, 3)[0], (1, 1, 3, 3))
        got = decode_mask(r)
        exp = np.ones((3, 3), dtype=int); exp[1, 1] = 0
        return np.array_equal(got, exp), got
    T("neighbors", _t)

def test_dneighbors():
    def _t():
        r = run_const(lambda n, i: dneighbors(n, i, (1, 1), 3, 3)[0], (1, 1, 3, 3))
        got = decode_mask(r)
        exp = np.zeros((3, 3), dtype=int)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            exp[1 + di, 1 + dj] = 1
        return np.array_equal(got, exp), got
    T("dneighbors", _t)

test_connect(); test_shoot(); test_neighbors(); test_dneighbors()

GRID_VP = [[1, 2], [3, 4], [1, 2], [3, 4]]
GRID_HP = [[1, 2, 1, 2], [3, 4, 3, 4]]

check_scalar_eq("vperiod", lambda n,i,x,h,w: vperiod(n,i,x,h,w)[0], GRID_VP, 2.0)
check_scalar_eq("hperiod", lambda n,i,x,h,w: hperiod(n,i,x,h,w)[0], GRID_HP, 2.0)

def test_occurrences():
    def _t():
        GRID_OCC = [[1, 2, 0], [3, 4, 0], [0, 0, 5]]
        OBJ_OCC  = [[1, 2, 9], [3, 4, 9], [9, 9, 9]]  # top-left 2x2 = pattern, rest irrelevant
        x, h, w = encode_grid(GRID_OCC)
        nodes, inits = [], []
        obj_name = _init(inits, "occ_obj", encode_grid(OBJ_OCC)[0])
        out, _, _ = occurrences(nodes, inits, "input", obj_name, h, w, 2, 2)
        model = make_model(nodes, inits, in_shape=(1, NUM_COLORS, h, w), out_name=out, out_shape=(1, 1, h, w))
        sess = ort.InferenceSession(model.SerializeToString())
        got = decode_mask(sess.run(["output"], {"input": x})[0])
        exp = np.zeros((3, 3), dtype=int); exp[0, 0] = 1
        return np.array_equal(got, exp), got
    T("occurrences", _t)
test_occurrences()

def test_gravitate():
    def _t():
        h, w = 5, 5
        src, _, _ = encode_grid([[0]*5, [0]*5, [0, 0, 7, 0, 0], [0]*5, [0]*5], bg=0)
        dst, _, _ = encode_grid([[0]*5, [0]*5, [0, 0, 0, 0, 7], [0]*5, [0]*5], bg=0)
        nodes, inits = [], []
        src_name = _init(inits, "gsrc", src)
        dst_name = _init(inits, "gdst", dst)
        out, _, _ = gravitate(nodes, inits, src_name, dst_name, h, w, max_steps=1)
        r = run_const_model(nodes, inits, out, (1, NUM_COLORS, h, w))
        got = decode_grid(r)
        exp = np.zeros((5, 5), dtype=int); exp[2, 3] = 7
        return np.array_equal(got, exp), got
    T("gravitate (max_steps=1 unit move)", _t)
test_gravitate()

def test_position():
    def _t():
        a, h, w = encode_grid([[0,0,0,0],[0,9,0,0],[0,0,0,0]], bg=0)
        nodes, inits = [], []
        a_name = _init(inits, "pa", a)
        b_name = _init(inits, "pb", encode_grid([[0,0,0,0],[0,0,0,9],[0,0,0,0]], bg=0)[0])
        out, _, _ = position(nodes, inits, a_name, b_name, 3, 4)
        r = run_const_model(nodes, inits, out, (1, 1, 1, 2))
        got = decode_vec(r)
        return got == [0.0, 1.0], got
    T("position", _t)
test_position()

def test_vmatching():
    def _t():
        nodes, inits = [], []
        a = _init(inits, "vma", encode_grid([[0,9,9,0],[0,0,0,0]], bg=0)[0])
        b = _init(inits, "vmb", encode_grid([[0,0,9,9],[0,0,0,0]], bg=0)[0])
        out, _, _ = vmatching(nodes, inits, a, b, 2, 4)
        r = run_const_model(nodes, inits, out, (1, 1, 1, 1))
        return abs(decode_scalar(r) - 1.0) < 1e-4, decode_scalar(r)
    T("vmatching", _t)
test_vmatching()


print("=" * 78)
print("SECTION J — combinators (15 primitives, tested via concrete pipelines)")
print("=" * 78)

check_grid_eq("compose(hmirror,vmirror)",
              lambda n,i,x,h,w: compose(hmirror, vmirror)(n,i,x,h,w)[0],
              GRID_B, ref_hmirror(ref_vmirror(GRID_B)))

check_grid_eq("chain(vmirror,hmirror,tophalf)",
              lambda n,i,x,h,w: chain(vmirror, hmirror, tophalf)(n,i,x,h,w)[0],
              GRID_B, ref_vmirror(ref_hmirror(np.array(GRID_B)[:2])))

def _ref_fork_merge(grid):
    a = encode_grid(ref_hmirror(grid).tolist())[0]
    b = encode_grid(ref_vmirror(grid).tolist())[0]
    return decode_grid(np.maximum(a, b))

check_grid_eq("fork(merge, hmirror, vmirror)",
              lambda n,i,x,h,w: fork(merge, hmirror, vmirror)(n,i,x,h,w)[0],
              GRID_B, _ref_fork_merge(np.array(GRID_B)))

def test_lbind_rbind():
    def _t1():
        r = run_const(lambda n, i: lbind(add, scalar_const(i, 'a', 3.0))(n, i, scalar_const(i, 'b', 4.0), 1, 1)[0], (1,1,1,1))
        return abs(decode_scalar(r) - 7.0) < 1e-4, decode_scalar(r)
    T("lbind(add,3)(4)", _t1)

    def _t2():
        r = run_const(lambda n, i: rbind(subtract, scalar_const(i, 'b', 4.0))(n, i, scalar_const(i, 'a', 10.0), 1, 1)[0], (1,1,1,1))
        return abs(decode_scalar(r) - 6.0) < 1e-4, decode_scalar(r)
    T("rbind(subtract,4)(10)", _t2)

test_lbind_rbind()

check_grid_eq("branch(True,...)", lambda n,i,x,h,w: branch(True, hmirror, vmirror)(n,i,x,h,w)[0],
              GRID_B, ref_hmirror(GRID_B))
check_grid_eq("branch(False,...)", lambda n,i,x,h,w: branch(False, hmirror, vmirror)(n,i,x,h,w)[0],
              GRID_B, ref_vmirror(GRID_B))

check_scalar_eq("matcher(height,2)", lambda n,i,x,h,w: matcher(height, scalar_const(i, 'tg', 2.0))(n,i,x,h,w)[0],
                 GRID_D, 1.0, bg=0)

def _apply_pipeline(n, i, x, h, w):
    objs, _, _ = objects(n, i, x, h, w, without_bg=True, K=8)
    applied, ah, aw = apply(hmirror, objs)(n, i, h, w)
    return merge(n, i, applied, None, ah, aw)[0]

check_grid_eq("apply(hmirror, objects)", _apply_pipeline, GRID_G, ref_hmirror(GRID_G))

def _mapply_pipeline(n, i, x, h, w):
    objs, _, _ = objects(n, i, x, h, w, without_bg=True, K=8)
    return mapply(hmirror, objs)(n, i, h, w)[0]

check_grid_eq("mapply(hmirror, objects)", _mapply_pipeline, GRID_G, ref_hmirror(GRID_G))

def _mpapply_pipeline(n, i, x, h, w):
    objs, _, _ = objects(n, i, x, h, w, without_bg=True, K=8)
    return mpapply(lambda nn, ii, a, b, hh, ww, prefix='m': merge(nn, ii, a, b, hh, ww, prefix=prefix), objs, objs)(n, i, h, w)[0]

check_grid_eq("mpapply(merge, objs, objs) idempotent", _mpapply_pipeline, GRID_G, GRID_G)

def _prapply_pipeline(n, i, x, h, w):
    objs, _, _ = objects(n, i, x, h, w, without_bg=True, K=8)
    a, _, _ = colorfilter(n, i, objs, h, w, color=5)
    b, _, _ = colorfilter(n, i, objs, h, w, color=6)
    prod, ph, pw = prapply(lambda nn, ii, x1, x2, hh, ww, prefix='pf': merge(nn, ii, x1, x2, hh, ww, prefix=prefix), a, b)(n, i, h, w)
    return merge(n, i, prod, None, ph, pw)[0]

check_grid_eq("prapply(merge, colorfilter5, colorfilter6)", _prapply_pipeline, GRID_G, GRID_G)

def test_papply():
    def _t():
        x, h, w = encode_grid(GRID_G, bg=0)
        nodes, inits = [], []
        objs, _, _ = objects(nodes, inits, "input", h, w, without_bg=True, K=8)
        p, ph, pw = papply(lambda nn, ii, a, b, hh, ww, prefix='pp': merge(nn, ii, a, b, hh, ww, prefix=prefix), objs, objs)(nodes, inits, h, w)
        merged, _, _ = merge(nodes, inits, p, None, ph, pw)
        model = make_model(nodes, inits, in_shape=(1, NUM_COLORS, h, w), out_name=merged, out_shape=(1, NUM_COLORS, h, w))
        sess = ort.InferenceSession(model.SerializeToString())
        got = decode_grid(sess.run(["output"], {"input": x})[0])
        return np.array_equal(got, np.array(GRID_G)), got
    T("papply(merge, objs, objs) idempotent", _t)
test_papply()

def test_rapply():
    def _t():
        x, h, w = encode_grid(GRID_B)
        nodes, inits = [], []
        outs = rapply((hmirror, vmirror), "input")(nodes, inits, h, w)
        cat = _op(nodes, "Concat", list(outs), "rapc", axis=1)  # stack along channel just to fetch both in one run
        model = make_model(nodes, inits, in_shape=(1, NUM_COLORS, h, w), out_name=cat, out_shape=(1, 2 * NUM_COLORS, h, w))
        sess = ort.InferenceSession(model.SerializeToString())
        r = sess.run(["output"], {"input": x})[0]
        got_h = decode_grid(r[:, :NUM_COLORS])
        got_v = decode_grid(r[:, NUM_COLORS:])
        ok = np.array_equal(got_h, ref_hmirror(GRID_B)) and np.array_equal(got_v, ref_vmirror(GRID_B))
        return ok, (got_h, got_v)
    T("rapply((hmirror,vmirror), x)", _t)
test_rapply()

check_grid_eq("power(hmirror, 2) == identity", lambda n,i,x,h,w: power(hmirror, 2)(n,i,x,h,w)[0], GRID_B, GRID_B)

def test_repeat():
    got = repeat(5, 3)
    T("repeat", lambda: (got == (5, 5, 5), got))
test_repeat()


print("=" * 78)
print("SECTION K — pure-python containers (7 primitives)")
print("=" * 78)

T("astuple", lambda: (astuple(1, 2) == (1, 2), astuple(1, 2)))
T("pair", lambda: (pair((1, 2), (3, 4)) == ((1, 3), (2, 4)), pair((1, 2), (3, 4))))
T("initset", lambda: (initset(5) == (5,), initset(5)))
T("totuple", lambda: (totuple([1, 2, 3]) == (1, 2, 3), totuple([1, 2, 3])))
T("interval", lambda: (interval(0, 5, 1) == (0, 1, 2, 3, 4), interval(0, 5, 1)))
T("product", lambda: (product((1, 2), (3, 4)) == ((1, 3), (1, 4), (2, 3), (2, 4)), product((1, 2), (3, 4))))
T("combine_py", lambda: (combine_py((1, 2), (3, 4)) == (1, 2, 3, 4), combine_py((1, 2), (3, 4))))


# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 78)
n_pass = sum(1 for _, ok in REGISTRY if ok)
n_total = len(REGISTRY)
print(f"{n_pass}/{n_total} tests passed.")
if n_pass != n_total:
    print("\nFailing tests:")
    for name, ok in REGISTRY:
        if not ok:
            print(f"  - {name}")