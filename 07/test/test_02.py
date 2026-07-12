"""
test_arc_onnx_primitives.py
Tests every major primitive in arc_onnx_primitives.py using real grids.
Run: python test_arc_onnx_primitives.py
"""

import numpy as np
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto, numpy_helper
import traceback
import sys

# ── import the module under test ────────────────────────────────────────────
import arc_onnx_primitives as P
from arc_onnx_primitives import (
    _node, _init, _fresh, _op, scalar_const, vec2_const,
    NUM_COLORS, BIG
)

# ============================================================================
# Helpers
# ============================================================================

PASS = 0
FAIL = 0
RESULTS = []


def color_grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    """
    grid: (h, w) int array with values 0-9
    returns: (1, 10, h, w) float32 one-hot
    """
    h, w = grid.shape
    out = np.zeros((1, NUM_COLORS, h, w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            out[0, grid[r, c], r, c] = 1.0
    return out


def onehot_to_color_grid(onehot: np.ndarray) -> np.ndarray:
    """
    onehot: (1, 10, h, w) float32
    returns: (h, w) int array
    """
    return np.argmax(onehot[0], axis=0).astype(np.int32)


def mask_to_bool_grid(mask: np.ndarray) -> np.ndarray:
    """
    mask: (1, 1, h, w) float32
    returns: (h, w) bool
    """
    return (mask[0, 0] > 0.5).astype(np.int32)


def run_model(model, input_array: np.ndarray) -> np.ndarray:
    """Run an ONNX model and return the output array."""
    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: input_array})
    return out[0]


def build_and_run(builder_fn, input_grid_color: np.ndarray):
    """
    builder_fn(nodes, inits, x_name, h, w) -> (out_name, oh, ow)
    input_grid_color: (h, w) int array
    Returns raw output numpy array.
    """
    h, w = input_grid_color.shape
    inp = color_grid_to_onehot(input_grid_color)

    nodes, inits = [], []
    out_name, oh, ow = builder_fn(nodes, inits, "input", h, w)

    # determine output shape from out_name usage
    # we'll let make_model figure it out via shape inference
    try:
        model = P.make_model(
            nodes, inits, inp.shape, out_name,
            [1, NUM_COLORS, oh, ow]
        )
        return run_model(model, inp), oh, ow
    except Exception:
        # try (1,1,oh,ow) shape for masks/scalars
        raise


def build_and_run_mask(builder_fn, input_grid_color: np.ndarray):
    """For primitives that output (1,1,h,w) masks."""
    h, w = input_grid_color.shape
    inp = color_grid_to_onehot(input_grid_color)
    nodes, inits = [], []
    out_name, oh, ow = builder_fn(nodes, inits, "input", h, w)
    model = P.make_model(nodes, inits, inp.shape, out_name, [1, 1, oh, ow])
    return run_model(model, inp), oh, ow


def build_and_run_scalar(builder_fn, input_grid_color: np.ndarray):
    """For primitives that output (1,1,1,1) scalars."""
    h, w = input_grid_color.shape
    inp = color_grid_to_onehot(input_grid_color)
    nodes, inits = [], []
    out_name, oh, ow = builder_fn(nodes, inits, "input", h, w)
    model = P.make_model(nodes, inits, inp.shape, out_name, [1, 1, 1, 1])
    return run_model(model, inp)


def check(name, got, expected, tol=0):
    global PASS, FAIL
    got = np.array(got)
    expected = np.array(expected)
    if got.shape != expected.shape:
        ok = False
        reason = f"shape mismatch got={got.shape} expected={expected.shape}"
    elif tol == 0:
        ok = np.array_equal(got, expected)
        reason = f"\nGOT:\n{got}\nEXPECTED:\n{expected}" if not ok else ""
    else:
        ok = np.allclose(got, expected, atol=tol)
        reason = f"\nGOT:\n{got}\nEXPECTED:\n{expected}" if not ok else ""

    sym = "✓ PASS" if ok else "✗ FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    RESULTS.append((name, ok, reason))
    print(f"  {sym}  {name}{reason}")
    return ok


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def safe_test(name, fn):
    """Wrap a test so one failure doesn't crash the suite."""
    try:
        fn()
    except Exception as e:
        global FAIL
        FAIL += 1
        RESULTS.append((name, False, str(e)))
        print(f"  ✗ FAIL  {name}  [EXCEPTION: {e}]")
        traceback.print_exc()


# ============================================================================
# Test Grids (color indices 0-9)
# ============================================================================

# 4x4 grid: checkerboard of colors 1 and 2
CHECKER = np.array([
    [1, 2, 1, 2],
    [2, 1, 2, 1],
    [1, 2, 1, 2],
    [2, 1, 2, 1],
], dtype=np.int32)

# 4x6 grid with a red (color=2) rectangle in the middle
RECT_GRID = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 0],
    [0, 2, 2, 2, 2, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=np.int32)

# 5x5 mostly black (0) with a 2x2 blue (1) block at top-left
BLUE_BLOCK = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)

# 3x3 solid color-3 grid
SOLID3 = np.full((3, 3), 3, dtype=np.int32)

# 6x6 grid with diagonal color 5
DIAG6 = np.zeros((6, 6), dtype=np.int32)
for i in range(6):
    DIAG6[i, i] = 5

# 4x4 grid with single color 7 in center
CENTER_DOT = np.array([
    [0, 0, 0, 0],
    [0, 7, 7, 0],
    [0, 7, 7, 0],
    [0, 0, 0, 0],
], dtype=np.int32)

# 6x6 grid: repeated 2x2 tile (period=2)
PERIODIC = np.array([
    [1, 2, 1, 2, 1, 2],
    [3, 4, 3, 4, 3, 4],
    [1, 2, 1, 2, 1, 2],
    [3, 4, 3, 4, 3, 4],
    [1, 2, 1, 2, 1, 2],
    [3, 4, 3, 4, 3, 4],
], dtype=np.int32)

# ============================================================================
# SECTION A — Scalar / Elementwise Arithmetic
# ============================================================================

section("A — Scalar / Elementwise Arithmetic")


def test_increment():
    # increment a scalar by 3
    nodes, inits = [], []
    c = P.scalar_const(inits, "test", 5.0)
    out, _, _ = P.increment(nodes, inits, c, 1, 1, prefix="inc", delta=3)
    model = P.make_model(nodes, inits, [1, NUM_COLORS, 1, 1], out, [1, 1, 1, 1])
    dummy = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    result = run_model(model, dummy)
    check("increment(5, delta=3) == 8", result.flat[0], 8.0, tol=1e-4)


safe_test("increment", test_increment)


def test_decrement():
    nodes, inits = [], []
    c = P.scalar_const(inits, "test", 10.0)
    out, _, _ = P.decrement(nodes, inits, c, 1, 1, prefix="dec", delta=4)
    model = P.make_model(nodes, inits, [1, NUM_COLORS, 1, 1], out, [1, 1, 1, 1])
    dummy = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    result = run_model(model, dummy)
    check("decrement(10, delta=4) == 6", result.flat[0], 6.0, tol=1e-4)


safe_test("decrement", test_decrement)


def test_double_halve():
    nodes, inits = [], []
    c = P.scalar_const(inits, "t", 6.0)
    d, _, _ = P.double(nodes, inits, c, 1, 1, prefix="d")
    h2, _, _ = P.halve(nodes, inits, d, 1, 1, prefix="h")
    model = P.make_model(nodes, inits, [1, NUM_COLORS, 1, 1], h2, [1, 1, 1, 1])
    dummy = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    result = run_model(model, dummy)
    check("halve(double(6)) == 6", result.flat[0], 6.0, tol=1e-4)


safe_test("double_halve", test_double_halve)


def test_negate():
    nodes, inits = [], []
    c = P.scalar_const(inits, "t", 3.0)
    out, _, _ = P.negate(nodes, inits, c, 1, 1, prefix="n")
    model = P.make_model(nodes, inits, [1, NUM_COLORS, 1, 1], out, [1, 1, 1, 1])
    dummy = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    result = run_model(model, dummy)
    check("negate(3) == -3", result.flat[0], -3.0, tol=1e-4)


safe_test("negate", test_negate)


def test_equality_true():
    nodes, inits = [], []
    a = P.scalar_const(inits, "a", 5.0)
    b = P.scalar_const(inits, "b", 5.0)
    out, _, _ = P.equality(nodes, inits, a, b, 1, 1, prefix="eq")
    model = P.make_model(nodes, inits, [1, NUM_COLORS, 1, 1], out, [1, 1, 1, 1])
    dummy = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    result = run_model(model, dummy)
    check("equality(5, 5) == 1.0", result.flat[0], 1.0, tol=1e-4)


safe_test("equality_true", test_equality_true)


def test_equality_false():
    nodes, inits = [], []
    a = P.scalar_const(inits, "a", 3.0)
    b = P.scalar_const(inits, "b", 7.0)
    out, _, _ = P.equality(nodes, inits, a, b, 1, 1, prefix="eq")
    model = P.make_model(nodes, inits, [1, NUM_COLORS, 1, 1], out, [1, 1, 1, 1])
    dummy = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    result = run_model(model, dummy)
    check("equality(3, 7) == 0.0", result.flat[0], 0.0, tol=1e-4)


safe_test("equality_false", test_equality_false)


def test_even():
    nodes, inits = [], []
    c = P.scalar_const(inits, "t", 4.0)
    out, _, _ = P.even(nodes, inits, c, 1, 1, prefix="ev")
    model = P.make_model(nodes, inits, [1, NUM_COLORS, 1, 1], out, [1, 1, 1, 1])
    dummy = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    result = run_model(model, dummy)
    check("even(4) == 1.0", result.flat[0], 1.0, tol=1e-4)

    nodes2, inits2 = [], []
    c2 = P.scalar_const(inits2, "t", 3.0)
    out2, _, _ = P.even(nodes2, inits2, c2, 1, 1, prefix="ev2")
    model2 = P.make_model(nodes2, inits2, [1, NUM_COLORS, 1, 1], out2, [1, 1, 1, 1])
    result2 = run_model(model2, dummy)
    check("even(3) == 0.0", result2.flat[0], 0.0, tol=1e-4)


safe_test("even", test_even)


def test_flip():
    nodes, inits = [], []
    c = P.scalar_const(inits, "t", 1.0)
    out, _, _ = P.flip(nodes, inits, c, 1, 1, prefix="fl")
    model = P.make_model(nodes, inits, [1, NUM_COLORS, 1, 1], out, [1, 1, 1, 1])
    dummy = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    result = run_model(model, dummy)
    check("flip(1.0) == 0.0", result.flat[0], 0.0, tol=1e-4)


safe_test("flip", test_flip)

# ============================================================================
# SECTION B — Geometric Transforms
# ============================================================================

section("B — Geometric Transforms")

# Input:      Expected hmirror (row-reversed):
# 1 2 1 2     2 1 2 1
# 2 1 2 1     1 2 1 2
# 1 2 1 2     2 1 2 1
# 2 1 2 1     1 2 1 2


def test_hmirror():
    def builder(nodes, inits, x, h, w):
        return P.hmirror(nodes, inits, x, h, w, prefix="hm")

    raw, oh, ow = build_and_run(builder, CHECKER)
    got = onehot_to_color_grid(raw)
    expected = CHECKER[::-1, :]   # reverse rows
    check("hmirror reverses rows", got, expected)


safe_test("hmirror", test_hmirror)


def test_vmirror():
    def builder(nodes, inits, x, h, w):
        return P.vmirror(nodes, inits, x, h, w, prefix="vm")

    raw, oh, ow = build_and_run(builder, CHECKER)
    got = onehot_to_color_grid(raw)
    expected = CHECKER[:, ::-1]   # reverse cols
    check("vmirror reverses cols", got, expected)


safe_test("vmirror", test_vmirror)


def test_rot180():
    def builder(nodes, inits, x, h, w):
        return P.rot180(nodes, inits, x, h, w, prefix="r180")

    raw, oh, ow = build_and_run(builder, CHECKER)
    got = onehot_to_color_grid(raw)
    expected = CHECKER[::-1, ::-1]
    check("rot180 == 180° rotation", got, expected)


safe_test("rot180", test_rot180)


def test_dmirror():
    # 3x3 asymmetric grid
    grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return P.dmirror(nodes, inits, x, h, w, prefix="dm")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    expected = grid.T   # transpose
    check("dmirror == transpose", got, expected)


safe_test("dmirror", test_dmirror)


def test_rot90():
    grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return P.rot90(nodes, inits, x, h, w, prefix="r90")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    # rot90 CW: row i, col j -> row j, col (h-1-i)
    expected = np.rot90(grid, k=-1)   # numpy rot90 k=1 is CCW; k=-1 is CW
    check("rot90 (CW)", got, expected)


safe_test("rot90", test_rot90)


def test_tophalf():
    def builder(nodes, inits, x, h, w):
        return P.tophalf(nodes, inits, x, h, w, prefix="th")

    raw, oh, ow = build_and_run(builder, CHECKER)
    got = onehot_to_color_grid(raw)
    check("tophalf rows==2", got, CHECKER[:2, :])


safe_test("tophalf", test_tophalf)


def test_bottomhalf():
    def builder(nodes, inits, x, h, w):
        return P.bottomhalf(nodes, inits, x, h, w, prefix="bh")

    raw, oh, ow = build_and_run(builder, CHECKER)
    got = onehot_to_color_grid(raw)
    check("bottomhalf rows 2..4", got, CHECKER[2:, :])


safe_test("bottomhalf", test_bottomhalf)


def test_lefthalf():
    def builder(nodes, inits, x, h, w):
        return P.lefthalf(nodes, inits, x, h, w, prefix="lh")

    raw, oh, ow = build_and_run(builder, CHECKER)
    got = onehot_to_color_grid(raw)
    check("lefthalf cols 0..2", got, CHECKER[:, :2])


safe_test("lefthalf", test_lefthalf)


def test_upscale():
    grid = np.array([[1, 2], [3, 4]], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return P.upscale(nodes, inits, x, h, w, factor=2, prefix="up")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    expected = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.int32)
    check("upscale 2x2 by factor 2 => 4x4", got, expected)


safe_test("upscale", test_upscale)


def test_shift():
    grid = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return P.shift(nodes, inits, x, h, w, di=1, dj=1, prefix="sh")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    expected = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=np.int32)
    check("shift(di=1,dj=1)", got, expected)


safe_test("shift", test_shift)


def test_crop():
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0],
        [0, 3, 3, 3, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return P.crop(nodes, inits, x, h, w,
                      top=1, left=1, height=2, width=3,
                      prefix="cr")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    expected = np.array([
        [3, 3, 3],
        [3, 3, 3],
    ], dtype=np.int32)
    check("crop(top=1,left=1,h=2,w=3)", got, expected)


safe_test("crop", test_crop)


def test_hupscale():
    grid = np.array([[1, 2], [3, 4]], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return P.hupscale(nodes, inits, x, h, w, factor=3, prefix="hu")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    expected = np.array([
        [1, 1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4, 4],
    ], dtype=np.int32)
    check("hupscale factor=3", got, expected)


safe_test("hupscale", test_hupscale)


def test_vupscale():
    grid = np.array([[1, 2], [3, 4]], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return P.vupscale(nodes, inits, x, h, w, factor=3, prefix="vu")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    expected = np.array([
        [1, 2],
        [1, 2],
        [1, 2],
        [3, 4],
        [3, 4],
        [3, 4],
    ], dtype=np.int32)
    check("vupscale factor=3", got, expected)


safe_test("vupscale", test_vupscale)

# ============================================================================
# SECTION C — Bounding Box
# ============================================================================

section("C — Bounding Box")


def test_height_scalar():
    def builder(nodes, inits, x, h, w):
        return P.height(nodes, inits, x, h, w, prefix="ht")

    result = build_and_run_scalar(builder, RECT_GRID)
    # RECT_GRID has color-2 rows at rows 1..2 => height=2
    check("height(rect_4x6) == 2", result.flat[0], 2.0, tol=1e-3)


safe_test("height_scalar", test_height_scalar)


def test_width_scalar():
    def builder(nodes, inits, x, h, w):
        return P.width(nodes, inits, x, h, w, prefix="wd")

    result = build_and_run_scalar(builder, RECT_GRID)
    check("width(rect_4x6) == 4", result.flat[0], 4.0, tol=1e-3)


safe_test("width_scalar", test_width_scalar)


def test_size_scalar():
    def builder(nodes, inits, x, h, w):
        return P.size(nodes, inits, x, h, w, prefix="sz")

    # RECT_GRID: 4x6=24 cells, all filled (full one-hot)
    result = build_and_run_scalar(builder, RECT_GRID)
    # size sums occupancy over all cells
    check("size(rect_4x6) == 24", result.flat[0], 24.0, tol=1e-3)


safe_test("size_scalar", test_size_scalar)


def test_ulcorner():
    h, w = RECT_GRID.shape
    inp = color_grid_to_onehot(RECT_GRID)
    nodes, inits = [], []
    out, oh, ow = P.ulcorner(nodes, inits, "input", h, w, prefix="ulc")
    model = P.make_model(nodes, inits, inp.shape, out, [1, 1, 1, 2])
    result = run_model(model, inp)
    # ulcorner of the full grid = (0,0) since every cell is occupied
    check("ulcorner(rect_grid) == [0, 0]",
          result.flatten()[:2], [0.0, 0.0], tol=1e-3)


safe_test("ulcorner", test_ulcorner)


def test_uppermost():
    def builder(nodes, inits, x, h, w):
        return P.uppermost(nodes, inits, x, h, w, prefix="upm")

    result = build_and_run_scalar(builder, BLUE_BLOCK)
    # BLUE_BLOCK: color-1 is at rows 0..1; background(0) fills rest
    # uppermost of the overall occupied area = row 0
    check("uppermost(blue_block) == 0", result.flat[0], 0.0, tol=1e-3)


safe_test("uppermost", test_uppermost)


def test_backdrop():
    # backdrop of BLUE_BLOCK should be bbox rows 0..1, cols 0..1
    def builder(nodes, inits, x, h, w):
        return P.backdrop(nodes, inits, x, h, w, prefix="bd")

    raw, oh, ow = build_and_run_mask(builder, BLUE_BLOCK)
    got = mask_to_bool_grid(raw)
    # whole grid is covered by bbox since bg(0) fills remainder
    # Actually: BLUE_BLOCK most-color=0 (fills 21 cells), bbox of whole = (0..4,0..4)
    # so backdrop = all 1s
    check("backdrop shape", got.shape, BLUE_BLOCK.shape)
    check("backdrop all ones (full coverage)", int(got.sum()), 25)


safe_test("backdrop", test_backdrop)


def test_backdrop_rect():
    # Use only-nonzero occupancy: ofcolor-2 on RECT_GRID
    h, w = RECT_GRID.shape
    inp = color_grid_to_onehot(RECT_GRID)
    nodes, inits = [], []
    # extract color-2 mask
    ofc, _, _ = P.ofcolor(nodes, inits, "input", h, w, color=2, prefix="ofc")
    # backdrop of that mask — recolor to a full grid first
    bd, _, _ = P.backdrop(nodes, inits, ofc, h, w, depth=1, prefix="bd")
    model = P.make_model(nodes, inits, inp.shape, bd, [1, 1, h, w])
    raw = run_model(model, inp)
    got = mask_to_bool_grid(raw)
    expected = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.int32)
    check("backdrop(color2 rect)", got, expected)


safe_test("backdrop_rect", test_backdrop_rect)

# ============================================================================
# SECTION D — Color Statistics
# ============================================================================

section("D — Color Statistics")


def test_mostcolor():
    # grid with mostly color 0
    grid = np.array([
        [0, 0, 0, 1],
        [0, 0, 2, 1],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return P.mostcolor(nodes, inits, x, h, w, prefix="mc")

    result = build_and_run_scalar(builder, grid)
    check("mostcolor == 0", result.flat[0], 0.0, tol=1e-3)


safe_test("mostcolor", test_mostcolor)


def test_leastcolor():
    grid = np.array([
        [0, 0, 0, 1],
        [0, 0, 2, 1],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return P.leastcolor(nodes, inits, x, h, w, prefix="lc")

    result = build_and_run_scalar(builder, grid)
    # color 2 appears once, color 1 appears twice, color 0 appears most
    check("leastcolor == 2", result.flat[0], 2.0, tol=1e-3)


safe_test("leastcolor", test_leastcolor)


def test_numcolors():
    grid = np.array([
        [1, 2, 3],
        [1, 2, 4],
        [5, 6, 7],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return P.numcolors(nodes, inits, x, h, w, prefix="nc")

    h, w = grid.shape
    inp = color_grid_to_onehot(grid)
    nodes, inits = [], []
    out, oh, ow = builder(nodes, inits, "input", h, w)
    model = P.make_model(nodes, inits, inp.shape, out, [1])
    result = run_model(model, inp)
    check("numcolors(7 distinct) == 7", result.flat[0], 7.0, tol=1e-3)


safe_test("numcolors", test_numcolors)


def test_colorcount():
    def builder(nodes, inits, x, h, w):
        return P.colorcount(nodes, inits, x, h, w, color=2, prefix="cc")

    result = build_and_run_scalar(builder, RECT_GRID)
    # RECT_GRID: color 2 appears in rows 1..2, cols 1..4 => 8 cells
    check("colorcount(rect, color=2) == 8", result.flat[0], 8.0, tol=1e-3)


safe_test("colorcount", test_colorcount)

# ============================================================================
# SECTION E — Pixel/Color Algebra
# ============================================================================

section("E — Pixel / Color Algebra")


def test_ofcolor():
    def builder(nodes, inits, x, h, w):
        return P.ofcolor(nodes, inits, x, h, w, color=2, prefix="ofc")

    raw, oh, ow = build_and_run_mask(builder, RECT_GRID)
    got = mask_to_bool_grid(raw)
    expected = (RECT_GRID == 2).astype(np.int32)
    check("ofcolor(rect, 2)", got, expected)


safe_test("ofcolor", test_ofcolor)


def test_canvas():
    nodes, inits = [], []
    out, oh, ow = P.canvas(nodes, inits, color=3, h=2, w=3, prefix="cv")
    model = P.make_model(nodes, inits, [1, NUM_COLORS, 1, 1], out, [1, NUM_COLORS, 2, 3])
    dummy = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    result = run_model(model, dummy)
    got = onehot_to_color_grid(result)
    expected = np.full((2, 3), 3, dtype=np.int32)
    check("canvas(color=3, 2x3)", got, expected)


safe_test("canvas", test_canvas)


def test_fill():
    grid = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.int32)
    h, w = grid.shape
    inp = color_grid_to_onehot(grid)
    nodes, inits = [], []

    # mask: fill the top-left 2x2
    mask_arr = np.zeros((1, 1, h, w), dtype=np.float32)
    mask_arr[0, 0, :2, :2] = 1.0
    mask_name = _init(inits, "mask", mask_arr)

    out, oh, ow = P.fill(nodes, inits, "input", h, w,
                         color=5, mask=mask_name, prefix="fl")
    model = P.make_model(nodes, inits, inp.shape, out, [1, NUM_COLORS, h, w])
    result = run_model(model, inp)
    got = onehot_to_color_grid(result)
    expected = np.array([
        [5, 5, 0],
        [5, 5, 0],
        [0, 0, 0],
    ], dtype=np.int32)
    check("fill(color=5, top-left 2x2)", got, expected)


safe_test("fill", test_fill)


def test_replace():
    grid = np.array([
        [1, 1, 2],
        [1, 3, 2],
        [2, 2, 2],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return P.replace(nodes, inits, x, h, w,
                         old_color=2, new_color=5, prefix="rpl")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    expected = np.where(grid == 2, 5, grid).astype(np.int32)
    check("replace(2->5)", got, expected)


safe_test("replace", test_replace)


def test_paint():
    # paint a 3x3 red (2) square onto a blue (1) background
    background = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=np.int32)
    obj = np.array([
        [0, 0, 0, 0],
        [0, 2, 2, 0],
        [0, 2, 2, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    h, w = background.shape
    bg_inp = color_grid_to_onehot(background)
    obj_inp = color_grid_to_onehot(obj)

    nodes, inits = [], []
    # store obj as a constant initializer
    obj_name = _init(inits, "obj_const", obj_inp)
    out, oh, ow = P.paint(nodes, inits, "input", obj_name, h, w, prefix="pnt")
    model = P.make_model(nodes, inits, bg_inp.shape, out, [1, NUM_COLORS, h, w])
    result = run_model(model, bg_inp)
    got = onehot_to_color_grid(result)

    expected = np.array([
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        [1, 1, 1, 1],
    ], dtype=np.int32)
    check("paint(blue_bg, red_square)", got, expected)


safe_test("paint", test_paint)


def test_switch():
    grid = np.array([
        [1, 2, 1],
        [2, 1, 2],
        [1, 2, 1],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return P.switch(nodes, inits, x, h, w, a=1, b=2, prefix="sw")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    # swap colors 1 and 2
    expected = np.where(grid == 1, 2, np.where(grid == 2, 1, grid)).astype(np.int32)
    check("switch(1<->2)", got, expected)


safe_test("switch", test_switch)

# ============================================================================
# SECTION F — Merge / Cover / Intersection / Difference
# ============================================================================

section("F — Merge / Cover / Set Ops")


def test_merge_two_grids():
    a = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=np.int32)
    b = np.array([
        [0, 0, 2],
        [0, 0, 2],
        [0, 0, 0],
    ], dtype=np.int32)

    h, w = a.shape
    a_inp = color_grid_to_onehot(a)
    b_inp = color_grid_to_onehot(b)

    nodes, inits = [], []
    b_name = _init(inits, "b_const", b_inp)
    out, oh, ow = P.merge(nodes, inits, "input", b_name, h=h, w=w, prefix="mrg")
    model = P.make_model(nodes, inits, a_inp.shape, out, [1, NUM_COLORS, h, w])
    result = run_model(model, a_inp)
    got = onehot_to_color_grid(result)

    expected = np.array([
        [1, 0, 2],
        [1, 0, 2],
        [0, 0, 0],
    ], dtype=np.int32)
    check("merge(a, b)", got, expected)


safe_test("merge_two_grids", test_merge_two_grids)


def test_cover():
    bg = np.array([
        [0, 0, 0, 0],
        [0, 3, 3, 0],
        [0, 3, 3, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    h, w = bg.shape
    inp = color_grid_to_onehot(bg)

    # obj to erase: color 3 region
    obj = np.array([
        [0, 0, 0, 0],
        [0, 3, 3, 0],
        [0, 3, 3, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    obj_name = _init([], "obj_c", color_grid_to_onehot(obj))

    nodes, inits = [], []
    obj_n = _init(inits, "obj_c2", color_grid_to_onehot(obj))
    out, oh, ow = P.cover(nodes, inits, "input", obj_n, h, w, prefix="cov")
    model = P.make_model(nodes, inits, inp.shape, out, [1, NUM_COLORS, h, w])
    result = run_model(model, inp)
    got = onehot_to_color_grid(result)

    # after cover, the 3s become 0 (background=0 = mostcolor)
    expected = np.zeros((4, 4), dtype=np.int32)
    check("cover erases object to bg", got, expected)


safe_test("cover", test_cover)


def test_intersection():
    a_mask = np.zeros((1, 1, 4, 4), dtype=np.float32)
    b_mask = np.zeros((1, 1, 4, 4), dtype=np.float32)
    a_mask[0, 0, :2, :2] = 1.0   # top-left 2x2
    b_mask[0, 0, 1:3, 1:3] = 1.0  # center 2x2

    nodes, inits = [], []
    a_n = _init(inits, "a", a_mask)
    b_n = _init(inits, "b", b_mask)
    out, _, _ = P.intersection(nodes, inits, a_n, b_n, 4, 4, prefix="isc")
    model = P.make_model(nodes, inits, [1, 1, 4, 4], out, [1, 1, 4, 4])
    dummy = np.zeros((1, 1, 4, 4), dtype=np.float32)
    result = run_model(model, dummy)
    got = (result[0, 0] > 0.5).astype(np.int32)

    expected = np.zeros((4, 4), dtype=np.int32)
    expected[1, 1] = 1  # only overlap cell
    check("intersection(top-left, center)", got, expected)


safe_test("intersection", test_intersection)


def test_difference():
    a_mask = np.zeros((1, 1, 3, 3), dtype=np.float32)
    b_mask = np.zeros((1, 1, 3, 3), dtype=np.float32)
    a_mask[0, 0, :, :] = 1.0   # all ones
    b_mask[0, 0, 0, :] = 1.0   # top row

    nodes, inits = [], []
    a_n = _init(inits, "a", a_mask)
    b_n = _init(inits, "b", b_mask)
    out, _, _ = P.difference(nodes, inits, a_n, b_n, 3, 3, prefix="dif")
    model = P.make_model(nodes, inits, [1, 1, 3, 3], out, [1, 1, 3, 3])
    dummy = np.zeros((1, 1, 3, 3), dtype=np.float32)
    result = run_model(model, dummy)
    got = (result[0, 0] > 0.5).astype(np.int32)

    expected = np.ones((3, 3), dtype=np.int32)
    expected[0, :] = 0   # top row removed
    check("difference(all, top-row)", got, expected)


safe_test("difference", test_difference)

# ============================================================================
# SECTION G — Partition / Objects
# ============================================================================

section("G — Partition / Objects")


def test_partition():
    grid = np.array([
        [1, 1, 0],
        [2, 2, 0],
        [3, 3, 0],
    ], dtype=np.int32)
    h, w = grid.shape
    inp = color_grid_to_onehot(grid)
    nodes, inits = [], []
    objset, _, _ = P.partition(nodes, inits, "input", h, w, prefix="prt")
    _, stack, valid, K = objset

    # we want to merge the partition and verify we recover the input
    out, oh, ow = P.merge(nodes, inits, objset, None, h, w, prefix="mrg")
    model = P.make_model(nodes, inits, inp.shape, out, [1, NUM_COLORS, h, w])
    result = run_model(model, inp)
    got = onehot_to_color_grid(result)
    check("partition→merge recovers input", got, grid)


safe_test("partition", test_partition)


def test_fgpartition_excludes_bg():
    grid = np.array([
        [0, 1, 0],
        [0, 2, 0],
        [0, 0, 0],
    ], dtype=np.int32)
    h, w = grid.shape
    inp = color_grid_to_onehot(grid)
    nodes, inits = [], []
    objset, _, _ = P.fgpartition(nodes, inits, "input", h, w, prefix="fgp")

    # merge the fg partition; background cells should remain 0
    out, oh, ow = P.merge(nodes, inits, objset, None, h, w, prefix="mrg")
    model = P.make_model(nodes, inits, inp.shape, out, [1, NUM_COLORS, h, w])
    result = run_model(model, inp)
    got = onehot_to_color_grid(result)
    # background cells => 0 from merge (reduce-max over valid slots, bg slot invalid)
    check("fgpartition excludes bg", got, grid)


safe_test("fgpartition_excludes_bg", test_fgpartition_excludes_bg)

# ============================================================================
# SECTION H — Object-Set Filters
# ============================================================================

section("H — Object-Set Filters")


def test_colorfilter():
    grid = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], dtype=np.int32)
    h, w = grid.shape
    inp = color_grid_to_onehot(grid)
    nodes, inits = [], []
    objset, _, _ = P.partition(nodes, inits, "input", h, w, prefix="prt")
    filtered, _, _ = P.colorfilter(nodes, inits, objset, h, w,
                                    color=1, prefix="cf")
    out, oh, ow = P.merge(nodes, inits, filtered, None, h, w, prefix="mrg")
    model = P.make_model(nodes, inits, inp.shape, out, [1, NUM_COLORS, h, w])
    result = run_model(model, inp)
    got = onehot_to_color_grid(result)
    # only color-1 cells remain; everything else becomes 0
    expected = np.where(grid == 1, 1, 0).astype(np.int32)
    check("colorfilter(color=1)", got, expected)


safe_test("colorfilter", test_colorfilter)


def test_sizefilter():
    # grid: 4 cells of color 1, 1 cell of color 2
    grid = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    h, w = grid.shape
    inp = color_grid_to_onehot(grid)
    nodes, inits = [], []
    objset, _, _ = P.partition(nodes, inits, "input", h, w, prefix="prt")
    filtered, _, _ = P.sizefilter(nodes, inits, objset, h, w, n=4, prefix="szf")
    out, oh, ow = P.merge(nodes, inits, filtered, None, h, w, prefix="mrg")
    model = P.make_model(nodes, inits, inp.shape, out, [1, NUM_COLORS, h, w])
    result = run_model(model, inp)
    got = onehot_to_color_grid(result)
    expected = np.where(grid == 1, 1, 0).astype(np.int32)
    check("sizefilter(n=4) keeps color-1 block", got, expected)


safe_test("sizefilter", test_sizefilter)

# ============================================================================
# SECTION I — Periods
# ============================================================================

section("I — Periods")


def test_hperiod():
    def builder(nodes, inits, x, h, w):
        return P.hperiod(nodes, inits, x, h, w, prefix="hp")

    result = build_and_run_scalar(builder, PERIODIC)
    check("hperiod(periodic_6x6) == 2", result.flat[0], 2.0, tol=1e-3)


safe_test("hperiod", test_hperiod)


def test_vperiod():
    def builder(nodes, inits, x, h, w):
        return P.vperiod(nodes, inits, x, h, w, prefix="vp")

    result = build_and_run_scalar(builder, PERIODIC)
    check("vperiod(periodic_6x6) == 2", result.flat[0], 2.0, tol=1e-3)


safe_test("vperiod", test_vperiod)

# ============================================================================
# SECTION J — Occurrences
# ============================================================================

section("J — Occurrences")


def test_occurrences():
    # Grid:
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    # Pattern to find: [[1, 2]]
    pattern = np.array([[1, 2]], dtype=np.int32)

    h, w = grid.shape
    oh, ow = pattern.shape
    grid_inp = color_grid_to_onehot(grid)
    pat_inp = color_grid_to_onehot(pattern)

    nodes, inits = [], []
    pat_name = _init(inits, "pat", pat_inp)
    out, _, _ = P.occurrences(nodes, inits, "input", pat_name, h, w, oh, ow, prefix="occ")
    model = P.make_model(nodes, inits, grid_inp.shape, out, [1, 1, h, w])
    result = run_model(model, grid_inp)
    got = (result[0, 0] > 0.5).astype(np.int32)

    expected = np.zeros((h, w), dtype=np.int32)
    expected[1, 1] = 1   # row=1, col=1
    expected[3, 2] = 1   # row=3, col=2
    check("occurrences finds [1,2] at (1,1) and (3,2)", got, expected)


safe_test("occurrences", test_occurrences)

# ============================================================================
# SECTION K — Combinators
# ============================================================================

section("K — Combinators")


def test_compose():
    # compose(hmirror, vmirror) = hmirror(vmirror(x))
    fn = P.compose(
        lambda n, i, x, h, w, **kw: P.hmirror(n, i, x, h, w, **kw),
        lambda n, i, x, h, w, **kw: P.vmirror(n, i, x, h, w, **kw),
    )
    grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return fn(nodes, inits, x, h, w, prefix="cmp")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    expected = grid[::-1, ::-1]   # hmirror(vmirror) = rot180
    check("compose(hmirror, vmirror) == rot180", got, expected)


safe_test("compose", test_compose)


def test_fork():
    # fork(merge, identity, rot180)(x) = merge(x, rot180(x))
    identity_fn = lambda n, i, x, h, w, **kw: (
        _op(n, "Identity", [x], "id"), h, w
    )
    rot180_fn = lambda n, i, x, h, w, **kw: P.rot180(n, i, x, h, w, **kw)
    merge_fn = lambda n, i, a, b, h, w, **kw: P.merge(n, i, a, b, h=h, w=w, **kw)

    fn = P.fork(merge_fn, identity_fn, rot180_fn)

    grid = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 2],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return fn(nodes, inits, x, h, w, prefix="frk")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    # merge(grid, rot180(grid)) via Max on one-hot
    rotated = grid[::-1, ::-1]
    # at each cell: take max channel between grid and rotated (both are one-hot)
    expected = np.where(grid != 0, grid, rotated)
    check("fork(merge, id, rot180)", got, expected)


safe_test("fork", test_fork)


def test_power():
    # power(rot90, 4)(x) should = identity
    fn = P.power(
        lambda n, i, x, h, w, **kw: P.rot90(n, i, x, h, w, **kw),
        4
    )
    grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return fn(nodes, inits, x, h, w, prefix="pw")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    check("power(rot90, 4) == identity", got, grid)


safe_test("power", test_power)


# In test_rbind, change the lambda to NOT pass prefix=:
def test_rbind():
    fn = P.rbind(
        lambda n, i, x, b, h, w, **kw: P.replace(n, i, x, h, w,
                                                    old_color=1,
                                                    new_color=b,
                                                    **kw),   # NO prefix= here
        9
    )
    grid = np.array([
        [1, 2, 1],
        [2, 1, 2],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        return fn(nodes, inits, x, h, w, prefix="rb")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    expected = np.where(grid == 1, 9, grid).astype(np.int32)
    check("rbind(replace, 9)", got, expected)


safe_test("rbind", test_rbind)

# ============================================================================
# SECTION L — Full Pipeline Tests (input grid → expected output grid)
# ============================================================================

section("L — Full Pipeline Tests")


def test_pipeline_color_swap_and_rotate():
    """
    Pipeline: rot90(switch(1,2)(x))
    Input:                Expected output:
    1 2 1    switch→    1 2 1   (already symmetric)
    2 1 2               2 1 2   (unchanged by switch on this grid)
    1 2 1               1 2 1

    With rot90 (CW) applied after switch:
    """
    grid = np.array([
        [1, 2, 1],
        [2, 1, 2],
        [1, 2, 1],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        s, sh, sw = P.switch(nodes, inits, x, h, w, a=1, b=2, prefix="sw")
        return P.rot90(nodes, inits, s, sh, sw, prefix="r90")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    switched = np.where(grid == 1, 2, np.where(grid == 2, 1, grid))
    expected = np.rot90(switched, k=-1)
    check("switch(1,2) then rot90", got, expected)


safe_test("pipeline_color_swap_rotate", test_pipeline_color_swap_and_rotate)


def test_pipeline_upscale_then_hmirror():
    """
    Pipeline: hmirror(upscale(x, 2))
    Input 2x2 → upscale 4x4 → hmirror 4x4
    """
    grid = np.array([
        [1, 2],
        [3, 4],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        u, uh, uw = P.upscale(nodes, inits, x, h, w, factor=2, prefix="up")
        return P.hmirror(nodes, inits, u, uh, uw, prefix="hm")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    upscaled = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ])
    expected = upscaled[::-1, :]
    check("upscale(2) then hmirror", got, expected)


safe_test("pipeline_upscale_hmirror", test_pipeline_upscale_then_hmirror)


def test_pipeline_partition_recolor_merge():
    """
    Pipeline: merge(fgpartition(x)) should equal x (preserving fg).
    Then recolor color-3 to color-7 in the partition, merge back.
    """
    grid = np.array([
        [3, 3, 0, 0],
        [3, 3, 0, 0],
        [0, 0, 5, 5],
        [0, 0, 5, 5],
    ], dtype=np.int32)
    h, w = grid.shape
    inp = color_grid_to_onehot(grid)

    nodes, inits = [], []
    objset, _, _ = P.fgpartition(nodes, inits, "input", h, w, prefix="fgp")
    _, stack, valid, K = objset

    # replace color 3 with color 7 in the stack
    stack_r, _, _ = P.replace(nodes, inits, stack, h, w,
                               old_color=3, new_color=7, prefix="rpl")
    new_objset = ("OBJSET", stack_r, valid, K)
    out, oh, ow = P.merge(nodes, inits, new_objset, None, h, w, prefix="mrg")
    model = P.make_model(nodes, inits, inp.shape, out, [1, NUM_COLORS, h, w])
    result = run_model(model, inp)
    got = onehot_to_color_grid(result)

    expected = np.where(grid == 3, 7, grid).astype(np.int32)
    check("fgpartition + recolor 3→7 + merge", got, expected)


safe_test("pipeline_partition_recolor_merge",
          test_pipeline_partition_recolor_merge)


def test_pipeline_shift_and_paint():
    """
    Pipeline: paint(grid, shift(obj, di=0, dj=2))
    Move a blue block right by 2 and paint onto grid.
    """
    background = np.zeros((4, 6), dtype=np.int32)
    obj = np.zeros((4, 6), dtype=np.int32)
    obj[1:3, 1:3] = 1   # 2x2 blue block at col 1..2

    h, w = background.shape
    bg_inp = color_grid_to_onehot(background)
    obj_inp = color_grid_to_onehot(obj)

    nodes, inits = [], []
    obj_name = _init(inits, "obj_c", obj_inp)
    shifted, sh, sw = P.shift(nodes, inits, obj_name, h, w,
                               di=0, dj=2, prefix="sh")
    out, oh, ow = P.paint(nodes, inits, "input", shifted, h, w, prefix="pnt")
    model = P.make_model(nodes, inits, bg_inp.shape, out, [1, NUM_COLORS, h, w])
    result = run_model(model, bg_inp)
    got = onehot_to_color_grid(result)

    expected = np.zeros((4, 6), dtype=np.int32)
    expected[1:3, 3:5] = 1   # shifted right by 2
    check("shift(obj, dj=2) then paint", got, expected)


safe_test("pipeline_shift_paint", test_pipeline_shift_and_paint)


def test_pipeline_crop_upscale():
    """
    Crop center 2x2 from 4x4, upscale by 2 → 4x4.
    """
    grid = np.array([
        [0, 0, 0, 0],
        [0, 5, 6, 0],
        [0, 7, 8, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    def builder(nodes, inits, x, h, w):
        c, ch, cw = P.crop(nodes, inits, x, h, w,
                            top=1, left=1, height=2, width=2,
                            prefix="cr")
        return P.upscale(nodes, inits, c, ch, cw, factor=2, prefix="up")

    raw, oh, ow = build_and_run(builder, grid)
    got = onehot_to_color_grid(raw)
    expected = np.array([
        [5, 5, 6, 6],
        [5, 5, 6, 6],
        [7, 7, 8, 8],
        [7, 7, 8, 8],
    ], dtype=np.int32)
    check("crop center 2x2, upscale 2x", got, expected)


safe_test("pipeline_crop_upscale", test_pipeline_crop_upscale)


def test_pipeline_objects_merge():
    """
    objects(x, univalued=True, without_bg=True) → merge should recover fg.
    """
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    h, w = grid.shape
    inp = color_grid_to_onehot(grid)
    nodes, inits = [], []
    objset, _, _ = P.objects(nodes, inits, "input", h, w,
                              univalued=True, diagonal=False,
                              without_bg=True, K=6, prefix="ob")
    out, oh, ow = P.merge(nodes, inits, objset, None, h, w, prefix="mrg")
    model = P.make_model(nodes, inits, inp.shape, out, [1, NUM_COLORS, h, w])
    result = run_model(model, inp)
    got = onehot_to_color_grid(result)
    check("objects→merge recovers fg", got, grid)


safe_test("pipeline_objects_merge", test_pipeline_objects_merge)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

section("SUMMARY")
total = PASS + FAIL
print(f"\n  Total : {total}")
print(f"  Passed: {PASS}  ({100*PASS//total if total else 0}%)")
print(f"  Failed: {FAIL}")

if FAIL:
    print("\n  Failed tests:")
    for name, ok, reason in RESULTS:
        if not ok:
            print(f"    - {name}: {reason[:120]}")

sys.exit(0 if FAIL == 0 else 1)