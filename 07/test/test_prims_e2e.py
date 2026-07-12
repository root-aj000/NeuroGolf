"""End-to-end tests for onnx_graph_prims.py

Each test defines:
  1. A concrete input grid (values 0-9, one-hot compatible)
  2. Expected output precomputed with numpy
  3. Builds ONNX graph, runs it, compares actual vs expected
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import onnx
import onnxruntime as ort
from onnx import TensorProto
import onnx_graph_prims as gp


# ============================================================================
# Helpers
# ============================================================================

def _encode(grid):
    h, w = grid.shape
    t = np.zeros((1, 10, h, w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            if 0 <= v < 10:
                t[0, v, r, c] = 1.0
    return t


def _decode(tensor):
    return np.argmax(tensor[0], axis=0).astype(np.int64)


def _build_model(nodes, inits, input_specs, out_h=30, out_w=30):
    graph = gp.helper.make_graph(
        nodes, "g",
        [gp.helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
         for name, shape in input_specs],
        [gp.helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                          [1, 10, out_h, out_w])],
        inits,
    )
    model = gp.helper.make_model(graph, opset_imports=[gp.helper.make_opsetid("", 17)])
    model.ir_version = 8
    return ort.InferenceSession(model.SerializeToString())


def _run_one(prim_fn, grid, prim_kwargs=None):
    if prim_kwargs is None:
        prim_kwargs = {}
    h, w = grid.shape
    nodes, inits = [], []
    out_name, oh, ow = prim_fn(nodes, inits, "input", h, w, prefix="t", **prim_kwargs)
    out_name, oh, ow = gp.pad_canvas(nodes, inits, out_name, oh, ow, prefix="pad")
    _n = gp.helper.make_node("Identity", [out_name], ["output"])
    nodes.append(_n)
    sess = _build_model(nodes, inits, [("input", [1, 10, h, w])])
    result = sess.run(None, {"input": _encode(grid)})[0]
    return _decode(result)


def _run_same_two(prim_fn, g1, g2, prim_kwargs=None):
    """For two-input prims that take (nodes, inits, x1, x2, h, w)."""
    if prim_kwargs is None:
        prim_kwargs = {}
    h1, w1 = g1.shape
    h2, w2 = g2.shape
    assert h1 == h2 and w1 == w2, "Same shape required for this helper"
    nodes, inits = [], []
    out_name, oh, ow = prim_fn(nodes, inits, "a", "b", h1, w1, prefix="t", **prim_kwargs)
    out_name, oh, ow = gp.pad_canvas(nodes, inits, out_name, oh, ow, prefix="pad")
    _n = gp.helper.make_node("Identity", [out_name], ["output"])
    nodes.append(_n)
    shape = [1, 10, h1, w1]
    sess = _build_model(nodes, inits, [("a", shape), ("b", shape)])
    result = sess.run(None, {"a": _encode(g1), "b": _encode(g2)})[0]
    return _decode(result)


def _run_fill(prim_fn, g1, g2, prim_kwargs=None):
    """For fill-style prims that take (nodes, inits, x, mask, h, w)."""
    if prim_kwargs is None:
        prim_kwargs = {}
    h, w = g1.shape
    nodes, inits = [], []
    out_name, oh, ow = prim_fn(nodes, inits, "x", "mask", h, w, prefix="t", **prim_kwargs)
    out_name, oh, ow = gp.pad_canvas(nodes, inits, out_name, oh, ow, prefix="pad")
    _n = gp.helper.make_node("Identity", [out_name], ["output"])
    nodes.append(_n)
    shape = [1, 10, h, w]
    sess = _build_model(nodes, inits, [("x", shape), ("mask", shape)])
    result = sess.run(None, {"x": _encode(g1), "mask": _encode(g2)})[0]
    return _decode(result)


def _run_vconcat(g1, g2):
    h1, w = g1.shape
    h2, _ = g2.shape
    nodes, inits = [], []
    out_name, oh, ow = gp.vconcat(nodes, inits, "a", "b", h1, h2, w, prefix="vc")
    out_name, oh, ow = gp.pad_canvas(nodes, inits, out_name, oh, ow, prefix="pad")
    _n = gp.helper.make_node("Identity", [out_name], ["output"])
    nodes.append(_n)
    sess = _build_model(nodes, inits,
                        [("a", [1, 10, h1, w]), ("b", [1, 10, h2, w])])
    result = sess.run(None, {"a": _encode(g1), "b": _encode(g2)})[0]
    return _decode(result)


def _run_hconcat(g1, g2):
    h, w1 = g1.shape
    _, w2 = g2.shape
    nodes, inits = [], []
    out_name, oh, ow = gp.hconcat(nodes, inits, "a", "b", h, w1, w2, prefix="hc")
    out_name, oh, ow = gp.pad_canvas(nodes, inits, out_name, oh, ow, prefix="pad")
    _n = gp.helper.make_node("Identity", [out_name], ["output"])
    nodes.append(_n)
    sess = _build_model(nodes, inits,
                        [("a", [1, 10, h, w1]), ("b", [1, 10, h, w2])])
    result = sess.run(None, {"a": _encode(g1), "b": _encode(g2)})[0]
    return _decode(result)


EQ = np.array_equal


# ============================================================================
# Input grids (all values 0-9)
# ============================================================================

GRID_3x3 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
], dtype=np.int64)

GRID_4x4 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 0, 1, 2],
    [3, 4, 5, 6],
], dtype=np.int64)

GRID_2x3 = np.array([
    [1, 2, 3],
    [4, 5, 6],
], dtype=np.int64)

GRID_3x2 = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
], dtype=np.int64)

GRID_4x3 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [0, 1, 2],
], dtype=np.int64)

GRID_2x4 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
], dtype=np.int64)

GRID_4x6 = np.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 0, 1, 2],
    [3, 4, 5, 6, 7, 8],
    [9, 0, 1, 2, 3, 4],
], dtype=np.int64)

GRID_6x4 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 0, 1, 2],
    [3, 4, 5, 6],
    [7, 8, 9, 0],
    [1, 2, 3, 4],
], dtype=np.int64)


# ============================================================================
# Rotations
# ============================================================================

class TestRot90:
    def test_3x3(self):
        assert EQ(_run_one(gp.rot90, GRID_3x3)[:3, :3], np.rot90(GRID_3x3, k=-1))

    def test_4x4(self):
        assert EQ(_run_one(gp.rot90, GRID_4x4)[:4, :4], np.rot90(GRID_4x4, k=-1))

    def test_rect(self):
        r = _run_one(gp.rot90, GRID_3x2)
        e = np.rot90(GRID_3x2, k=-1)
        assert EQ(r[:e.shape[0], :e.shape[1]], e)


class TestRot180:
    def test_3x3(self):
        assert EQ(_run_one(gp.rot180, GRID_3x3)[:3, :3], np.rot90(GRID_3x3, k=2))

    def test_4x4(self):
        assert EQ(_run_one(gp.rot180, GRID_4x4)[:4, :4], np.rot90(GRID_4x4, k=2))


class TestRot270:
    def test_3x3(self):
        assert EQ(_run_one(gp.rot270, GRID_3x3)[:3, :3], np.rot90(GRID_3x3, k=1))

    def test_4x4(self):
        assert EQ(_run_one(gp.rot270, GRID_4x4)[:4, :4], np.rot90(GRID_4x4, k=1))


# ============================================================================
# Mirrors
# ============================================================================

class TestHmirror:
    def test_3x3(self):
        assert EQ(_run_one(gp.hmirror, GRID_3x3)[:3, :3], np.flip(GRID_3x3, axis=0))

    def test_4x4(self):
        assert EQ(_run_one(gp.hmirror, GRID_4x4)[:4, :4], np.flip(GRID_4x4, axis=0))


class TestVmirror:
    def test_3x3(self):
        assert EQ(_run_one(gp.vmirror, GRID_3x3)[:3, :3], np.flip(GRID_3x3, axis=1))

    def test_4x4(self):
        assert EQ(_run_one(gp.vmirror, GRID_4x4)[:4, :4], np.flip(GRID_4x4, axis=1))


class TestCmirror:
    def test_3x3(self):
        assert EQ(_run_one(gp.cmirror, GRID_3x3)[:3, :3], np.rot90(GRID_3x3.T, k=2))


class TestDmirror:
    def test_3x3(self):
        assert EQ(_run_one(gp.dmirror, GRID_3x3)[:3, :3], GRID_3x3.T.copy())


# ============================================================================
# Spatial splitting
# ============================================================================

class TestTophalf:
    def test(self):
        assert EQ(_run_one(gp.tophalf, GRID_4x3)[:2, :3], GRID_4x3[:2])


class TestBottomhalf:
    def test(self):
        assert EQ(_run_one(gp.bottomhalf, GRID_4x3)[:2, :3], GRID_4x3[2:])


class TestLefthalf:
    def test(self):
        assert EQ(_run_one(gp.lefthalf, GRID_2x4)[:2, :2], GRID_2x4[:, :2])


class TestRighthalf:
    def test(self):
        assert EQ(_run_one(gp.righthalf, GRID_2x4)[:2, :2], GRID_2x4[:, 2:])


class TestHsplit:
    def test(self):
        assert EQ(_run_one(gp.hsplit, GRID_4x6)[:4, :3], GRID_4x6[:, :3])


class TestVsplit:
    def test(self):
        assert EQ(_run_one(gp.vsplit, GRID_6x4)[:3, :4], GRID_6x4[:3, :])


# ============================================================================
# Concatenation
# ============================================================================

class TestVconcat:
    def test(self):
        g2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.int64)
        assert EQ(_run_vconcat(GRID_3x3, g2)[:6, :3], np.vstack([GRID_3x3, g2]))


class TestHconcat:
    def test(self):
        g2 = np.array([[6, 5], [4, 3], [2, 1]], dtype=np.int64)
        assert EQ(_run_hconcat(GRID_3x2, g2)[:3, :4], np.hstack([GRID_3x2, g2]))


# ============================================================================
# Upscaling
# ============================================================================

class TestHupscale:
    def test_factor2(self):
        assert EQ(_run_one(gp.hupscale, GRID_3x2, {"factor": 2})[:3, :4],
                   np.repeat(GRID_3x2, 2, axis=1))


class TestVupscale:
    def test_factor2(self):
        assert EQ(_run_one(gp.vupscale, GRID_3x2, {"factor": 2})[:6, :2],
                   np.repeat(GRID_3x2, 2, axis=0))


class TestUpscale:
    def test_factor2(self):
        assert EQ(_run_one(gp.upscale, GRID_2x3, {"factor": 2})[:4, :6],
                   np.repeat(np.repeat(GRID_2x3, 2, axis=0), 2, axis=1))


class TestDownscale:
    def test_factor2(self):
        g = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ], dtype=np.int64)
        result = _run_one(gp.downscale, g, {"factor": 2})
        expected = g[::2, ::2]
        np.testing.assert_array_equal(result[:2, :2], expected)


# ============================================================================
# Cropping / trimming
# ============================================================================

class TestCrop:
    def test(self):
        assert EQ(_run_one(gp.crop, GRID_4x4, {"top": 1, "left": 1, "height": 2, "width": 2})[:2, :2],
                   GRID_4x4[1:3, 1:3])


class TestTrim:
    def test(self):
        assert EQ(_run_one(gp.trim, GRID_4x4)[:2, :2], GRID_4x4[1:3, 1:3])


# ============================================================================
# Arithmetic
# ============================================================================

class TestAdd:
    def test(self):
        g2 = np.ones((3, 3), dtype=np.int64)
        assert EQ(_run_same_two(gp.add, GRID_3x3, g2)[:3, :3], (GRID_3x3 + g2) % 10)


class TestSubtract:
    def test(self):
        g2 = np.ones((3, 3), dtype=np.int64)
        assert EQ(_run_same_two(gp.subtract, GRID_3x3, g2)[:3, :3], (GRID_3x3 - g2) % 10)


class TestMultiply:
    def test(self):
        g2 = np.full((3, 3), 2, dtype=np.int64)
        assert EQ(_run_same_two(gp.multiply, GRID_3x3, g2)[:3, :3], (GRID_3x3 * g2) % 10)


class TestMinimum:
    def test(self):
        g2 = np.full((3, 3), 5, dtype=np.int64)
        assert EQ(_run_same_two(gp.minimum, GRID_3x3, g2)[:3, :3], np.minimum(GRID_3x3, g2))


class TestMaximum:
    def test(self):
        g2 = np.full((3, 3), 5, dtype=np.int64)
        assert EQ(_run_same_two(gp.maximum, GRID_3x3, g2)[:3, :3], np.maximum(GRID_3x3, g2))


class TestIncrement:
    def test(self):
        assert EQ(_run_one(gp.increment, GRID_3x3, {"delta": 1})[:3, :3],
                   (GRID_3x3 + 1) % 10)


class TestDecrement:
    def test(self):
        assert EQ(_run_one(gp.decrement, GRID_3x3, {"delta": 1})[:3, :3],
                   (GRID_3x3 - 1) % 10)


class TestDouble:
    def test(self):
        assert EQ(_run_one(gp.double, GRID_3x3)[:3, :3], (GRID_3x3 * 2) % 10)


# ============================================================================
# Comparison
# ============================================================================

class TestBoth:
    def test(self):
        g1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
        g2 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
        expected = ((g1 > 0) & (g2 > 0)).astype(np.int64)
        assert EQ(_run_same_two(gp.both, g1, g2)[:3, :3], expected)


class TestEither:
    def test(self):
        g1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int64)
        g2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.int64)
        expected = ((g1 > 0) | (g2 > 0)).astype(np.int64)
        assert EQ(_run_same_two(gp.either, g1, g2)[:3, :3], expected)


class TestEquality:
    def test(self):
        g2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        assert EQ(_run_same_two(gp.equality, GRID_3x3, g2)[:3, :3],
                   (GRID_3x3 == g2).astype(np.int64))


class TestGreater:
    def test(self):
        g2 = np.full((3, 3), 5, dtype=np.int64)
        assert EQ(_run_same_two(gp.greater, GRID_3x3, g2)[:3, :3],
                   (GRID_3x3 > g2).astype(np.int64))


class TestLess:
    def test(self):
        g2 = np.full((3, 3), 5, dtype=np.int64)
        assert EQ(_run_same_two(gp.less, GRID_3x3, g2)[:3, :3],
                   (GRID_3x3 < g2).astype(np.int64))


# ============================================================================
# Canvas
# ============================================================================

class TestCanvas:
    def test(self):
        nodes, inits = [], []
        out_name, h, w = gp.canvas(nodes, inits, color=5, prefix="cvs")
        _n = gp.helper.make_node("Identity", [out_name], ["output"])
        nodes.append(_n)
        sess = _build_model(nodes, inits, [("input", [1, 10, 1, 1])])
        result = sess.run(None, {"input": np.zeros((1, 10, 1, 1), dtype=np.float32)})[0]
        assert np.all(_decode(result) == 5)


class TestCanvasLike:
    def test(self):
        nodes, inits = [], []
        out_name, h, w = gp.canvas_like(nodes, inits, "input", 4, 4, color=3, prefix="cvsl")
        _n = gp.helper.make_node("Identity", [out_name], ["output"])
        nodes.append(_n)
        graph = gp.helper.make_graph(
            nodes, "g",
            [gp.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 4, 4])],
            [gp.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 4, 4])],
            inits,
        )
        model = gp.helper.make_model(graph, opset_imports=[gp.helper.make_opsetid("", 17)])
        model.ir_version = 8
        sess = ort.InferenceSession(model.SerializeToString())
        result = sess.run(None, {"input": np.zeros((1, 10, 4, 4), dtype=np.float32)})[0]
        grid = _decode(result)
        assert np.all(grid == 3) and grid.shape == (4, 4)


# ============================================================================
# Color filtering
# ============================================================================

class TestOfcolor:
    def test_color5(self):
        assert EQ(_run_one(gp.ofcolor, GRID_3x3, {"color": 5})[:3, :3],
                   (GRID_3x3 == 5).astype(np.int64))

    def test_color0(self):
        assert EQ(_run_one(gp.ofcolor, GRID_4x4, {"color": 0})[:4, :4],
                   (GRID_4x4 == 0).astype(np.int64))


class TestFill:
    def test(self):
        mask_grid = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
        expected = GRID_3x3.copy()
        expected[mask_grid > 0] = 9
        assert EQ(_run_fill(gp.fill, GRID_3x3, mask_grid, {"color": 9})[:3, :3], expected)


class TestReplace:
    def test(self):
        result = _run_one(gp.replace, GRID_3x3, {"old_color": 1, "new_color": 9})
        expected = GRID_3x3.copy()
        expected[expected == 1] = 9
        assert EQ(result[:3, :3], expected)


class TestMerge:
    def test(self):
        g1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int64)
        g2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 9]], dtype=np.int64)
        assert EQ(_run_same_two(gp.merge, g1, g2)[:3, :3], np.maximum(g1, g2))


class TestPaint:
    def test(self):
        canvas = np.zeros((3, 3), dtype=np.int64)
        mask_grid = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
        expected = np.zeros((3, 3), dtype=np.int64)
        expected[mask_grid > 0] = 7
        assert EQ(_run_fill(gp.paint, canvas, mask_grid, {"color": 7})[:3, :3], expected)


# ============================================================================
# Shape / size
# ============================================================================

class TestHeight:
    def test(self):
        nodes, inits = [], []
        out, h, w = gp.height(nodes, inits, "x", 7, 5)
        assert h == 1 and w == 1 and len(inits) == 1


class TestWidth:
    def test(self):
        nodes, inits = [], []
        out, h, w = gp.width(nodes, inits, "x", 7, 5)
        assert h == 1 and w == 1


class TestShape:
    def test(self):
        nodes, inits = [], []
        out, h, w = gp.shape(nodes, inits, "x", 7, 5)
        assert h == 1 and w == 2


class TestSize:
    def test(self):
        nodes, inits = [], []
        out, h, w = gp.size(nodes, inits, "x", 7, 5)
        assert h == 1 and w == 1


class TestCorners:
    def test_ulcorner(self):
        nodes, inits = [], []
        _, h, w = gp.ulcorner(nodes, inits, "x", 5, 5)
        assert h == 1 and w == 2

    def test_urcorner(self):
        nodes, inits = [], []
        _, h, w = gp.urcorner(nodes, inits, "x", 5, 5)
        assert h == 1 and w == 2

    def test_llcorner(self):
        nodes, inits = [], []
        _, h, w = gp.llcorner(nodes, inits, "x", 5, 5)
        assert h == 1 and w == 2

    def test_lrcorner(self):
        nodes, inits = [], []
        _, h, w = gp.lrcorner(nodes, inits, "x", 5, 5)
        assert h == 1 and w == 2


# ============================================================================
# Node-level tests (verify ONNX op types)
# ============================================================================

class TestNodeLevelOps:
    def test_rot90(self):
        ns, ii = [], []
        gp.rot90(ns, ii, "x", 3, 3, prefix="r")
        assert any(n.op_type == "Transpose" for n in ns)
        assert any(n.op_type == "Gather" for n in ns)

    def test_rot180(self):
        ns, ii = [], []
        gp.rot180(ns, ii, "x", 3, 3, prefix="r")
        assert sum(1 for n in ns if n.op_type == "Gather") == 2

    def test_rot270(self):
        ns, ii = [], []
        gp.rot270(ns, ii, "x", 3, 3, prefix="r")
        assert any(n.op_type == "Transpose" for n in ns)

    def test_hmirror(self):
        ns, ii = [], []
        gp.hmirror(ns, ii, "x", 3, 3, prefix="h")
        assert any(n.op_type == "Gather" for n in ns)

    def test_vmirror(self):
        ns, ii = [], []
        gp.vmirror(ns, ii, "x", 3, 3, prefix="v")
        assert any(n.op_type == "Gather" for n in ns)

    def test_tophalf(self):
        ns, ii = [], []
        gp.tophalf(ns, ii, "x", 4, 3, prefix="t")
        assert any(n.op_type == "Slice" for n in ns)

    def test_bottomhalf(self):
        ns, ii = [], []
        gp.bottomhalf(ns, ii, "x", 4, 3, prefix="b")
        assert any(n.op_type == "Slice" for n in ns)

    def test_lefthalf(self):
        ns, ii = [], []
        gp.lefthalf(ns, ii, "x", 3, 4, prefix="l")
        assert any(n.op_type == "Slice" for n in ns)

    def test_righthalf(self):
        ns, ii = [], []
        gp.righthalf(ns, ii, "x", 3, 4, prefix="r")
        assert any(n.op_type == "Slice" for n in ns)

    def test_vconcat(self):
        ns, ii = [], []
        gp.vconcat(ns, ii, "a", "b", 3, 3, 3, prefix="vc")
        assert any(n.op_type == "Concat" for n in ns)

    def test_hconcat(self):
        ns, ii = [], []
        gp.hconcat(ns, ii, "a", "b", 3, 2, 2, prefix="hc")
        assert any(n.op_type == "Concat" for n in ns)

    def test_hupscale(self):
        ns, ii = [], []
        gp.hupscale(ns, ii, "x", 3, 3, factor=2, prefix="hu")
        assert any(n.op_type == "Tile" for n in ns)

    def test_vupscale(self):
        ns, ii = [], []
        gp.vupscale(ns, ii, "x", 3, 3, factor=2, prefix="vu")
        assert any(n.op_type == "Tile" for n in ns)

    def test_upscale(self):
        ns, ii = [], []
        gp.upscale(ns, ii, "x", 3, 3, factor=2, prefix="u")
        assert any(n.op_type == "Tile" for n in ns)

    def test_downscale(self):
        ns, ii = [], []
        gp.downscale(ns, ii, "x", 6, 6, factor=3, prefix="d")
        assert any(n.op_type == "AveragePool" for n in ns)

    def test_crop(self):
        ns, ii = [], []
        gp.crop(ns, ii, "x", 4, 4, prefix="c")
        assert any(n.op_type == "Slice" for n in ns)

    def test_trim(self):
        ns, ii = [], []
        gp.trim(ns, ii, "x", 4, 4, prefix="t")
        assert any(n.op_type == "Slice" for n in ns)

    def test_add(self):
        ns, ii = [], []
        gp.add(ns, ii, "a", "b", 3, 3)
        assert any(n.op_type == "Add" for n in ns)

    def test_subtract(self):
        ns, ii = [], []
        gp.subtract(ns, ii, "a", "b", 3, 3)
        assert any(n.op_type == "Sub" for n in ns)

    def test_multiply(self):
        ns, ii = [], []
        gp.multiply(ns, ii, "a", "b", 3, 3)
        assert any(n.op_type == "Mul" for n in ns)

    def test_divide(self):
        ns, ii = [], []
        gp.divide(ns, ii, "a", "b", 3, 3)
        assert any(n.op_type == "Div" for n in ns)

    def test_increment(self):
        ns, ii = [], []
        gp.increment(ns, ii, "x", 3, 3)
        assert any(n.op_type == "Add" for n in ns)

    def test_decrement(self):
        ns, ii = [], []
        gp.decrement(ns, ii, "x", 3, 3)
        assert any(n.op_type == "Neg" for n in ns)

    def test_double(self):
        ns, ii = [], []
        gp.double(ns, ii, "x", 3, 3)
        assert any(n.op_type == "Mul" for n in ns)

    def test_negate(self):
        ns, ii = [], []
        gp.negate(ns, ii, "x", 3, 3)
        assert any(n.op_type == "Neg" for n in ns)

    def test_both(self):
        ns, ii = [], []
        gp.both(ns, ii, "a", "b", 3, 3)
        assert any(n.op_type == "And" for n in ns)

    def test_either(self):
        ns, ii = [], []
        gp.either(ns, ii, "a", "b", 3, 3)
        assert any(n.op_type == "Or" for n in ns)

    def test_equality(self):
        ns, ii = [], []
        gp.equality(ns, ii, "a", "b", 3, 3)
        assert any(n.op_type == "Equal" for n in ns)

    def test_greater(self):
        ns, ii = [], []
        gp.greater(ns, ii, "a", "b", 3, 3)
        assert any(n.op_type == "Greater" for n in ns)

    def test_less(self):
        ns, ii = [], []
        gp.less(ns, ii, "a", "b", 3, 3)
        assert any(n.op_type == "Less" for n in ns)

    def test_even(self):
        ns, ii = [], []
        gp.even(ns, ii, "x", 3, 3)
        assert any(n.op_type == "Mod" for n in ns)

    def test_sign(self):
        ns, ii = [], []
        gp.sign(ns, ii, "x", 3, 3)
        assert any(n.op_type == "Sign" for n in ns)

    def test_positive(self):
        ns, ii = [], []
        gp.positive(ns, ii, "x", 3, 3)
        assert any(n.op_type == "Greater" for n in ns)

    def test_invert(self):
        ns, ii = [], []
        gp.invert(ns, ii, "x", 3, 3)
        assert any(n.op_type == "Neg" for n in ns)

    def test_ofcolor(self):
        ns, ii = [], []
        gp.ofcolor(ns, ii, "x", 3, 3, color=1)
        assert any(n.op_type == "Equal" for n in ns)

    def test_replace(self):
        ns, ii = [], []
        gp.replace(ns, ii, "x", 3, 3)
        assert any(n.op_type == "Greater" for n in ns)

    def test_merge(self):
        ns, ii = [], []
        gp.merge(ns, ii, "a", "b", 3, 3)
        assert any(n.op_type == "Max" for n in ns)

    def test_fill(self):
        ns, ii = [], []
        gp.fill(ns, ii, "x", "mask", 3, 3)
        assert any(n.op_type == "Mul" for n in ns)

    def test_cellwise_add(self):
        ns, ii = [], []
        gp.cellwise(ns, ii, "a", "b", 3, 3, func="add")
        assert any(n.op_type == "Add" for n in ns)

    def test_cellwise_sub(self):
        ns, ii = [], []
        gp.cellwise(ns, ii, "a", "b", 3, 3, func="sub")
        assert any(n.op_type == "Sub" for n in ns)

    def test_cellwise_mul(self):
        ns, ii = [], []
        gp.cellwise(ns, ii, "a", "b", 3, 3, func="mul")
        assert any(n.op_type == "Mul" for n in ns)


# ============================================================================
# Integration: chain primitives
# ============================================================================

class TestIntegration:
    def test_rot90_then_tophalf(self):
        nodes, inits = [], []
        o1, h1, w1 = gp.rot90(nodes, inits, "input", 4, 4, prefix="r0")
        o2, h2, w2 = gp.tophalf(nodes, inits, o1, h1, w1, prefix="t0")
        o2, h2, w2 = gp.pad_canvas(nodes, inits, o2, h2, w2, prefix="p0")
        nodes.append(gp.helper.make_node("Identity", [o2], ["output"]))
        sess = _build_model(nodes, inits, [("input", [1, 10, 4, 4])])
        grid_out = _decode(sess.run(None, {"input": _encode(GRID_4x4)})[0])
        expected = np.rot90(GRID_4x4, k=-1)[:2]
        assert EQ(grid_out[:2, :4], expected)

    def test_hupscale_vupscale(self):
        nodes, inits = [], []
        o1, h1, w1 = gp.hupscale(nodes, inits, "input", 3, 3, factor=2, prefix="hu")
        o2, h2, w2 = gp.vupscale(nodes, inits, o1, h1, w1, factor=2, prefix="vu")
        o2, h2, w2 = gp.pad_canvas(nodes, inits, o2, h2, w2, prefix="p")
        nodes.append(gp.helper.make_node("Identity", [o2], ["output"]))
        sess = _build_model(nodes, inits, [("input", [1, 10, 3, 3])])
        grid_out = _decode(sess.run(None, {"input": _encode(GRID_3x3)})[0])
        expected = np.repeat(np.repeat(GRID_3x3, 2, axis=0), 2, axis=1)
        assert EQ(grid_out[:6, :6], expected)

    def test_crop_then_upscale(self):
        nodes, inits = [], []
        o1, h1, w1 = gp.crop(nodes, inits, "input", 4, 4,
                              top=1, left=1, height=2, width=2, prefix="c0")
        o2, h2, w2 = gp.upscale(nodes, inits, o1, h1, w1, factor=2, prefix="u0")
        o2, h2, w2 = gp.pad_canvas(nodes, inits, o2, h2, w2, prefix="p0")
        nodes.append(gp.helper.make_node("Identity", [o2], ["output"]))
        sess = _build_model(nodes, inits, [("input", [1, 10, 4, 4])])
        grid_out = _decode(sess.run(None, {"input": _encode(GRID_4x4)})[0])
        cropped = GRID_4x4[1:3, 1:3]
        expected = np.repeat(np.repeat(cropped, 2, axis=0), 2, axis=1)
        assert EQ(grid_out[:4, :4], expected)


# ============================================================================
# Pass-through stubs — signature only
# ============================================================================

STUBS_SINGLE = [
    "objects", "partition", "fgpartition",
    "mfilter", "sfilter", "sizefilter",
    "delta", "box", "backdrop", "inbox", "outbox",
    "corners", "frontiers",
    "hline", "vline", "connect", "shoot",
    "neighbors", "dneighbors", "gravitate",
    "compose", "chain", "fork",
    "lbind", "rbind",
    "apply", "mapply",
    "branch", "power", "repeat",
    "order", "mpapply", "prapply",
    "papply", "rapply",
    "dup", "swap",
    "astuple", "initset", "totuple",
    "insert", "remove", "other",
    "first", "last", "extract",
    "dedupe", "contained",
    "normalize", "compress",
    "index", "interval",
    "toivec", "tojvec",
    "toindices", "toobject",
    "occurrences", "position",
    "mostcolor", "leastcolor",
    "mostcommon", "leastcommon",
    "colorcount", "valmax", "valmin",
    "matcher", "switch",
]

STUBS_TWO = ["combine", "pair", "product"]


class TestStubSignatures:
    @pytest.mark.parametrize("name", STUBS_SINGLE)
    def test_single(self, name):
        fn = getattr(gp, name)
        nodes, inits = [], []
        out, h, w = fn(nodes, inits, "x", 5, 5, prefix=f"s_{name}")
        assert isinstance(out, str) and isinstance(h, int) and isinstance(w, int)

    @pytest.mark.parametrize("name", STUBS_TWO)
    def test_two(self, name):
        fn = getattr(gp, name)
        nodes, inits = [], []
        out, h, w = fn(nodes, inits, "a", "b", 5, 5, prefix=f"s_{name}")
        assert isinstance(out, str) and isinstance(h, int) and isinstance(w, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
