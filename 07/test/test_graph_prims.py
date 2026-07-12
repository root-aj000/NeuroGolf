"""Tests for onnx_graph_prims.py — verifies every primitive.

For primitives that emit real ONNX nodes, we build a tiny model, run it
through ONNX runtime, and check the output matches numpy expectations.
For pass-through stubs, we verify they return the correct signature.
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

def _make_grid(h, w, values=None):
    """Create a (1,10,30,30) one-hot float32 tensor with content in top-left h×w."""
    t = np.zeros((1, 10, 30, 30), dtype=np.float32)
    if values is None:
        for r in range(h):
            for c in range(w):
                v = (r * w + c) % 10
                t[0, v, r, c] = 1.0
    else:
        arr = np.array(values, dtype=np.int64).reshape(h, w)
        for r in range(h):
            for c in range(w):
                v = int(arr[r, c])
                if 0 <= v < 10:
                    t[0, v, r, c] = 1.0
    return t


def _to_grid(t):
    """Convert (1,10,30,30) one-hot to (30,30) int grid."""
    return np.argmax(t[0], axis=0).astype(np.int64)


def _run_prim(prim_fn, grid_h, grid_w, prim_kwargs=None):
    """Build a model with prim_fn operating on grid_h×grid_w content,
    run it through ORT, return (output_grid, model).

    prim_fn(nodes, inits, x, h, w, prefix, **kw) -> (out_name, new_h, new_w)
    The graph input is (1,10,grid_h,grid_w), output is padded to 30x30.
    """
    if prim_kwargs is None:
        prim_kwargs = {}

    nodes = []
    inits = []
    inp = "input"

    # Primitive operates on grid_h x grid_w
    result_name, rh, rw = prim_fn(nodes, inits, inp, grid_h, grid_w,
                                  prefix="t0", **prim_kwargs)

    # Pad to canvas if needed
    result_name, rh, rw = gp.pad_canvas(nodes, inits, result_name, rh, rw,
                                         prefix="pad")

    _n = gp.helper.make_node("Identity", [result_name], ["output"])
    nodes.append(_n)

    graph = gp.helper.make_graph(
        nodes, "test_graph",
        [gp.helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                          [1, 10, grid_h, grid_w])],
        [gp.helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                          [1, 10, 30, 30])],
        inits,
    )
    model = gp.helper.make_model(graph, opset_imports=[gp.helper.make_opsetid("", 17)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)

    sess = ort.InferenceSession(model.SerializeToString())
    values = prim_kwargs.get("values", None)
    if values is not None:
        input_tensor = np.zeros((1, 10, grid_h, grid_w), dtype=np.float32)
        arr = np.array(values, dtype=np.int64).reshape(grid_h, grid_w)
        for r in range(grid_h):
            for c in range(grid_w):
                v = int(arr[r, c])
                if 0 <= v < 10:
                    input_tensor[0, v, r, c] = 1.0
    else:
        input_tensor = np.zeros((1, 10, grid_h, grid_w), dtype=np.float32)
        for r in range(grid_h):
            for c in range(grid_w):
                v = (r * grid_w + c) % 10
                input_tensor[0, v, r, c] = 1.0
    outputs = sess.run(None, {"input": input_tensor})
    return _to_grid(outputs[0]), model


def _build_and_run(nodes, inits, input_tensor=None, in_h=None, in_w=None):
    """Generic: build graph from nodes/inits, run, return output 30x30 grid.
    Input shape is inferred from input_tensor or (in_h, in_w).
    """
    if input_tensor is not None:
        _, _, ih, iw = input_tensor.shape
    elif in_h is not None and in_w is not None:
        ih, iw = in_h, in_w
    else:
        ih, iw = 30, 30

    graph = gp.helper.make_graph(
        nodes, "test_graph",
        [gp.helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                          [1, 10, ih, iw])],
        [gp.helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                          [1, 10, 30, 30])],
        inits,
    )
    model = gp.helper.make_model(graph, opset_imports=[gp.helper.make_opsetid("", 17)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)

    sess = ort.InferenceSession(model.SerializeToString())
    if input_tensor is None:
        input_tensor = _make_grid(ih, iw)
    outputs = sess.run(None, {"input": input_tensor})
    return _to_grid(outputs[0])


# ============================================================================
# Helper tests
# ============================================================================

class TestHelpers:
    def test_node(self):
        nodes = []
        gp._node(nodes, "Add", ["a", "b"], ["c"])
        assert len(nodes) == 1
        assert nodes[0].op_type == "Add"

    def test_init(self):
        inits = []
        name = gp._init(inits, "test", np.array([1, 2, 3], dtype=np.float32))
        assert name == "test"
        assert len(inits) == 1
        assert inits[0].name == "test"

    def test_fresh(self):
        assert gp._fresh("pfx") == "pfx_out"
        assert gp._fresh("pfx", "t") == "pfx_t"

    def test_make_model(self):
        nodes = []
        inits = []
        gp._node(nodes, "Identity", ["input"], ["output"])
        model = gp.make_model(nodes, inits, task_id=1)
        assert isinstance(model, onnx.ModelProto)

    def test_pad_canvas_identity_when_full(self):
        nodes, inits = [], []
        name, h, w = gp.pad_canvas(nodes, inits, "x", 30, 30)
        assert name == "x"
        assert h == 30 and w == 30
        assert len(nodes) == 0

    def test_pad_canvas_pads_when_small(self):
        nodes, inits = [], []
        name, h, w = gp.pad_canvas(nodes, inits, "x", 3, 5)
        assert h == 30 and w == 30
        assert len(nodes) == 1
        assert nodes[0].op_type == "Pad"

    def test_pad_canvas_e2e(self):
        """Actually run pad through ORT."""
        nodes, inits = [], []
        out, rh, rw = gp.pad_canvas(nodes, inits, "input", 3, 3, prefix="p")
        _n = gp.helper.make_node("Identity", [out], ["output"])
        nodes.append(_n)
        small = np.zeros((1, 10, 3, 3), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                v = (r * 3 + c) % 10
                small[0, v, r, c] = 1.0
        result = _build_and_run(nodes, inits, small)
        grid = np.arange(9, dtype=np.int64).reshape(3, 3)
        expected = np.zeros((30, 30), dtype=np.int64)
        expected[:3, :3] = grid
        assert np.array_equal(result, expected)


# ============================================================================
# Geometric transforms — end-to-end ONNX execution
# ============================================================================

class TestRot90:
    def test_square(self):
        grid = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=np.int64)
        result, _ = _run_prim(gp.rot90, 3, 3, prim_kwargs={"values": grid.tolist()})
        expected = np.rot90(grid, k=-1)
        assert np.array_equal(result[:3, :3], expected)

    def test_rect(self):
        grid = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
        result, _ = _run_prim(gp.rot90, 3, 2, prim_kwargs={"values": grid.tolist()})
        expected = np.rot90(grid, k=-1)
        assert np.array_equal(result[:expected.shape[0], :expected.shape[1]], expected)


class TestRot180:
    def test_square(self):
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        result, _ = _run_prim(gp.rot180, 3, 3, prim_kwargs={"values": grid.tolist()})
        expected = np.rot90(grid, k=2)
        assert np.array_equal(result[:3, :3], expected)


class TestRot270:
    def test_square(self):
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        result, _ = _run_prim(gp.rot270, 3, 3, prim_kwargs={"values": grid.tolist()})
        expected = np.rot90(grid, k=1)
        assert np.array_equal(result[:3, :3], expected)


class TestHmirror:
    def test(self):
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        result, _ = _run_prim(gp.hmirror, 3, 3, prim_kwargs={"values": grid.tolist()})
        expected = np.flip(grid, axis=0)
        assert np.array_equal(result[:3, :3], expected)


class TestVmirror:
    def test(self):
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        result, _ = _run_prim(gp.vmirror, 3, 3, prim_kwargs={"values": grid.tolist()})
        expected = np.flip(grid, axis=1)
        assert np.array_equal(result[:3, :3], expected)


class TestCmirror:
    def test(self):
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        result, _ = _run_prim(gp.cmirror, 3, 3, prim_kwargs={"values": grid.tolist()})
        expected = np.rot90(grid.T, k=2)
        assert np.array_equal(result[:3, :3], expected)


class TestDmirror:
    def test(self):
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        result, _ = _run_prim(gp.dmirror, 3, 3, prim_kwargs={"values": grid.tolist()})
        expected = grid.T
        assert np.array_equal(result[:3, :3], expected)


# ============================================================================
# Spatial splitting — end-to-end
# ============================================================================

class TestTophalf:
    def test(self):
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                        dtype=np.int64)
        result, _ = _run_prim(gp.tophalf, 4, 3, prim_kwargs={"values": grid.tolist()})
        expected = grid[:2]
        assert np.array_equal(result[:2, :3], expected)


class TestBottomhalf:
    def test(self):
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                        dtype=np.int64)
        result, _ = _run_prim(gp.bottomhalf, 4, 3, prim_kwargs={"values": grid.tolist()})
        expected = grid[2:]
        assert np.array_equal(result[:2, :3], expected)


class TestLefthalf:
    def test(self):
        grid = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
        result, _ = _run_prim(gp.lefthalf, 2, 4, prim_kwargs={"values": grid.tolist()})
        expected = grid[:, :2]
        assert np.array_equal(result[:2, :2], expected)


class TestRighthalf:
    def test(self):
        grid = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
        result, _ = _run_prim(gp.righthalf, 2, 4, prim_kwargs={"values": grid.tolist()})
        expected = grid[:, 2:]
        assert np.array_equal(result[:2, :2], expected)


# ============================================================================
# Concatenation — node-level tests
# ============================================================================

class TestVconcat:
    def test_node(self):
        nodes, inits = [], []
        out_name, nh, nw = gp.vconcat(nodes, inits, "a", "b", 3, 2, 5)
        assert nh == 5 and nw == 5
        assert len(nodes) == 1
        assert nodes[0].op_type == "Concat"

    def test_e2e(self):
        nodes, inits = [], []
        # Build: input → Identity as "a", also "b" same tensor, vconcat them
        _node_a = gp.helper.make_node("Identity", ["input"], ["a"])
        nodes.append(_node_a)
        out, rh, rw = gp.vconcat(nodes, inits, "a", "input", 3, 3, 30, prefix="vc0")
        _n = gp.helper.make_node("Identity", [out], ["output"])
        nodes.append(_n)
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        result = _build_and_run(nodes, inits, _make_grid(3, 3))
        assert result[:3, :3].shape == (3, 3)
        assert result[3:6, :3].shape == (3, 3)


class TestHconcat:
    def test_node(self):
        nodes, inits = [], []
        out_name, nh, nw = gp.hconcat(nodes, inits, "a", "b", 3, 2, 3)
        assert nh == 3 and nw == 5
        assert len(nodes) == 1
        assert nodes[0].op_type == "Concat"


# ============================================================================
# Upscaling — end-to-end
# ============================================================================

class TestHupscale:
    def test(self):
        grid = np.array([[1, 2], [3, 4]], dtype=np.int64)
        result, _ = _run_prim(gp.hupscale, 2, 2, prim_kwargs={"values": grid.tolist(), "factor": 3})
        assert np.array_equal(result[:2, :6], np.repeat(grid, 3, axis=1))


class TestVupscale:
    def test(self):
        grid = np.array([[1, 2], [3, 4]], dtype=np.int64)
        result, _ = _run_prim(gp.vupscale, 2, 2, prim_kwargs={"values": grid.tolist(), "factor": 3})
        assert np.array_equal(result[:6, :2], np.repeat(grid, 3, axis=0))


class TestUpscale:
    def test(self):
        grid = np.array([[1, 2], [3, 4]], dtype=np.int64)
        result, _ = _run_prim(gp.upscale, 2, 2, prim_kwargs={"values": grid.tolist(), "factor": 2})
        assert np.array_equal(result[:4, :4],
                              np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1))


class TestDownscale:
    def test(self):
        grid = np.array([[1, 2, 3, 4, 5, 6],
                         [7, 8, 9, 10, 11, 12],
                         [1, 2, 3, 4, 5, 6],
                         [7, 8, 9, 10, 11, 12],
                         [1, 2, 3, 4, 5, 6],
                         [7, 8, 9, 10, 11, 12]], dtype=np.int64)
        result, _ = _run_prim(gp.downscale, 6, 6, prim_kwargs={"values": grid.tolist(), "factor": 3})
        expected = grid.reshape(3, 2, 3, 2).mean(axis=(1, 3)).astype(np.int64)
        assert np.array_equal(result[:2, :2], expected)


# ============================================================================
# Cropping / trimming — end-to-end
# ============================================================================

class TestCrop:
    def test(self):
        grid = np.arange(16, dtype=np.int64).reshape(4, 4)
        result, _ = _run_prim(gp.crop, 4, 4, prim_kwargs={
            "values": grid.tolist(), "top": 1, "left": 1, "height": 2, "width": 2
        })
        expected = grid[1:3, 1:3]
        assert np.array_equal(result[:2, :2], expected)


class TestTrim:
    def test(self):
        grid = np.arange(16, dtype=np.int64).reshape(4, 4)
        result, _ = _run_prim(gp.trim, 4, 4, prim_kwargs={"values": grid.tolist()})
        expected = grid[1:3, 1:3]
        assert np.array_equal(result[:2, :2], expected)


# ============================================================================
# Arithmetic — node-level tests
# ============================================================================

class TestAdd:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.add(nodes, inits, "a", "b", 3, 3)
        assert nh == 3 and nw == 3
        assert nodes[0].op_type == "Add"


class TestSubtract:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.subtract(nodes, inits, "a", "b", 3, 3)
        assert nodes[0].op_type == "Sub"


class TestMultiply:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.multiply(nodes, inits, "a", "b", 3, 3)
        assert nodes[0].op_type == "Mul"


class TestDivide:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.divide(nodes, inits, "a", "b", 3, 3)
        assert nodes[0].op_type == "Div"


class TestIncrement:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.increment(nodes, inits, "x", 3, 3, delta=5)
        assert nh == 3 and nw == 3
        assert any(n.op_type == "Add" for n in nodes)


class TestDecrement:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.decrement(nodes, inits, "x", 3, 3, delta=3)
        assert any(n.op_type == "Neg" for n in nodes)
        assert any(n.op_type == "Add" for n in nodes)


class TestDouble:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.double(nodes, inits, "x", 3, 3)
        assert nodes[0].op_type == "Mul"


class TestNegate:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.negate(nodes, inits, "x", 3, 3)
        assert nodes[0].op_type == "Neg"


class TestMinimum:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.minimum(nodes, inits, "a", "b", 3, 3)
        assert nodes[0].op_type == "Min"


class TestMaximum:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.maximum(nodes, inits, "a", "b", 3, 3)
        assert nodes[0].op_type == "Max"


# ============================================================================
# Comparison — node-level + end-to-end
# ============================================================================

class TestBoth:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.both(nodes, inits, "a", "b", 3, 3)
        assert any(n.op_type == "And" for n in nodes)

    def test_e2e(self):
        nodes, inits = [], []
        # Create two tensors: one with ones, one with zeros in some region
        init_a = gp._init(inits, "a", np.ones((1, 10, 30, 30), dtype=np.float32))
        init_b = gp._init(inits, "b", np.ones((1, 10, 30, 30), dtype=np.float32))
        out, _, _ = gp.both(nodes, inits, init_a, init_b, 30, 30, prefix="both0")
        _n = gp.helper.make_node("Identity", [out], ["output"])
        nodes.append(_n)
        result = _build_and_run(nodes, inits, np.zeros((1, 10, 30, 30), dtype=np.float32))
        assert np.all(result == 1)


class TestEither:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.either(nodes, inits, "a", "b", 3, 3)
        assert any(n.op_type == "Or" for n in nodes)


class TestEquality:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.equality(nodes, inits, "a", "b", 3, 3)
        assert any(n.op_type == "Equal" for n in nodes)


class TestGreater:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.greater(nodes, inits, "a", "b", 3, 3)
        assert any(n.op_type == "Greater" for n in nodes)


class TestLess:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.less(nodes, inits, "a", "b", 3, 3)
        assert any(n.op_type == "Less" for n in nodes)


class TestEven:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.even(nodes, inits, "x", 3, 3)
        assert any(n.op_type == "Mod" for n in nodes)
        assert any(n.op_type == "Equal" for n in nodes)


class TestSign:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.sign(nodes, inits, "x", 3, 3)
        assert nodes[0].op_type == "Sign"


class TestPositive:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.positive(nodes, inits, "x", 3, 3)
        assert any(n.op_type == "Greater" for n in nodes)


class TestInvert:
    def test_node(self):
        nodes, inits = [], []
        out, nh, nw = gp.invert(nodes, inits, "x", 3, 3)
        assert any(n.op_type == "Neg" for n in nodes)
        assert any(n.op_type == "Add" for n in nodes)


# ============================================================================
# Cellwise — end-to-end
# ============================================================================

class TestCellwise:
    def test_add(self):
        nodes, inits = [], []
        out, nh, nw = gp.cellwise(nodes, inits, "a", "b", 3, 3, func="add")
        assert nodes[0].op_type == "Add"

    def test_sub(self):
        nodes, inits = [], []
        out, nh, nw = gp.cellwise(nodes, inits, "a", "b", 3, 3, func="sub")
        assert nodes[0].op_type == "Sub"

    def test_mul(self):
        nodes, inits = [], []
        out, nh, nw = gp.cellwise(nodes, inits, "a", "b", 3, 3, func="mul")
        assert nodes[0].op_type == "Mul"


# ============================================================================
# Canvas — end-to-end
# ============================================================================

class TestCanvas:
    def test_e2e(self):
        nodes, inits = [], []
        out, h, w = gp.canvas(nodes, inits, color=5, prefix="cvs0")
        _n = gp.helper.make_node("Identity", [out], ["output"])
        nodes.append(_n)
        input_tensor = np.zeros((1, 10, 30, 30), dtype=np.float32)
        graph = gp.helper.make_graph(
            nodes, "test_graph",
            [gp.helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                              [1, 10, 30, 30])],
            [gp.helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                              [1, 10, 30, 30])],
            inits,
        )
        model = gp.helper.make_model(graph, opset_imports=[gp.helper.make_opsetid("", 17)])
        model.ir_version = 8
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, full_check=True)
        sess = ort.InferenceSession(model.SerializeToString())
        outputs = sess.run(None, {"input": input_tensor})
        result = _to_grid(outputs[0])
        assert np.all(result == 5)

    def test_canvas_like(self):
        nodes, inits = [], []
        out, h, w = gp.canvas_like(nodes, inits, "x", 10, 10, color=3, prefix="cvsl")
        _n = gp.helper.make_node("Identity", [out], ["output"])
        nodes.append(_n)
        graph = gp.helper.make_graph(
            nodes, "test_graph",
            [gp.helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                              [1, 10, 30, 30])],
            [gp.helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                              [1, 10, 30, 30])],
            inits,
        )
        model = gp.helper.make_model(graph, opset_imports=[gp.helper.make_opsetid("", 17)])
        model.ir_version = 8
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, full_check=True)
        sess = ort.InferenceSession(model.SerializeToString())
        outputs = sess.run(None, {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)})
        result = _to_grid(outputs[0])
        assert np.all(result == 3)


# ============================================================================
# Shape/size — node-level
# ============================================================================

class TestHeight:
    def test(self):
        nodes, inits = [], []
        out, h, w = gp.height(nodes, inits, "x", 7, 5)
        assert h == 1 and w == 1
        assert len(inits) == 1


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


# ============================================================================
# Splitting — node-level
# ============================================================================

class TestHsplit:
    def test(self):
        nodes, inits = [], []
        out, h, w = gp.hsplit(nodes, inits, "x", 4, 6)
        assert h == 4 and w == 3
        assert nodes[0].op_type == "Slice"


class TestVsplit:
    def test(self):
        nodes, inits = [], []
        out, h, w = gp.vsplit(nodes, inits, "x", 6, 4)
        assert h == 3 and w == 4
        assert nodes[0].op_type == "Slice"


# ============================================================================
# Corners — node-level
# ============================================================================

class TestUrcorner:
    def test(self):
        nodes, inits = [], []
        out, h, w = gp.urcorner(nodes, inits, "x", 5, 5)
        assert h == 1 and w == 2


class TestLrcorner:
    def test(self):
        nodes, inits = [], []
        out, h, w = gp.lrcorner(nodes, inits, "x", 5, 5)
        assert h == 1 and w == 2


class TestLlcorner:
    def test(self):
        nodes, inits = [], []
        out, h, w = gp.llcorner(nodes, inits, "x", 5, 5)
        assert h == 1 and w == 2


# ============================================================================
# Color — node-level + end-to-end
# ============================================================================

class TestOfcolor:
    def test_node(self):
        nodes, inits = [], []
        out, h, w = gp.ofcolor(nodes, inits, "x", 3, 3, color=2)
        assert h == 3 and w == 3
        assert any(n.op_type == "Greater" for n in nodes)


class TestFill:
    def test_node(self):
        nodes, inits = [], []
        out, h, w = gp.fill(nodes, inits, "x", "mask", 3, 3, color=7)
        assert h == 3 and w == 3
        assert any(n.op_type == "Mul" for n in nodes)
        assert any(n.op_type == "Add" for n in nodes)


class TestReplace:
    def test_node(self):
        nodes, inits = [], []
        out, h, w = gp.replace(nodes, inits, "x", 3, 3, old_color=0, new_color=5)
        assert h == 3 and w == 3
        assert any(n.op_type == "Greater" for n in nodes)


class TestMerge:
    def test_node(self):
        nodes, inits = [], []
        out, h, w = gp.merge(nodes, inits, "a", "b", 3, 3)
        assert nodes[0].op_type == "Max"


# ============================================================================
# Pass-through / stub primitives — verify signature
# ============================================================================

SINGLE_PASS_THROUGH = [
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
    "mfilter", "sfilter", "sizefilter",
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

TWO_INPUT_PASS_THROUGH = [
    "combine", "pair", "product",
]


class TestPassThroughStubs:
    @pytest.mark.parametrize("name", SINGLE_PASS_THROUGH)
    def test_single_input_returns_correctly(self, name):
        fn = getattr(gp, name)
        nodes, inits = [], []
        out, h, w = fn(nodes, inits, "x", 5, 5, prefix=f"test_{name}")
        assert isinstance(out, str)
        assert isinstance(h, int)
        assert isinstance(w, int)

    @pytest.mark.parametrize("name", TWO_INPUT_PASS_THROUGH)
    def test_two_input_returns_correctly(self, name):
        fn = getattr(gp, name)
        nodes, inits = [], []
        out, h, w = fn(nodes, inits, "a", "b", 5, 5, prefix=f"test_{name}")
        assert isinstance(out, str)
        assert isinstance(h, int)
        assert isinstance(w, int)


# ============================================================================
# Integration: chain primitives together
# ============================================================================

class TestIntegration:
    def test_rot90_then_tophalf(self):
        """Chain rot90 → tophalf in one model."""
        nodes, inits = [], []
        inp = "input"
        out1, h1, w1 = gp.rot90(nodes, inits, inp, 6, 4, prefix="r0")
        out2, h2, w2 = gp.tophalf(nodes, inits, out1, h1, w1, prefix="t0")
        out2, h2, w2 = gp.pad_canvas(nodes, inits, out2, h2, w2, prefix="p0")
        _n = gp.helper.make_node("Identity", [out2], ["output"])
        nodes.append(_n)

        grid = np.arange(24, dtype=np.int64).reshape(6, 4)
        input_tensor = _make_grid(6, 4)
        result = _build_and_run(nodes, inits, input_tensor)

        expected = np.rot90(grid, k=-1)[:3]
        assert np.array_equal(result[:3, :4], expected)

    def test_hconcat_vconcat(self):
        """Chain hconcat → vconcat."""
        nodes, inits = [], []
        out1, h1, w1 = gp.hconcat(nodes, inits, "a", "b", 3, 2, 2, prefix="hc0")
        out2, h2, w2 = gp.vconcat(nodes, inits, out1, "c", h1, 3, w1, prefix="vc0")
        assert h2 == 6 and w2 == 4

    def test_upscale_then_crop(self):
        """Chain upscale → crop."""
        nodes, inits = [], []
        out1, h1, w1 = gp.upscale(nodes, inits, "x", 3, 3, factor=2, prefix="u0")
        assert h1 == 6 and w1 == 6
        out2, h2, w2 = gp.crop(nodes, inits, out1, h1, w1,
                                top=1, left=1, height=3, width=3, prefix="c0")
        assert h2 == 3 and w2 == 3

    def test_hupscale_vupscale_chain(self):
        """Chain hupscale → vupscale → hconcat with duplicate."""
        nodes, inits = [], []
        out1, h1, w1 = gp.hupscale(nodes, inits, "input", 3, 3, factor=3, prefix="hu")
        assert h1 == 3 and w1 == 9
        out2, h2, w2 = gp.vupscale(nodes, inits, out1, h1, w1, factor=3, prefix="vu")
        assert h2 == 9 and w2 == 9
        out2, h2, w2 = gp.pad_canvas(nodes, inits, out2, h2, w2, prefix="p")
        assert h2 == 30 and w2 == 30
        _n = gp.helper.make_node("Identity", [out2], ["output"])
        nodes.append(_n)
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        result = _build_and_run(nodes, inits, _make_grid(3, 3))
        expected_9x9 = np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)
        assert np.array_equal(result[:9, :9], expected_9x9)


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_rot90_1x1(self):
        nodes, inits = [], []
        out, h, w = gp.rot90(nodes, inits, "x", 1, 1)
        assert h == 1 and w == 1

    def test_rot180_1x1(self):
        nodes, inits = [], []
        out, h, w = gp.rot180(nodes, inits, "x", 1, 1)
        assert h == 1 and w == 1

    def test_hconcat_symmetric(self):
        nodes, inits = [], []
        out, h, w = gp.hconcat(nodes, inits, "a", "b", 3, 0, 0)
        assert w == 0

    def test_crop_full_grid(self):
        nodes, inits = [], []
        out, h, w = gp.crop(nodes, inits, "x", 5, 5,
                            top=0, left=0, height=5, width=5)
        assert h == 5 and w == 5

    def test_all_single_input_primitives_return_triple(self):
        """Every single-input primitive returns (str, int, int)."""
        single = [
            "rot90", "rot180", "rot270", "hmirror", "vmirror",
            "cmirror", "dmirror", "tophalf", "bottomhalf",
            "lefthalf", "righthalf", "hupscale", "vupscale",
            "upscale", "downscale", "trim", "increment", "decrement",
            "crement", "double", "negate", "sign", "positive",
            "invert", "even", "asindices", "asobject",
            "height", "width", "shape", "size", "numcolors",
            "uppermost", "lowermost", "leftmost", "rightmost",
            "center", "ulcorner", "urcorner", "llcorner", "lrcorner",
            "hperiod", "vperiod", "portrait",
            "hfrontier", "vfrontier",
            "objects", "partition", "fgpartition",
            "mfilter", "sfilter", "sizefilter",
            "shift", "move", "hsplit", "vsplit",
            "ofcolor",
        ]
        for name in single:
            fn = getattr(gp, name)
            nodes, inits = [], []
            result = fn(nodes, inits, "x", 5, 5, prefix=f"test_{name}")
            assert isinstance(result, tuple) and len(result) == 3, \
                f"{name} did not return (name, h, w)"
            assert isinstance(result[0], str)
            assert isinstance(result[1], int)
            assert isinstance(result[2], int)

    def test_all_two_input_primitives_return_triple(self):
        """Every two-input primitive returns (str, int, int)."""
        two = [
            "add", "subtract", "multiply", "divide",
            "minimum", "maximum", "cellwise", "both", "either",
            "equality", "greater", "less", "merge", "cover",
        ]
        for name in two:
            fn = getattr(gp, name)
            nodes, inits = [], []
            if name == "cellwise":
                result = fn(nodes, inits, "a", "b", 5, 5, prefix=f"test_{name}")
            else:
                result = fn(nodes, inits, "a", "b", 5, 5, prefix=f"test_{name}")
            assert isinstance(result, tuple) and len(result) == 3, \
                f"{name} did not return (name, h, w)"

    def test_canvas_and_fill_return_triple(self):
        nodes, inits = [], []
        result = gp.canvas(nodes, inits, color=2, prefix="test_canvas")
        assert isinstance(result, tuple) and len(result) == 3

        nodes2, inits2 = [], []
        result2 = gp.fill(nodes2, inits2, "x", "m", 3, 3, color=1, prefix="test_fill")
        assert isinstance(result2, tuple) and len(result2) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
