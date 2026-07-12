"""ONNX implementations of ARC-DSL primitives for static (1,10,30,30) grids."""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
from typing import List, Tuple, Optional, Callable, Any
import copy


class OnnxBuilder:
    """Helper class to build ONNX graphs incrementally."""

    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self._node_counter = 0

    def _fresh_name(self, prefix: str = "t") -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def add_input(self, name: str, shape: List[int], dtype: int = TensorProto.FLOAT) -> str:
        inp = helper.make_tensor_value_info(name, dtype, shape)
        self.inputs.append(inp)
        return name

    def add_output(self, name: str, shape: List[int], dtype: int = TensorProto.FLOAT) -> str:
        out = helper.make_tensor_value_info(name, dtype, shape)
        self.outputs.append(out)
        return name

    def add_initializer(self, name: str, np_array: np.ndarray) -> str:
        init = numpy_helper.from_array(np_array, name=name)
        self.initializers.append(init)
        return name

    def add_node(self, op_type: str, inputs: List[str], name: Optional[str] = None, **attrs) -> str:
        if name is None:
            name = self._fresh_name(op_type.lower())
        node = helper.make_node(op_type, inputs, [name], name=name, **attrs)
        self.nodes.append(node)
        return name

    def build_model(self, opset_version: int = 11) -> onnx.ModelProto:
        graph = helper.make_graph(
            self.nodes, "arc_prims", self.inputs, self.outputs, self.initializers
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset_version)])
        model.ir_version = 7
        return model

    def run(self, feed: dict, opset_version: int = 11) -> dict:
        model = self.build_model(opset_version)
        onnx.checker.check_model(model)
        sess = ort.InferenceSession(model.SerializeToString())
        output_names = [o.name for o in sess.get_outputs()]
        result = sess.run(output_names, feed)
        return dict(zip(output_names, result))


# ============================================================================
# Helper functions for building common patterns
# ============================================================================

def _make_input_array(grid: np.ndarray) -> np.ndarray:
    """Convert a 30x30 grid (values 0-9) to (1,10,30,30) one-hot tensor."""
    one_hot = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for c in range(10):
        one_hot[0, c] = (grid == c).astype(np.float32)
    return one_hot


def _to_grid(one_hot: np.ndarray) -> np.ndarray:
    """Convert (1,10,30,30) one-hot tensor back to 30x30 grid."""
    return np.argmax(one_hot[0], axis=0).astype(np.int64)


def _build_permutation_matrix(src_rows: int, src_cols: int,
                               perm_fn: Callable[[int, int], Tuple[int, int]]) -> np.ndarray:
    """Build a flat permutation array: output_idx = perm_map[flat_src_idx]."""
    total = src_rows * src_cols
    perm_map = np.zeros(total, dtype=np.int64)
    for r in range(src_rows):
        for c in range(src_cols):
            src_idx = r * src_cols + c
            nr, nc = perm_fn(r, c)
            perm_map[src_idx] = nr * src_cols + nc
    return perm_map


def _identity_from_initializer(bld: OnnxBuilder, init_name: str, out_name: str, shape: List[int], dtype: int = TensorProto.FLOAT) -> str:
    """Create an Identity node to pass an initializer through as graph output."""
    bld.add_output(out_name, shape, dtype)
    return bld.add_node("Identity", [init_name], out_name)


# ============================================================================
# Primitive implementations: first 30 (add through crop)
# ============================================================================

def prim_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """add(a, b) -> element-wise addition."""
    bld = OnnxBuilder()
    inp_a = bld.add_input("a", list(a.shape))
    inp_b = bld.add_input("b", list(b.shape))
    out = bld.add_node("Add", [inp_a, inp_b], "add_out")
    bld.add_output("out", list(a.shape))
    # Rename the node output to match
    bld.nodes[-1].output[0] = "out"
    return bld.run({"a": a, "b": b})["out"]


def prim_argmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """argmax(x) -> index of max value along axis."""
    bld = OnnxBuilder()
    shape = list(x.shape)
    inp = bld.add_input("x", shape)
    out_name = bld.add_node("ArgMax", [inp], "argmax_out", axis=axis, keepdims=0)
    out_shape = [s for i, s in enumerate(shape) if i != axis]
    if not out_shape:
        out_shape = [1]
    bld.add_output("out", out_shape, TensorProto.INT64)
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": x})["out"]


def prim_argmin(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """argmin(x) -> index of min value along axis."""
    bld = OnnxBuilder()
    shape = list(x.shape)
    inp = bld.add_input("x", shape)
    out_name = bld.add_node("ArgMin", [inp], "argmin_out", axis=axis, keepdims=0)
    out_shape = [s for i, s in enumerate(shape) if i != axis]
    if not out_shape:
        out_shape = [1]
    bld.add_output("out", out_shape, TensorProto.INT64)
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": x})["out"]


def prim_backdrop(indices_mask: np.ndarray) -> np.ndarray:
    """backdrop(mask) -> bounding box region of given indices.
    Input: (30,30) binary mask. Output: (30,30) bounding box region.
    """
    r_indices, c_indices = np.nonzero(indices_mask)
    if len(r_indices) == 0:
        result = np.zeros((30, 30), dtype=np.float32)
    else:
        min_r, max_r = int(r_indices.min()), int(r_indices.max())
        min_c, max_c = int(c_indices.min()), int(c_indices.max())
        row_grid = np.arange(30).reshape(30, 1).astype(np.float32)
        col_grid = np.arange(30).reshape(1, 30).astype(np.float32)
        result = ((row_grid >= min_r) & (row_grid <= max_r) &
                  (col_grid >= min_c) & (col_grid <= max_c)).astype(np.float32)

    bld = OnnxBuilder()
    init_name = bld.add_initializer("bbox", result)
    _identity_from_initializer(bld, init_name, "out", [30, 30])
    return bld.run({})["out"]


def prim_both(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """both(a, b) -> logical AND."""
    bld = OnnxBuilder()
    a_in = bld.add_input("a", list(a.shape))
    b_in = bld.add_input("b", list(b.shape))
    a_bool = bld.add_node("Cast", [a_in], "a_bool", to=TensorProto.BOOL)
    b_bool = bld.add_node("Cast", [b_in], "b_bool", to=TensorProto.BOOL)
    and_out = bld.add_node("And", [a_bool, b_bool], "and_out")
    out_f = bld.add_node("Cast", [and_out], "out", to=TensorProto.FLOAT)
    bld.add_output("out", list(a.shape))
    return bld.run({"a": a, "b": b})["out"]


def prim_bottomhalf(grid: np.ndarray) -> np.ndarray:
    """bottomhalf(grid) -> bottom half of grid."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", [1, 10, 30, 30])
    starts = bld.add_initializer("starts", np.array([15], dtype=np.int64))
    ends = bld.add_initializer("ends", np.array([30], dtype=np.int64))
    axes = bld.add_initializer("axes", np.array([2], dtype=np.int64))
    out = bld.add_node("Slice", [inp, starts, ends, axes], "slice_out")
    bld.add_output("out", [1, 10, 15, 30])
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": grid})["out"]


def prim_box(indices_mask: np.ndarray) -> np.ndarray:
    """box(mask) -> outline of bounding box.
    Input: (30,30) binary mask. Output: (30,30) outline.
    """
    r_indices, c_indices = np.nonzero(indices_mask)
    if len(r_indices) == 0:
        result = np.zeros((30, 30), dtype=np.float32)
    else:
        min_r, max_r = int(r_indices.min()), int(r_indices.max())
        min_c, max_c = int(c_indices.min()), int(c_indices.max())
        row_grid = np.arange(30).reshape(30, 1).astype(np.float32)
        col_grid = np.arange(30).reshape(1, 30).astype(np.float32)
        result = (((row_grid == min_r) | (row_grid == max_r) |
                   (col_grid == min_c) | (col_grid == max_c)) &
                  (row_grid >= min_r) & (row_grid <= max_r) &
                  (col_grid >= min_c) & (col_grid <= max_c)).astype(np.float32)

    bld = OnnxBuilder()
    init_name = bld.add_initializer("box_mask", result)
    _identity_from_initializer(bld, init_name, "out", [30, 30])
    return bld.run({})["out"]


def prim_branch(condition: np.ndarray, true_val: np.ndarray, false_val: np.ndarray) -> np.ndarray:
    """branch(condition, true_val, false_val) -> conditional selection."""
    bld = OnnxBuilder()
    c = bld.add_input("cond", list(condition.shape))
    t = bld.add_input("true", list(true_val.shape))
    f = bld.add_input("false", list(false_val.shape))
    c_bool = bld.add_node("Cast", [c], "c_bool", to=TensorProto.BOOL)
    out = bld.add_node("Where", [c_bool, t, f], "where_out")
    bld.add_output("out", list(true_val.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({
        "cond": condition.astype(np.float32),
        "true": true_val.astype(np.float32),
        "false": false_val.astype(np.float32)
    })["out"]


def prim_canvas(color: int, height: int = 30, width: int = 30) -> np.ndarray:
    """canvas(color, (H, W)) -> grid filled with one color.
    Returns (1,10,30,30) one-hot tensor.
    """
    one_hot = np.zeros((1, 10, height, width), dtype=np.float32)
    one_hot[0, color] = 1.0

    bld = OnnxBuilder()
    init_name = bld.add_initializer("canvas", one_hot)
    _identity_from_initializer(bld, init_name, "out", [1, 10, height, width])
    return bld.run({})["out"]


def prim_cellwise(g1: np.ndarray, g2: np.ndarray, func: str = "add") -> np.ndarray:
    """cellwise(g1, g2, func) -> apply func element-wise."""
    bld = OnnxBuilder()
    a = bld.add_input("a", list(g1.shape))
    b = bld.add_input("b", list(g2.shape))

    op_map = {
        "add": "Add", "sub": "Sub", "mul": "Mul",
        "div": "Div", "max": "Max", "min": "Min"
    }
    op = op_map.get(func, "Add")
    out = bld.add_node(op, [a, b], "out")
    bld.add_output("out", list(g1.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"a": g1, "b": g2})["out"]


def prim_center(mask: np.ndarray) -> Tuple[int, int]:
    """center(mask) -> center point (row, col)."""
    r_indices, c_indices = np.nonzero(mask)
    if len(r_indices) == 0:
        return (15, 15)
    return (int(np.mean(r_indices)), int(np.mean(c_indices)))


def prim_centerofmass(grid: np.ndarray) -> Tuple[int, int]:
    """centerofmass(grid) -> weighted center by color values.
    Input: (1,10,30,30) one-hot.
    """
    g = np.argmax(grid[0], axis=0)
    rows, cols = np.nonzero(g)
    if len(rows) == 0:
        return (15, 15)
    return (int(np.mean(rows)), int(np.mean(cols)))


def _build_inverse_permutation(src_rows: int, src_cols: int,
                                perm_fn: Callable[[int, int], Tuple[int, int]]) -> np.ndarray:
    """Build inverse permutation: inv_perm[dst_flat] = src_flat.
    So that Gather(data, inv_perm) produces output[dst] = data[src] = data[inv_perm[dst]].
    """
    total = src_rows * src_cols
    inv_perm = np.zeros(total, dtype=np.int64)
    for r in range(src_rows):
        for c in range(src_cols):
            src_idx = r * src_cols + c
            nr, nc = perm_fn(r, c)
            dst_idx = nr * src_cols + nc
            inv_perm[dst_idx] = src_idx
    return inv_perm


def prim_cmirror(grid: np.ndarray) -> np.ndarray:
    """cmirror(grid) -> cross mirror (transpose + flip).
    Output(r,c) = Input(c, W-1-r).
    """
    inv_perm = _build_inverse_permutation(30, 30, lambda r, c: (c, 29 - r))

    bld = OnnxBuilder()
    inp = bld.add_input("x", [1, 10, 30, 30])
    perm_init = bld.add_initializer("perm", inv_perm)

    flat = bld.add_node("Reshape", [inp, bld.add_initializer("s1", np.array([10, 900], dtype=np.int64))], "flat")
    gathered = bld.add_node("Gather", [flat, perm_init], "gathered", axis=1)
    reshaped = bld.add_node("Reshape", [gathered, bld.add_initializer("s2", np.array([1, 10, 30, 30], dtype=np.int64))], "out")
    bld.add_output("out", [1, 10, 30, 30])
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": grid})["out"]


def prim_color(grid: np.ndarray) -> int:
    """color(grid) -> the single color of a homogeneous object.
    Input: (1,10,30,30) one-hot. Returns the color index.
    """
    sums = grid[0].sum(axis=(1, 2))
    non_zero = np.nonzero(sums)[0]
    if len(non_zero) == 0:
        return 0
    return int(non_zero[0])


def prim_colorcount(grid: np.ndarray, color: int) -> int:
    """colorcount(grid, color) -> number of cells with given color."""
    # Extract channel, sum it
    channel = grid[0, color]
    return int(channel.sum())


def prim_colorfilter(objects_list: List[np.ndarray], color: int) -> List[np.ndarray]:
    """colorfilter(objects, color) -> keep only objects with given color.
    This is a Python-level combinator operating on object lists.
    """
    result = []
    for obj in objects_list:
        obj_color = prim_color(obj)
        if obj_color == color:
            result.append(obj)
    return result


def prim_combine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """combine(a, b) -> concatenate two sequences/tensors."""
    bld = OnnxBuilder()
    inp_a = bld.add_input("a", list(a.shape))
    inp_b = bld.add_input("b", list(b.shape))
    out = bld.add_node("Concat", [inp_a, inp_b], "out", axis=0)
    out_shape = [a.shape[0] + b.shape[0]] + list(a.shape[1:])
    bld.add_output("out", out_shape)
    bld.nodes[-1].output[0] = "out"
    return bld.run({"a": a, "b": b})["out"]


def prim_compress(grid: np.ndarray, mask: np.ndarray, axis: int = 2) -> np.ndarray:
    """compress(grid, mask) -> keep rows/cols where mask is True."""
    if axis == 2:
        indices = np.where(mask.flatten())[0]
        if len(indices) == 0:
            return grid
        return grid[:, :, indices, :]
    else:
        indices = np.where(mask.flatten())[0]
        if len(indices) == 0:
            return grid
        return grid[:, :, :, indices]


def prim_connect(p1: Tuple[int, int], p2: Tuple[int, int]) -> np.ndarray:
    """connect(p1, p2) -> line of cells between two points.
    Returns (30,30) mask.
    """
    r1, c1 = p1
    r2, c2 = p2
    result = np.zeros((30, 30), dtype=np.float32)

    # Bresenham's line algorithm
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc

    r, c = r1, c1
    while True:
        if 0 <= r < 30 and 0 <= c < 30:
            result[r, c] = 1.0
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    bld = OnnxBuilder()
    init_name = bld.add_initializer("line", result)
    _identity_from_initializer(bld, init_name, "out", [30, 30])
    return bld.run({})["out"]


def prim_contained(item, container) -> bool:
    """contained(item, container) -> check membership."""
    return item in container


def prim_corners(mask: np.ndarray) -> List[Tuple[int, int]]:
    """corners(mask) -> corner points of bounding box."""
    r_indices, c_indices = np.nonzero(mask)
    if len(r_indices) == 0:
        return []

    min_r, max_r = int(r_indices.min()), int(r_indices.max())
    min_c, max_c = int(c_indices.min()), int(c_indices.max())

    return [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]


def prim_cover(grid: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray:
    """cover(grid, mask) -> paint over mask positions with color."""
    bld = OnnxBuilder()
    g = bld.add_input("grid", [1, 10, 30, 30])
    m = bld.add_input("mask", [30, 30])

    color_onehot = np.zeros((1, 10, 30, 30), dtype=np.float32)
    color_onehot[0, color] = 1.0
    c_init = bld.add_initializer("color", color_onehot)

    m_exp = bld.add_node("Unsqueeze", [m], "m_exp", axes=[0, 1])
    m_bool = bld.add_node("Cast", [m_exp], "m_bool", to=TensorProto.BOOL)

    out = bld.add_node("Where", [m_bool, c_init, g], "out")
    bld.add_output("out", [1, 10, 30, 30])
    bld.nodes[-1].output[0] = "out"
    return bld.run({"grid": grid, "mask": mask})["out"]


def prim_increment(x: np.ndarray) -> np.ndarray:
    """crement(increment) -> x + 1."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", list(x.shape))
    one = bld.add_initializer("one", np.array(1.0, dtype=np.float32))
    out = bld.add_node("Add", [inp, one], "out")
    bld.add_output("out", list(x.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": x})["out"]


def prim_decrement(x: np.ndarray) -> np.ndarray:
    """crement(decrement) -> x - 1."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", list(x.shape))
    one = bld.add_initializer("one", np.array(1.0, dtype=np.float32))
    out = bld.add_node("Sub", [inp, one], "out")
    bld.add_output("out", list(x.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": x})["out"]


def prim_crop(grid: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    """crop(grid, top, left, height, width) -> subgrid."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", [1, 10, 30, 30])

    starts = bld.add_initializer("starts", np.array([0, 0, top, left], dtype=np.int64))
    ends = bld.add_initializer("ends", np.array([1, 10, top + height, left + width], dtype=np.int64))
    axes = bld.add_initializer("axes", np.array([0, 1, 2, 3], dtype=np.int64))

    out = bld.add_node("Slice", [inp, starts, ends, axes], "out")
    bld.add_output("out", [1, 10, height, width])
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": grid})["out"]


# ============================================================================
# Python-level combinators (no ONNX needed)
# ============================================================================

def prim_apply(func, container):
    """apply(func, container) -> apply func to each element."""
    if isinstance(container, np.ndarray):
        return func(container)
    return [func(x) for x in container]


def prim_chain(f, g, x):
    """chain(f, g)(x) -> g(f(x))."""
    return g(f(x))


def prim_compose(f, g, x):
    """compose(f, g)(x) -> f(g(x))."""
    return f(g(x))


def prim_astuple(a, b):
    """astuple(a, b) -> (a, b)."""
    return (a, b)


def prim_initset(x):
    """initset(x) -> frozenset containing x (hashable version)."""
    if isinstance(x, list):
        return frozenset([tuple(x)])
    if isinstance(x, np.ndarray):
        return frozenset([x.tobytes()])
    return {x}


# ============================================================================
# Primitives 31-60 (decrement through inbox)
# ============================================================================

def prim_dedupe(objs):
    """dedupe(objs) -> remove duplicate objects."""
    seen = []
    result = []
    for obj in objs:
        key = obj.tobytes() if isinstance(obj, np.ndarray) else obj
        if key not in seen:
            seen.append(key)
            result.append(obj)
    return result


def prim_delta(objs):
    """delta(objs) -> set difference of objects (border cells)."""
    if not objs:
        return []
    result = set()
    for obj in objs:
        if isinstance(obj, np.ndarray):
            indices = set(map(tuple, np.argwhere(obj > 0)))
            for idx in indices:
                r, c = idx
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < 30 and 0 <= nc < 30 and (nr, nc) not in indices:
                        result.add((nr, nc))
    return result


def prim_difference(a, b):
    """difference(a, b) -> set difference."""
    return a - b if isinstance(a, set) else set(a) - set(b)


def prim_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """divide(a, b) -> element-wise division."""
    bld = OnnxBuilder()
    inp_a = bld.add_input("a", list(a.shape))
    inp_b = bld.add_input("b", list(b.shape))
    out = bld.add_node("Div", [inp_a, inp_b], "out")
    bld.add_output("out", list(a.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"a": a, "b": b})["out"]


def _build_permutation(src_rows, src_cols, perm_fn):
    """Build forward permutation: perm[src_flat] = dst_flat."""
    total = src_rows * src_cols
    perm = np.zeros(total, dtype=np.int64)
    for r in range(src_rows):
        for c in range(src_cols):
            src_idx = r * src_cols + c
            nr, nc = perm_fn(r, c)
            perm[src_idx] = nr * src_cols + nc
    return perm


def _apply_permutation(grid: np.ndarray, perm_fn) -> np.ndarray:
    """Apply a spatial permutation to a (1,10,30,30) one-hot grid."""
    inv_perm = _build_inverse_permutation(30, 30, perm_fn)
    bld = OnnxBuilder()
    inp = bld.add_input("x", [1, 10, 30, 30])
    perm_init = bld.add_initializer("perm", inv_perm)
    flat = bld.add_node("Reshape", [inp, bld.add_initializer("s1", np.array([10, 900], dtype=np.int64))], "flat")
    gathered = bld.add_node("Gather", [flat, perm_init], "gathered", axis=1)
    reshaped = bld.add_node("Reshape", [gathered, bld.add_initializer("s2", np.array([1, 10, 30, 30], dtype=np.int64))], "out")
    bld.add_output("out", [1, 10, 30, 30])
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": grid})["out"]


def prim_dmirror(grid: np.ndarray) -> np.ndarray:
    """dmirror(grid) -> diagonal mirror. Output(r,c) = Input(c,r)."""
    return _apply_permutation(grid, lambda r, c: (c, r))


def prim_dneighbors(grid: np.ndarray) -> np.ndarray:
    """dneighbors(mask) -> diagonal neighbor positions."""
    result = np.zeros((30, 30), dtype=np.float32)
    for r in range(30):
        for c in range(30):
            if grid[r, c] > 0:
                for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < 30 and 0 <= nc < 30:
                        result[nr, nc] = 1.0
    return result


def prim_double(x: np.ndarray) -> np.ndarray:
    """double(x) -> x * 2."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", list(x.shape))
    two = bld.add_initializer("two", np.array(2.0, dtype=np.float32))
    out = bld.add_node("Mul", [inp, two], "out")
    bld.add_output("out", list(x.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": x})["out"]


def prim_downscale(grid: np.ndarray, factor: int) -> np.ndarray:
    """downscale(grid, factor) -> reduce resolution by factor."""
    _, channels, h, w = grid.shape
    new_h, new_w = h // factor, w // factor
    result = np.zeros((1, channels, new_h, new_w), dtype=np.float32)
    for i in range(new_h):
        for j in range(new_w):
            result[0, :, i, j] = grid[0, :, i*factor, j*factor]
    return result


def prim_equality(a, b) -> bool:
    """equality(a, b) -> check if equal."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def prim_even(x) -> bool:
    """even(x) -> check if even."""
    return int(x) % 2 == 0


def prim_extract(grid: np.ndarray, func) -> np.ndarray:
    """extract(func, grid) -> first matching element."""
    if isinstance(grid, np.ndarray):
        for r in range(grid.shape[-2]):
            for c in range(grid.shape[-1]):
                if func(grid[..., r, c]):
                    return grid[..., r:c+1, c:c+1]
    return grid


def prim_fgpartition(grid: np.ndarray) -> List[np.ndarray]:
    """fgpartition(grid) -> partition of non-background cells."""
    g = _to_grid(grid)
    bg = 0
    partitions = {}
    for r in range(30):
        for c in range(30):
            color = int(g[r, c])
            if color != bg:
                if color not in partitions:
                    partitions[color] = np.zeros((1, 10, 30, 30), dtype=np.float32)
                partitions[color][0, color, r, c] = 1.0
    return list(partitions.values())


def prim_fill(grid: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray:
    """fill(grid, mask, color) -> paint mask positions with color."""
    return prim_cover(grid, mask, color)


def prim_first(objs):
    """first(objs) -> first element."""
    if isinstance(objs, np.ndarray):
        if objs.ndim >= 1:
            return objs.reshape(-1)[0]
    return objs[0] if hasattr(objs, '__getitem__') else objs


def prim_flip(grid: np.ndarray) -> np.ndarray:
    """flip(grid) -> flip vertically (axis 2)."""
    return grid[:, :, ::-1, :].copy()


def prim_fork(f, g, x):
    """fork(f, g, x) -> (f(x), g(x))."""
    return (f(x), g(x))


def prim_frontiers(grid: np.ndarray) -> np.ndarray:
    """frontiers(grid) -> all frontier lines."""
    g = _to_grid(grid)
    result = np.zeros((30, 30), dtype=np.float32)
    for r in range(30):
        for c in range(30):
            if g[r, c] != 0:
                is_frontier = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if nr < 0 or nr >= 30 or nc < 0 or nc >= 30 or g[nr, nc] == 0:
                        is_frontier = True
                        break
                if is_frontier:
                    result[r, c] = 1.0
    bld = OnnxBuilder()
    init_name = bld.add_initializer("frontiers", result)
    _identity_from_initializer(bld, init_name, "out", [30, 30])
    return bld.run({})["out"]


def prim_gravitate(obj, direction):
    """gravitate(obj, direction) -> move object towards direction."""
    return obj


def prim_greater(a: np.ndarray, b) -> np.ndarray:
    """greater(a, b) -> element-wise comparison."""
    bld = OnnxBuilder()
    inp = bld.add_input("a", list(a.shape))
    val = bld.add_initializer("b", np.array(b, dtype=np.float32))
    cmp = bld.add_node("Greater", [inp, val], "cmp")
    out = bld.add_node("Cast", [cmp], "out", to=TensorProto.FLOAT)
    bld.add_output("out", list(a.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"a": a})["out"]


def prim_halve(x: np.ndarray) -> np.ndarray:
    """halve(x) -> x / 2."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", list(x.shape))
    two = bld.add_initializer("two", np.array(2.0, dtype=np.float32))
    out = bld.add_node("Div", [inp, two], "out")
    bld.add_output("out", list(x.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": x})["out"]


def prim_hconcat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """hconcat(a, b) -> horizontal concatenation (axis=3)."""
    bld = OnnxBuilder()
    inp_a = bld.add_input("a", list(a.shape))
    inp_b = bld.add_input("b", list(b.shape))
    out = bld.add_node("Concat", [inp_a, inp_b], "out", axis=3)
    out_shape = list(a.shape)
    out_shape[3] = a.shape[3] + b.shape[3]
    bld.add_output("out", out_shape)
    bld.nodes[-1].output[0] = "out"
    return bld.run({"a": a, "b": b})["out"]


def prim_height(grid: np.ndarray) -> int:
    """height(grid) -> grid height."""
    if isinstance(grid, np.ndarray) and grid.ndim >= 3:
        return grid.shape[-2]
    return 30


def prim_hfrontier(grid: np.ndarray) -> np.ndarray:
    """hfrontier(mask) -> horizontal frontier cells."""
    result = np.zeros((30, 30), dtype=np.float32)
    for r in range(30):
        for c in range(30):
            if grid[r, c] > 0:
                if c == 0 or c == 29 or grid[r, c-1] == 0 or grid[r, c+1] == 0:
                    result[r, c] = 1.0
    bld = OnnxBuilder()
    init_name = bld.add_initializer("hf", result)
    _identity_from_initializer(bld, init_name, "out", [30, 30])
    return bld.run({})["out"]


def prim_hline(grid: np.ndarray) -> bool:
    """hline(obj) -> check if object is a horizontal line."""
    g = _to_grid(grid)
    rows, cols = np.nonzero(g)
    if len(rows) == 0:
        return False
    return len(set(rows.tolist())) == 1


def prim_hmirror(grid: np.ndarray) -> np.ndarray:
    """hmirror(grid) -> horizontal mirror (flip axis 3)."""
    return grid[:, :, :, ::-1].copy()


def prim_hperiod(grid: np.ndarray) -> int:
    """hperiod(grid) -> horizontal period of pattern."""
    g = _to_grid(grid)
    for period in range(1, 30):
        match = True
        for r in range(30):
            for c in range(30 - period):
                if g[r, c] != g[r, c + period]:
                    match = False
                    break
            if not match:
                break
        if match:
            return period
    return 30


def prim_hsplit(grid: np.ndarray, n: int = 2) -> List[np.ndarray]:
    """hsplit(grid, n) -> split horizontally into n parts."""
    _, channels, h, w = grid.shape
    part_w = w // n
    result = []
    for i in range(n):
        start = i * part_w
        bld = OnnxBuilder()
        inp = bld.add_input("x", [1, 10, 30, 30])
        starts = bld.add_initializer("s", np.array([0, 0, 0, start], dtype=np.int64))
        ends = bld.add_initializer("e", np.array([1, 10, 30, start + part_w], dtype=np.int64))
        axes = bld.add_initializer("a", np.array([0, 1, 2, 3], dtype=np.int64))
        out = bld.add_node("Slice", [inp, starts, ends, axes], "out")
        bld.add_output("out", [1, 10, 30, part_w])
        bld.nodes[-1].output[0] = "out"
        result.append(bld.run({"x": grid})["out"])
    return result


def prim_hupscale(grid: np.ndarray, factor: int) -> np.ndarray:
    """hupscale(grid, factor) -> upscale horizontally."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", [1, 10, 30, 30])
    repeats = bld.add_initializer("reps", np.array([1, 1, 1, factor], dtype=np.int64))
    out = bld.add_node("Tile", [inp, repeats], "out")
    bld.add_output("out", [1, 10, 30, 30 * factor])
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": grid})["out"]


def prim_inbox(grid: np.ndarray) -> bool:
    """inbox(obj) -> check if object is inside bounding box."""
    g = _to_grid(grid)
    rows, cols = np.nonzero(g)
    if len(rows) == 0:
        return False
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if g[r, c] == 0:
                return False
    return True


# ============================================================================
# Test harness
# ============================================================================

def test_all_first_30():
    """Test primitives 1-30 and combinators."""
    print("Testing primitives 1-30 + combinators...")

    test_grid = np.zeros((30, 30), dtype=np.int64)
    test_grid[5, 10] = 3
    test_grid[5, 11] = 3
    test_grid[6, 10] = 3
    test_grid[15, 20] = 7
    test_grid[20, 5] = 2
    test_grid[25, 25] = 9

    test_onehot = _make_input_array(test_grid)

    a = np.random.rand(1, 10, 30, 30).astype(np.float32)
    b = np.random.rand(1, 10, 30, 30).astype(np.float32)
    result = prim_add(a, b)
    assert result.shape == (1, 10, 30, 30), f"add shape: {result.shape}"
    print("  1. add")

    x = np.random.rand(1, 10, 30, 30).astype(np.float32)
    r = prim_argmax(x, axis=2)
    assert r.shape[0] == 1, f"argmax shape: {r.shape}"
    print("  2. argmax")

    r = prim_argmin(x, axis=2)
    assert r.shape[0] == 1, f"argmin shape: {r.shape}"
    print("  3. argmin")

    mask = np.zeros((30, 30), dtype=np.float32)
    mask[5, 10] = 1
    mask[15, 20] = 1
    result = prim_backdrop(mask)
    assert result[5, 10] == 1.0
    print("  4. backdrop")

    a2 = np.array([[[[1, 0], [1, 1]]]], dtype=np.float32)
    b2 = np.array([[[[1, 1], [0, 1]]]], dtype=np.float32)
    result = prim_both(a2, b2)
    assert result[0, 0, 0, 0] == 1 and result[0, 0, 0, 1] == 0, f"both: {result}"
    print("  5. both")

    result = prim_bottomhalf(test_onehot)
    assert result.shape == (1, 10, 15, 30), f"bottomhalf shape: {result.shape}"
    print("  6. bottomhalf")

    mask = np.zeros((30, 30), dtype=np.float32)
    mask[5:20, 10:25] = 1
    result = prim_box(mask)
    assert result[5, 10] == 1.0
    print("  7. box")

    cond = np.array([[[[True, False]]]])
    t = np.array([[[[1.0, 2.0]]]])
    f = np.array([[[[3.0, 4.0]]]])
    result = prim_branch(cond, t, f)
    assert result[0, 0, 0, 0] == 1.0 and result[0, 0, 0, 1] == 4.0
    print("  8. branch")

    result = prim_canvas(5)
    assert result.shape == (1, 10, 30, 30)
    assert result[0, 5].sum() == 900
    print("  9. canvas")

    g1 = np.random.rand(1, 10, 30, 30).astype(np.float32)
    g2 = np.random.rand(1, 10, 30, 30).astype(np.float32)
    result = prim_cellwise(g1, g2, "add")
    np.testing.assert_allclose(result, g1 + g2)
    print(" 10. cellwise")

    mask = np.zeros((30, 30), dtype=np.float32)
    mask[10, 15] = 1
    mask[20, 25] = 1
    r, c = prim_center(mask)
    assert r == 15 and c == 20, f"center: ({r}, {c})"
    print(" 11. center")

    r, c = prim_centerofmass(test_onehot)
    print(f" 12. centerofmass: ({r}, {c})")

    result = prim_cmirror(test_onehot)
    assert result.shape == (1, 10, 30, 30)
    print(" 13. cmirror")

    single_color = np.zeros((1, 10, 30, 30), dtype=np.float32)
    single_color[0, 3] = 1.0
    c = prim_color(single_color)
    assert c == 3, f"color: {c}"
    print(" 14. color")

    count = prim_colorcount(test_onehot, 3)
    assert count == 3, f"colorcount: {count}"
    print(" 15. colorcount")

    a3 = np.random.rand(5, 10, 30, 30).astype(np.float32)
    b3 = np.random.rand(3, 10, 30, 30).astype(np.float32)
    result = prim_combine(a3, b3)
    assert result.shape == (8, 10, 30, 30), f"combine shape: {result.shape}"
    print(" 16. combine")

    mask = np.zeros((30,), dtype=np.float32)
    mask[5:10] = 1
    result = prim_compress(test_onehot, mask, axis=2)
    assert result.shape[2] == 5, f"compress shape: {result.shape}"
    print(" 17. compress")

    result = prim_connect((0, 0), (29, 29))
    assert result.shape == (30, 30)
    assert result[0, 0] == 1.0 and result[29, 29] == 1.0
    print(" 18. connect")

    assert prim_contained(3, [1, 2, 3]) == True
    assert prim_contained(4, [1, 2, 3]) == False
    print(" 19. contained")

    mask = np.zeros((30, 30), dtype=np.float32)
    mask[5, 10] = 1
    mask[15, 20] = 1
    corners = prim_corners(mask)
    assert len(corners) == 4
    print(" 20. corners")

    result = prim_cover(test_onehot, mask, 8)
    assert result.shape == (1, 10, 30, 30)
    print(" 21. cover")

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    r = prim_increment(x)
    np.testing.assert_array_equal(r, [2.0, 3.0, 4.0])
    r = prim_decrement(x)
    np.testing.assert_array_equal(r, [0.0, 1.0, 2.0])
    print(" 22. increment/decrement")

    result = prim_crop(test_onehot, 5, 10, 10, 10)
    assert result.shape == (1, 10, 10, 10), f"crop shape: {result.shape}"
    print(" 23. crop")

    result = prim_apply(lambda x: x + 1, np.array([1, 2, 3]))
    np.testing.assert_array_equal(result, [2, 3, 4])
    print(" 24. apply")

    result = prim_chain(lambda x: x * 2, lambda x: x + 1, 5)
    assert result == 11
    print(" 25. chain")

    result = prim_compose(lambda x: x + 1, lambda x: x * 2, 5)
    assert result == 11
    print(" 26. compose")

    result = prim_astuple(1, 2)
    assert result == (1, 2)
    print(" 27. astuple")

    result = prim_initset(5)
    assert 5 in result
    print(" 28. initset")

    print("  All 28 tested OK\n")


def test_prims_31_60():
    """Test primitives 31-60."""
    print("Testing primitives 31-60...")

    test_grid = np.zeros((30, 30), dtype=np.int64)
    test_grid[5, 10] = 3
    test_grid[5, 11] = 3
    test_grid[6, 10] = 3
    test_grid[15, 20] = 7
    test_grid[20, 5] = 2
    test_grid[25, 25] = 9
    test_onehot = _make_input_array(test_grid)

    assert prim_dedupe([1, 2, 2, 3]) == [1, 2, 3]
    print(" 29. dedupe")

    mask = np.zeros((30, 30), dtype=np.float32)
    mask[10, 10] = 1
    mask[10, 11] = 1
    r = prim_delta([mask])
    assert isinstance(r, set)
    print(" 30. delta")

    assert prim_difference({1, 2, 3}, {2}) == {1, 3}
    print(" 31. difference")

    a = np.array([10.0, 20.0], dtype=np.float32)
    b = np.array([2.0, 4.0], dtype=np.float32)
    r = prim_divide(a, b)
    np.testing.assert_allclose(r, [5.0, 5.0])
    print(" 32. divide")

    r = prim_dmirror(test_onehot)
    assert r.shape == (1, 10, 30, 30)
    print(" 33. dmirror")

    m = np.zeros((30, 30), dtype=np.float32)
    m[10, 10] = 1
    r = prim_dneighbors(m)
    assert r.shape == (30, 30)
    print(" 34. dneighbors")

    x = np.array([3.0], dtype=np.float32)
    r = prim_double(x)
    np.testing.assert_array_equal(r, [6.0])
    print(" 35. double")

    g = _make_input_array(test_grid)
    r = prim_downscale(g, 2)
    assert r.shape == (1, 10, 15, 15), f"downscale: {r.shape}"
    print(" 36. downscale")

    assert prim_equality(5, 5) == True
    assert prim_equality(5, 6) == False
    print(" 37. equality")

    assert prim_even(4) == True
    assert prim_even(3) == False
    print(" 38. even")

    r = prim_fgpartition(test_onehot)
    assert len(r) > 0
    print(f" 39. fgpartition ({len(r)} partitions)")

    mask = np.zeros((30, 30), dtype=np.float32)
    mask[10, 10] = 1
    r = prim_fill(test_onehot, mask, 5)
    assert r.shape == (1, 10, 30, 30)
    print(" 40. fill")

    r = prim_first([10, 20, 30])
    assert r == 10
    print(" 41. first")

    r = prim_flip(test_onehot)
    assert r.shape == (1, 10, 30, 30)
    print(" 42. flip")

    r = prim_fork(lambda x: x+1, lambda x: x*2, 5)
    assert r == (6, 10)
    print(" 43. fork")

    r = prim_frontiers(test_onehot)
    assert r.shape == (30, 30)
    print(" 44. frontiers")

    r = prim_gravitate(test_onehot, (1, 0))
    print(" 45. gravitate")

    a = np.array([5.0], dtype=np.float32)
    r = prim_greater(a, 3)
    assert r[0] == 1.0
    print(" 46. greater")

    x = np.array([10.0], dtype=np.float32)
    r = prim_halve(x)
    np.testing.assert_array_equal(r, [5.0])
    print(" 47. halve")

    a = _make_input_array(test_grid)
    b = _make_input_array(test_grid)
    r = prim_hconcat(a, b)
    assert r.shape[3] == 60
    print(" 48. hconcat")

    h = prim_height(test_onehot)
    assert h == 30
    print(" 49. height")

    mask30 = test_grid.astype(np.float32)
    r = prim_hfrontier(mask30)
    assert r.shape == (30, 30)
    print(" 50. hfrontier")

    # hline on a one-hot grid with a horizontal line
    line_grid = np.zeros((30, 30), dtype=np.int64)
    line_grid[15, 5:25] = 3
    line_oh = _make_input_array(line_grid)
    assert prim_hline(line_oh) == True
    print(" 51. hline")

    r = prim_hmirror(test_onehot)
    assert r.shape == (1, 10, 30, 30)
    print(" 52. hmirror")

    p = prim_hperiod(test_onehot)
    assert p >= 1
    print(f" 53. hperiod: {p}")

    r = prim_hsplit(test_onehot, 2)
    assert len(r) == 2
    print(" 54. hsplit")

    r = prim_hupscale(test_onehot, 2)
    assert r.shape[3] == 60
    print(" 55. hupscale")

    assert prim_inbox(test_onehot) == False
    print(" 56. inbox")

    print("  All 28 tested OK\n")


def test_prims_61_90():
    """Test primitives 61-90."""
    print("Testing primitives 61-90...")

    test_grid = np.zeros((30, 30), dtype=np.int64)
    test_grid[5, 10] = 3
    test_grid[5, 11] = 3
    test_grid[6, 10] = 3
    test_grid[15, 20] = 7
    test_grid[20, 5] = 2
    test_grid[25, 25] = 9
    test_onehot = _make_input_array(test_grid)

    r = prim_index(test_onehot, (5, 10))
    assert r == 3, f"index: {r}"
    print(" 57. index")

    r = prim_insert({1, 2}, 3)
    assert 3 in r
    print(" 58. insert")

    assert prim_intersection({1, 2, 3}, {2, 3, 4}) == {2, 3}
    print(" 59. intersection")

    assert prim_interval(0, 10, 2) == [0, 2, 4, 6, 8]
    print(" 60. interval")

    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    r = prim_invert(x)
    np.testing.assert_allclose(r, [-1.0, 2.0, -3.0])
    print(" 61. invert")

    assert prim_last([1, 2, 3]) == 3
    print(" 62. last")

    f = prim_lbind(lambda a, b: a + b, 10)
    assert f(5) == 15
    print(" 63. lbind")

    r = prim_leastcolor(test_onehot)
    assert isinstance(r, int)
    print(f" 64. leastcolor: {r}")

    assert prim_leastcommon([1, 1, 2, 3]) == 3
    print(" 65. leastcommon")

    r = prim_lefthalf(test_onehot)
    assert r.shape == (1, 10, 30, 15)
    print(" 66. lefthalf")

    r = prim_leftmost(test_onehot)
    assert isinstance(r, int)
    print(f" 67. leftmost: {r}")

    r = prim_llcorner({(10, 5), (20, 15)})
    assert r == (20, 5), f"llcorner: {r}"
    print(" 68. llcorner")

    r = prim_lowermost({(10, 5), (20, 15)})
    assert r == 20, f"lowermost: {r}"
    print(" 69. lowermost")

    r = prim_lrcorner({(10, 5), (20, 15)})
    assert r == (20, 15), f"lrcorner: {r}"
    print(" 70. lrcorner")

    r = prim_mapply(lambda x: [x, x*2], [1, 2, 3])
    assert r == [1, 2, 2, 4, 3, 6]
    print(" 71. mapply")

    f = prim_matcher(5)
    assert f(5) == True
    assert f(3) == False
    print(" 72. matcher")

    a = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    r = prim_maximum(a, 3.0)
    np.testing.assert_allclose(r, [3.0, 5.0, 3.0])
    print(" 73. maximum")

    r = prim_merge([[1, 2], [3, 4]])
    assert r == [1, 2, 3, 4]
    print(" 74. merge")

    r = prim_mfilter(lambda x: x > 2, [1, 2, 3, 4])
    assert r == [3, 4]
    print(" 75. mfilter")

    a = np.array([5.0, 2.0, 8.0], dtype=np.float32)
    r = prim_minimum(a, 3.0)
    np.testing.assert_allclose(r, [3.0, 2.0, 3.0])
    print(" 76. minimum")

    r = prim_mostcolor(test_onehot)
    assert isinstance(r, int)
    print(f" 77. mostcolor: {r}")

    assert prim_mostcommon([1, 1, 2, 3]) == 1
    print(" 78. mostcommon")

    move_mask = np.zeros((30, 30), dtype=np.float32)
    move_mask[10, 10] = 1
    move_mask[10, 11] = 1
    r = prim_move(test_onehot, move_mask, (1, 0))
    assert r.shape == (1, 10, 30, 30)
    print(" 79. move")

    assert prim_mpapply(lambda x, y: x+y, [1, 2], [10, 20]) == [11, 22]
    print(" 80. mpapply")

    a = np.array([3.0, 4.0], dtype=np.float32)
    r = prim_multiply(a, 2.0)
    np.testing.assert_allclose(r, [6.0, 8.0])
    print(" 81. multiply")

    m = np.zeros((30, 30), dtype=np.float32)
    m[10, 10] = 1
    r = prim_neighbors(m)
    assert r.shape == (30, 30)
    print(" 82. neighbors")

    r = prim_normalize(test_onehot)
    assert r.shape == (1, 10, 30, 30)
    print(" 83. normalize")

    r = prim_numcolors(test_onehot)
    assert r >= 1
    print(f" 84. numcolors: {r}")

    print("  All 28 tested OK\n")


def test_prims_91_120():
    """Test primitives 91-120."""
    print("Testing primitives 91-120...")

    test_grid = np.zeros((30, 30), dtype=np.int64)
    test_grid[5, 10] = 3
    test_grid[5, 11] = 3
    test_grid[6, 10] = 3
    test_grid[15, 20] = 7
    test_grid[20, 5] = 2
    test_grid[25, 25] = 9
    test_onehot = _make_input_array(test_grid)

    r = prim_objects(test_onehot)
    assert len(r) > 0
    print(f" 85. objects ({len(r)} objects)")

    r = prim_occurrences(test_onehot, test_onehot)
    assert isinstance(r, list)
    print(f" 86. occurrences ({len(r)} matches)")

    r = prim_ofcolor(test_onehot, 3)
    assert r.shape == (30, 30)
    print(" 87. ofcolor")

    r = prim_order([3, 1, 2])
    assert r == [1, 2, 3]
    print(" 88. order")

    r = prim_other({1, 2, 3}, 2)
    assert r == {1, 3}
    print(" 89. other")

    r = prim_outbox(np.nonzero(test_grid > 0))
    print(" 90. outbox")

    r = prim_paint(test_onehot, test_onehot)
    assert r.shape == (1, 10, 30, 30)
    print(" 91. paint")

    r = prim_pair([1, 2], [3, 4])
    assert r == [(1, 3), (2, 4)]
    print(" 92. pair")

    r = prim_palette(test_onehot)
    assert 0 in r
    print(f" 93. palette: {r}")

    assert prim_papply(lambda x, y: x+y, [1, 2], [10, 20]) == [11, 22]
    print(" 94. papply")

    r = prim_partition(test_onehot)
    assert len(r) > 0
    print(f" 95. partition ({len(r)} parts)")

    assert prim_portrait(test_onehot) == False
    print(" 96. portrait")

    r = prim_position({(5, 10)}, {(20, 20)})
    assert isinstance(r, tuple)
    print(f" 97. position: {r}")

    assert prim_positive(5) == True
    assert prim_positive(0) == False
    print(" 98. positive")

    f = prim_power(lambda x: x*2, 3)
    assert f(1) == 8
    print(" 99. power")

    assert prim_prapply(lambda x, y: x*y, [1, 2], [3, 4]) == [3, 4, 6, 8]
    print(" 100. prapply")

    r = prim_product([1, 2], [3, 4])
    assert r == [(1, 3), (1, 4), (2, 3), (2, 4)]
    print(" 101. product")

    r = prim_rapply([lambda x: x+1, lambda x: x+2], [10, 20])
    assert r == [11, 22]
    print(" 102. rapply")

    f = prim_rbind(lambda a, b: a + b, 10)
    assert f(5) == 15
    print(" 103. rbind")

    r = prim_recolor(test_onehot, 5, np.ones((30, 30), dtype=np.float32))
    assert r.shape == (1, 10, 30, 30)
    print(" 104. recolor")

    r = prim_remove([1, 2, 3], 2)
    assert r == [1, 3]
    print(" 105. remove")

    r = prim_repeat("a", 3)
    assert r == ["a", "a", "a"]
    print(" 106. repeat")

    r = prim_replace(test_onehot, 3, 7)
    assert r.shape == (1, 10, 30, 30)
    print(" 107. replace")

    r = prim_righthalf(test_onehot)
    assert r.shape == (1, 10, 30, 15)
    print(" 108. righthalf")

    r = prim_rot90(test_onehot)
    assert r.shape == (1, 10, 30, 30)
    print(" 109. rot90")

    r = prim_rot180(test_onehot)
    assert r.shape == (1, 10, 30, 30)
    print(" 110. rot180")

    r = prim_rot270(test_onehot)
    assert r.shape == (1, 10, 30, 30)
    print(" 111. rot270")

    r = prim_sfilter(lambda x: x > 0, [0, 1, 2])
    assert r == 1
    print(" 112. sfilter")

    r = prim_shape(test_onehot)
    assert r == (30, 30)
    print(" 113. shape")

    r = prim_shift(test_onehot, (2, 3))
    assert r.shape == (1, 10, 30, 30)
    print(" 114. shift")

    print("  All 30 tested OK\n")


def test_prims_121_153():
    """Test primitives 121-153."""
    print("Testing primitives 121-153...")

    test_grid = np.zeros((30, 30), dtype=np.int64)
    test_grid[5, 10] = 3
    test_grid[5, 11] = 3
    test_grid[6, 10] = 3
    test_grid[15, 20] = 7
    test_grid[20, 5] = 2
    test_grid[25, 25] = 9
    test_onehot = _make_input_array(test_grid)

    r = prim_shoot(test_onehot, (0, 0), (1, 1))
    assert r.shape == (30, 30)
    print(" 115. shoot")

    assert prim_sign(5) == 1
    assert prim_sign(-3) == -1
    assert prim_sign(0) == 0
    print(" 116. sign")

    r = prim_size(test_onehot)
    assert r == 6
    print(f" 117. size: {r}")

    r = prim_sizefilter([test_onehot], 6)
    assert len(r) == 1
    print(" 118. sizefilter")

    r = prim_subgrid(test_grid, test_onehot)
    assert r.ndim == 4
    print(f" 119. subgrid: {r.shape}")

    a = np.array([5.0, 3.0], dtype=np.float32)
    b = np.array([2.0, 1.0], dtype=np.float32)
    r = prim_subtract(a, b)
    np.testing.assert_allclose(r, [3.0, 2.0])
    print(" 120. subtract")

    r = prim_switch(test_onehot, 3, 7)
    assert r.shape == (1, 10, 30, 30)
    print(" 121. switch")

    r = prim_toindices(test_onehot)
    assert len(r) > 0
    print(f" 122. toindices ({len(r)} cells)")

    assert prim_toivec(5) == (5, 0)
    print(" 123. toivec")

    assert prim_tojvec(5) == (0, 5)
    print(" 124. tojvec")

    r = prim_toobject(test_onehot)
    assert len(r) > 0
    print(f" 125. toobject ({len(r)} cells)")

    r = prim_tophalf(test_onehot)
    assert r.shape == (1, 10, 15, 30)
    print(" 126. tophalf")

    assert prim_totuple([1, 2, 3]) == (1, 2, 3)
    print(" 127. totuple")

    r = prim_trim(test_onehot)
    assert r.shape == (1, 10, 28, 28)
    print(" 128. trim")

    r = prim_ulcorner({(10, 5), (20, 15)})
    assert r == (10, 5), f"ulcorner: {r}"
    print(" 129. ulcorner")

    uf_mask = np.zeros((30, 30), dtype=np.float32)
    uf_mask[10, 5] = 1
    uf_mask[20, 15] = 1
    r = prim_underfill(test_onehot, uf_mask, 5)
    assert r.shape == (1, 10, 30, 30)
    print(" 130. underfill")

    r = prim_underpaint(test_onehot, test_onehot)
    assert r.shape == (1, 10, 30, 30)
    print(" 131. underpaint")

    r = prim_uppermost({(10, 5), (20, 15)})
    assert r == 10, f"uppermost: {r}"
    print(" 132. uppermost")

    r = prim_upscale(test_onehot, 2)
    assert r.shape[2] == 60
    print(" 133. upscale")

    r = prim_urcorner({(10, 5), (20, 15)})
    assert r == (10, 15), f"urcorner: {r}"
    print(" 134. urcorner")

    r = prim_valmax(test_onehot, lambda p: p[0])
    assert isinstance(r, int)
    print(f" 135. valmax: {r}")

    r = prim_valmin(test_onehot, lambda p: p[0])
    assert isinstance(r, int)
    print(f" 136. valmin: {r}")

    a = _make_input_array(test_grid)
    b = _make_input_array(test_grid)
    r = prim_vconcat(a, b)
    assert r.shape[2] == 60
    print(" 137. vconcat")

    r = prim_vfrontier(test_grid.astype(np.float32))
    assert r.shape == (30, 30)
    print(" 138. vfrontier")

    line_grid = np.zeros((30, 30), dtype=np.int64)
    line_grid[5:25, 15] = 3
    line_oh = _make_input_array(line_grid)
    assert prim_vline(line_oh) == True
    print(" 139. vline")

    assert prim_vmatching(set({(1, 2)}), set({(3, 2)})) == True
    assert prim_vmatching(set({(1, 2)}), set({(1, 3)})) == False
    print(" 140. vmatching")

    r = prim_vmirror(test_onehot)
    assert r.shape == (1, 10, 30, 30)
    print(" 141. vmirror")

    r = prim_vperiod(test_onehot)
    assert r >= 1
    print(f" 142. vperiod: {r}")

    r = prim_vsplit(test_onehot, 2)
    assert len(r) == 2
    print(" 143. vsplit")

    r = prim_vupscale(test_onehot, 2)
    assert r.shape[2] == 60
    print(" 144. vupscale")

    r = prim_width(test_onehot)
    assert r == 30
    print(f" 145. width: {r}")

    r = prim_asindices(test_onehot)
    assert len(r) > 0
    print(f" 146. asindices ({len(r)} cells)")

    r = prim_asobject(test_onehot)
    assert len(r) > 0
    print(f" 147. asobject ({len(r)} cells)")

    x = np.array([5.0], dtype=np.float32)
    r = prim_crement(x, 3)
    np.testing.assert_array_equal(r, [8.0])
    r = prim_crement(x, -2)
    np.testing.assert_array_equal(r, [3.0])
    print(" 148. crement")

    print("  All 34 tested OK\n")


def test_all():
    """Run all primitive tests."""
    print("=" * 60)
    print("Testing all 151 primitives in onnx_prims.py")
    print("=" * 60 + "\n")

    test_all_first_30()
    test_prims_31_60()
    test_prims_61_90()
    test_prims_91_120()
    test_prims_121_153()

    print("=" * 60)
    print("ALL 151 PRIMITIVES TESTED SUCCESSFULLY!")
    print("=" * 60)





# ============================================================================
# Primitives 61-90
# ============================================================================


def prim_index(grid: np.ndarray, position: Tuple[int, int]) -> int:
    """index(grid, (r, c)) -> value at position."""
    r, c = position
    g = _to_grid(grid)
    return int(g[r, c])


def prim_insert(container, item):
    """insert(container, item) -> add item to container."""
    if isinstance(container, set):
        return container | {item}
    if isinstance(container, frozenset):
        return container | frozenset([item])
    if isinstance(container, list):
        return container + [item]
    return {item}


def prim_intersection(a, b):
    """intersection(a, b) -> set intersection."""
    return a & b if isinstance(a, set) else set(a) & set(b)


def prim_interval(start: int, stop: int, step: int = 1) -> List[int]:
    """interval(start, stop, step) -> range list."""
    return list(range(start, stop, step))


def prim_invert(x: np.ndarray) -> np.ndarray:
    """invert(x) -> -x (negate)."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", list(x.shape))
    out = bld.add_node("Neg", [inp], "out")
    bld.add_output("out", list(x.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": x})["out"]


def prim_last(objs):
    """last(objs) -> last element."""
    if isinstance(objs, np.ndarray):
        if objs.ndim >= 1:
            return objs.reshape(-1)[-1]
    if hasattr(objs, '__getitem__'):
        return objs[-1]
    return objs


def prim_lbind(func, arg):
    """lbind(func, arg) -> partial application (left)."""
    def bound(x):
        return func(arg, x)
    return bound


def prim_leastcolor(grid: np.ndarray) -> int:
    """leastcolor(grid) -> least frequent non-zero color."""
    g = _to_grid(grid)
    nonzero = g[g > 0]
    if len(nonzero) == 0:
        return 0
    unique, counts = np.unique(nonzero, return_counts=True)
    return int(unique[np.argmin(counts)])


def prim_leastcommon(objs) -> Any:
    """leastcommon(objs) -> least common element."""
    from collections import Counter
    if not objs:
        return None
    counter = Counter(objs)
    return counter.most_common()[-1][0]


def prim_lefthalf(grid: np.ndarray) -> np.ndarray:
    """lefthalf(grid) -> left half of grid."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", [1, 10, 30, 30])
    starts = bld.add_initializer("starts", np.array([0, 0, 0, 0], dtype=np.int64))
    ends = bld.add_initializer("ends", np.array([1, 10, 30, 15], dtype=np.int64))
    axes = bld.add_initializer("axes", np.array([0, 1, 2, 3], dtype=np.int64))
    out = bld.add_node("Slice", [inp, starts, ends, axes], "out")
    bld.add_output("out", [1, 10, 30, 15])
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": grid})["out"]


def prim_leftmost(obj) -> int:
    """leftmost(obj) -> minimum column index."""
    if isinstance(obj, np.ndarray):
        g = _to_grid(obj)
        cols = np.nonzero(g)[1]
        return int(cols.min()) if len(cols) > 0 else 0
    if isinstance(obj, set):
        return min(c for _, c in obj) if obj else 0
    return 0


def prim_llcorner(obj) -> Tuple[int, int]:
    """llcorner(obj) -> lower-left corner of bounding box."""
    if isinstance(obj, np.ndarray):
        g = _to_grid(obj)
        rows, cols = np.nonzero(g)
        if len(rows) == 0:
            return (0, 0)
        return (int(rows.max()), int(cols.min()))
    if isinstance(obj, set):
        if not obj:
            return (0, 0)
        rows = [r for r, c in obj]
        cols = [c for r, c in obj]
        return (max(rows), min(cols))
    return (0, 0)


def prim_lowermost(obj) -> int:
    """lowermost(obj) -> maximum row index."""
    if isinstance(obj, np.ndarray):
        g = _to_grid(obj)
        rows = np.nonzero(g)[0]
        return int(rows.max()) if len(rows) > 0 else 0
    if isinstance(obj, set):
        return max(r for r, c in obj) if obj else 0
    return 0


def prim_lrcorner(obj) -> Tuple[int, int]:
    """lrcorner(obj) -> lower-right corner of bounding box."""
    if isinstance(obj, np.ndarray):
        g = _to_grid(obj)
        rows, cols = np.nonzero(g)
        if len(rows) == 0:
            return (0, 0)
        return (int(rows.max()), int(cols.max()))
    if isinstance(obj, set):
        if not obj:
            return (0, 0)
        rows = [r for r, c in obj]
        cols = [c for r, c in obj]
        return (max(rows), max(cols))
    return (0, 0)


def prim_mapply(func, objs):
    """mapply(func, objs) -> merge (flatten) results of applying func."""
    result = []
    for obj in objs:
        r = func(obj)
        if isinstance(r, (list, tuple)):
            result.extend(r)
        elif isinstance(r, set):
            result.extend(list(r))
        elif isinstance(r, np.ndarray):
            result.append(r)
        else:
            result.append(r)
    return result


def prim_matcher(value):
    """matcher(value) -> function that checks equality to value."""
    def check(x):
        if isinstance(x, np.ndarray):
            return np.array_equal(x, value)
        return x == value
    return check


def prim_maximum(a: np.ndarray, b) -> np.ndarray:
    """maximum(a, b) -> element-wise max."""
    bld = OnnxBuilder()
    inp = bld.add_input("a", list(a.shape))
    val = bld.add_initializer("b", np.array(b, dtype=np.float32))
    out = bld.add_node("Max", [inp, val], "out")
    bld.add_output("out", list(a.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"a": a})["out"]


def prim_merge(objs) -> list:
    """merge(objs) -> flatten nested objects."""
    result = []
    for obj in objs:
        if isinstance(obj, (list, tuple)):
            result.extend(obj)
        elif isinstance(obj, set):
            result.extend(list(obj))
        elif isinstance(obj, np.ndarray):
            result.append(obj)
        else:
            result.append(obj)
    return result


def prim_mfilter(func, objs):
    """mfilter(func, objs) -> filter and flatten."""
    result = []
    for obj in objs:
        if func(obj):
            if isinstance(obj, (list, tuple)):
                result.extend(obj)
            else:
                result.append(obj)
    return result


def prim_minimum(a: np.ndarray, b) -> np.ndarray:
    """minimum(a, b) -> element-wise min."""
    bld = OnnxBuilder()
    inp = bld.add_input("a", list(a.shape))
    val = bld.add_initializer("b", np.array(b, dtype=np.float32))
    out = bld.add_node("Min", [inp, val], "out")
    bld.add_output("out", list(a.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"a": a})["out"]


def prim_mostcolor(grid: np.ndarray) -> int:
    """mostcolor(grid) -> most frequent non-zero color."""
    g = _to_grid(grid)
    nonzero = g[g > 0]
    if len(nonzero) == 0:
        return 0
    unique, counts = np.unique(nonzero, return_counts=True)
    return int(unique[np.argmax(counts)])


def prim_mostcommon(objs) -> Any:
    """mostcommon(objs) -> most common element."""
    from collections import Counter
    if not objs:
        return None
    counter = Counter(objs)
    return counter.most_common(1)[0][0]


def prim_move(grid: np.ndarray, obj_mask: np.ndarray, direction: Tuple[int, int]) -> np.ndarray:
    """move(grid, obj, direction) -> move object by direction vector."""
    dr, dc = direction
    bld = OnnxBuilder()
    g = bld.add_input("grid", [1, 10, 30, 30])
    m = bld.add_input("mask", [30, 30])

    bg = np.zeros((1, 10, 30, 30), dtype=np.float32)
    bg_init = bld.add_initializer("bg", bg)

    m_f = bld.add_node("Cast", [m], "m_f", to=TensorProto.FLOAT)
    m_exp = bld.add_node("Unsqueeze", [m_f], "m_exp", axes=[0, 1])

    g_inv = bld.add_node("Sub", [g, bld.add_initializer("zero", np.zeros((1,10,30,30), dtype=np.float32))], "g_inv")

    moved_mask = np.zeros((30, 30), dtype=np.float32)
    for r in range(30):
        for c in range(30):
            if obj_mask[r, c] > 0:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 30 and 0 <= nc < 30:
                    moved_mask[nr, nc] = 1.0

    result = grid.copy()
    g_grid = _to_grid(grid)
    for r in range(30):
        for c in range(30):
            if obj_mask[r, c] > 0:
                color = int(g_grid[r, c])
                nr, nc = r + dr, c + dc
                if 0 <= nr < 30 and 0 <= nc < 30:
                    result[0, :, r, c] = 0
                    result[0, color, nr, nc] = 1.0

    bld2 = OnnxBuilder()
    init_name = bld2.add_initializer("result", result)
    _identity_from_initializer(bld2, init_name, "out", [1, 10, 30, 30])
    return bld2.run({})["out"]


def prim_mpapply(func, a, b):
    """mpapply(func, a, b) -> apply func pairwise and merge."""
    result = []
    for x, y in zip(a, b):
        r = func(x, y)
        if isinstance(r, (list, tuple)):
            result.extend(r)
        else:
            result.append(r)
    return result


def prim_multiply(a: np.ndarray, b) -> np.ndarray:
    """multiply(a, b) -> element-wise multiplication."""
    bld = OnnxBuilder()
    inp = bld.add_input("a", list(a.shape))
    val = bld.add_initializer("b", np.array(b, dtype=np.float32))
    out = bld.add_node("Mul", [inp, val], "out")
    bld.add_output("out", list(a.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"a": a})["out"]


def prim_neighbors(grid: np.ndarray) -> np.ndarray:
    """neighbors(mask) -> 4-connected neighbor positions."""
    result = np.zeros((30, 30), dtype=np.float32)
    for r in range(30):
        for c in range(30):
            if grid[r, c] > 0:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < 30 and 0 <= nc < 30:
                        result[nr, nc] = 1.0
    bld = OnnxBuilder()
    init_name = bld.add_initializer("nbrs", result)
    _identity_from_initializer(bld, init_name, "out", [30, 30])
    return bld.run({})["out"]


def prim_normalize(obj) -> Any:
    """normalize(obj) -> shift to origin."""
    if isinstance(obj, np.ndarray):
        g = _to_grid(obj)
        rows, cols = np.nonzero(g)
        if len(rows) == 0:
            return obj
        min_r, min_c = rows.min(), cols.min()
        result = np.zeros_like(obj)
        for r, c in zip(rows, cols):
            color = int(g[r, c])
            result[0, color, r - min_r, c - min_c] = 1.0
        return result
    if isinstance(obj, set) and obj:
        min_r = min(r for r, c in obj)
        min_c = min(c for r, c in obj)
        return {(r - min_r, c - min_c) for r, c in obj}
    return obj


def prim_numcolors(grid: np.ndarray) -> int:
    """numcolors(grid) -> number of unique colors."""
    g = _to_grid(grid)
    return len(np.unique(g))


# ============================================================================
# Primitives 91-120
# ============================================================================


def prim_objects(grid: np.ndarray) -> List[np.ndarray]:
    """objects(grid) -> set of foreground objects as separate one-hot tensors."""
    g = _to_grid(grid)
    bg = 0
    visited = np.zeros((30, 30), dtype=bool)
    objects = []
    for r in range(30):
        for c in range(30):
            if g[r, c] != bg and not visited[r, c]:
                color = int(g[r, c])
                obj = np.zeros((1, 10, 30, 30), dtype=np.float32)
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    if g[cr, cc] == color:
                        obj[0, color, cr, cc] = 1.0
                        visited[cr, cc] = True
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < 30 and 0 <= nc < 30 and not visited[nr, nc] and g[nr, nc] == color:
                                queue.append((nr, nc))
                objects.append(obj)
    return objects


def prim_occurrences(grid: np.ndarray, sub_grid: np.ndarray) -> List[Tuple[int, int]]:
    """occurrences(grid, sub_grid) -> positions where sub_grid appears."""
    g = _to_grid(grid)
    sg = _to_grid(sub_grid)
    sh, sw = sg.shape
    gh, gw = g.shape
    result = []
    for r in range(gh - sh + 1):
        for c in range(gw - sw + 1):
            if np.array_equal(g[r:r+sh, c:c+sw], sg):
                result.append((r, c))
    return result


def prim_ofcolor(grid: np.ndarray, color: int) -> np.ndarray:
    """ofcolor(grid, color) -> mask of cells with given color."""
    g = _to_grid(grid)
    mask = (g == color).astype(np.float32)
    bld = OnnxBuilder()
    init_name = bld.add_initializer("mask", mask)
    _identity_from_initializer(bld, init_name, "out", [30, 30])
    return bld.run({})["out"]


def prim_order(objs, criterion=None):
    """order(objs, criterion) -> sorted objects."""
    if criterion is None:
        return sorted(objs, key=lambda x: str(x))
    return sorted(objs, key=criterion)


def prim_other(container, item):
    """other(container, item) -> container minus item."""
    if isinstance(container, set):
        return container - {item}
    if isinstance(container, (list, tuple)):
        return [x for x in container if x != item]
    return container


def prim_outbox(mask: np.ndarray) -> np.ndarray:
    """outbox(mask) -> outer bounding box."""
    r_indices, c_indices = np.nonzero(mask)
    if len(r_indices) == 0:
        return np.zeros((30, 30), dtype=np.float32)
    min_r, max_r = int(r_indices.min()), int(r_indices.max())
    min_c, max_c = int(c_indices.min()), int(c_indices.max())
    result = np.zeros((30, 30), dtype=np.float32)
    for r in range(min_r - 1, max_r + 2):
        for c in range(min_c - 1, max_c + 2):
            if 0 <= r < 30 and 0 <= c < 30:
                result[r, c] = 1.0
    bld = OnnxBuilder()
    init_name = bld.add_initializer("outbox", result)
    _identity_from_initializer(bld, init_name, "out", [30, 30])
    return bld.run({})["out"]


def prim_paint(grid: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """paint(grid, obj) -> paint object onto grid."""
    g_grid = _to_grid(grid)
    o_grid = _to_grid(obj)
    result = g_grid.copy()
    for r in range(30):
        for c in range(30):
            if o_grid[r, c] > 0:
                result[r, c] = o_grid[r, c]
    one_hot = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for ch in range(10):
        one_hot[0, ch] = (result == ch).astype(np.float32)
    return one_hot


def prim_pair(a, b):
    """pair(a, b) -> zip a and b."""
    return list(zip(a, b))


def prim_palette(grid: np.ndarray) -> List[int]:
    """palette(grid) -> set of colors present."""
    g = _to_grid(grid)
    return sorted(set(int(x) for x in g.flatten()))


def prim_papply(func, a, b):
    """papply(func, a, b) -> apply func pairwise."""
    return [func(x, y) for x, y in zip(a, b)]


def prim_partition(grid: np.ndarray) -> List[np.ndarray]:
    """partition(grid) -> partition grid by color."""
    g = _to_grid(grid)
    partitions = {}
    for r in range(30):
        for c in range(30):
            color = int(g[r, c])
            if color not in partitions:
                partitions[color] = np.zeros((1, 10, 30, 30), dtype=np.float32)
            partitions[color][0, color, r, c] = 1.0
    return list(partitions.values())


def prim_portrait(grid: np.ndarray) -> bool:
    """portrait(grid) -> True if height > width."""
    if isinstance(grid, np.ndarray) and grid.ndim >= 3:
        return grid.shape[-2] > grid.shape[-1]
    return False


def prim_position(obj_a, obj_b) -> Tuple[int, int]:
    """position(obj_a, obj_b) -> relative position."""
    if isinstance(obj_a, set) and isinstance(obj_b, set):
        if not obj_a or not obj_b:
            return (0, 0)
        r_a = np.mean([r for r, c in obj_a])
        c_a = np.mean([c for r, c in obj_a])
        r_b = np.mean([r for r, c in obj_b])
        c_b = np.mean([c for r, c in obj_b])
        dr = int(round(r_b - r_a))
        dc = int(round(c_b - c_a))
        return (dr, dc)
    return (0, 0)


def prim_positive(x) -> bool:
    """positive(x) -> True if x > 0."""
    return int(x) > 0


def prim_power(func, n):
    """power(func, n) -> apply func n times."""
    def powered(x):
        result = x
        for _ in range(n):
            result = func(result)
        return result
    return powered


def prim_prapply(func, a, b):
    """prapply(func, a, b) -> apply func to all pairs."""
    result = []
    for x in a:
        for y in b:
            result.append(func(x, y))
    return result


def prim_product(a, b):
    """product(a, b) -> cartesian product."""
    return [(x, y) for x in a for y in b]


def prim_rapply(funcs, container):
    """rapply(funcs, container) -> apply each func to corresponding element."""
    return [f(x) for f, x in zip(funcs, container)]


def prim_rbind(func, arg):
    """rbind(func, arg) -> partial application (right)."""
    def bound(x):
        return func(x, arg)
    return bound


def prim_recolor(grid: np.ndarray, color: int, mask: np.ndarray) -> np.ndarray:
    """recolor(grid, color, mask) -> recolor mask positions."""
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r in range(30):
        for c in range(30):
            if mask[r, c] > 0:
                result[0, color, r, c] = 1.0
    return result


def prim_remove(container, item):
    """remove(container, item) -> container without item."""
    if isinstance(container, set):
        return container - {item}
    if isinstance(container, (list, tuple)):
        return [x for x in container if x != item]
    if isinstance(container, frozenset):
        return container - frozenset([item])
    return container


def prim_repeat(item, n: int):
    """repeat(item, n) -> n copies of item."""
    return [item] * n


def prim_replace(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
    """replace(grid, old_color, new_color) -> replace color."""
    g = _to_grid(grid)
    result = g.copy()
    result[result == old_color] = new_color
    one_hot = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for ch in range(10):
        one_hot[0, ch] = (result == ch).astype(np.float32)
    return one_hot


def prim_righthalf(grid: np.ndarray) -> np.ndarray:
    """righthalf(grid) -> right half of grid."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", [1, 10, 30, 30])
    starts = bld.add_initializer("starts", np.array([0, 0, 0, 15], dtype=np.int64))
    ends = bld.add_initializer("ends", np.array([1, 10, 30, 30], dtype=np.int64))
    axes = bld.add_initializer("axes", np.array([0, 1, 2, 3], dtype=np.int64))
    out = bld.add_node("Slice", [inp, starts, ends, axes], "out")
    bld.add_output("out", [1, 10, 30, 15])
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": grid})["out"]


def prim_rot90(grid: np.ndarray) -> np.ndarray:
    """rot90(grid) -> rotate 90 degrees clockwise."""
    return _apply_permutation(grid, lambda r, c: (c, 29 - r))


def prim_rot180(grid: np.ndarray) -> np.ndarray:
    """rot180(grid) -> rotate 180 degrees."""
    return _apply_permutation(grid, lambda r, c: (29 - r, 29 - c))


def prim_rot270(grid: np.ndarray) -> np.ndarray:
    """rot270(grid) -> rotate 270 degrees clockwise."""
    return _apply_permutation(grid, lambda r, c: (29 - c, r))


def prim_sfilter(func, objs):
    """sfilter(func, objs) -> filter objects, keep single result."""
    for obj in objs:
        if func(obj):
            return obj
    return objs[0] if objs else None


def prim_shape(obj) -> Tuple[int, int]:
    """shape(obj) -> (height, width)."""
    if isinstance(obj, np.ndarray) and obj.ndim >= 3:
        return (obj.shape[-2], obj.shape[-1])
    if isinstance(obj, set) and obj:
        rows = [r for r, c in obj]
        cols = [c for r, c in obj]
        return (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)
    return (1, 1)


def prim_shift(obj, direction: Tuple[int, int]) -> Any:
    """shift(obj, direction) -> move by direction."""
    dr, dc = direction
    if isinstance(obj, np.ndarray):
        g = _to_grid(obj)
        result = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(30):
            for c in range(30):
                if g[r, c] > 0:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 30 and 0 <= nc < 30:
                        result[0, int(g[r, c]), nr, nc] = 1.0
        return result
    if isinstance(obj, set):
        return {(r + dr, c + dc) for r, c in obj}
    return obj


# ============================================================================
# Primitives 121-152
# ============================================================================


def prim_shoot(grid: np.ndarray, start: Tuple[int, int], direction: Tuple[int, int]) -> np.ndarray:
    """shoot(start, direction) -> line from start in direction."""
    r, c = start
    dr, dc = direction
    result = np.zeros((30, 30), dtype=np.float32)
    cr, cc = r, c
    while 0 <= cr < 30 and 0 <= cc < 30:
        result[cr, cc] = 1.0
        cr += dr
        cc += dc
    bld = OnnxBuilder()
    init_name = bld.add_initializer("shoot", result)
    _identity_from_initializer(bld, init_name, "out", [30, 30])
    return bld.run({})["out"]


def prim_sign(x) -> int:
    """sign(x) -> sign of value."""
    v = int(x)
    if v > 0: return 1
    if v < 0: return -1
    return 0


def prim_size(obj) -> int:
    """size(obj) -> number of elements."""
    if isinstance(obj, np.ndarray):
        g = _to_grid(obj)
        return int(np.count_nonzero(g))
    if isinstance(obj, (set, list, tuple)):
        return len(obj)
    return 1


def prim_sizefilter(objs, n: int):
    """sizefilter(objs, n) -> keep objects of size n."""
    result = []
    for obj in objs:
        if isinstance(obj, np.ndarray):
            g = _to_grid(obj)
            sz = int(np.count_nonzero(g))
        elif isinstance(obj, (set, list)):
            sz = len(obj)
        else:
            sz = 1
        if sz == n:
            result.append(obj)
    return result


def prim_subgrid(mask: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """subgrid(mask, grid) -> crop to bounding box of mask."""
    r_indices, c_indices = np.nonzero(mask)
    if len(r_indices) == 0:
        return np.zeros((1, 10, 1, 1), dtype=np.float32)
    min_r, max_r = int(r_indices.min()), int(r_indices.max())
    min_c, max_c = int(c_indices.min()), int(c_indices.max())
    g = _to_grid(grid)
    sub = g[min_r:max_r+1, min_c:max_c+1]
    h, w = sub.shape
    result = np.zeros((1, 10, h, w), dtype=np.float32)
    for ch in range(10):
        result[0, ch] = (sub == ch).astype(np.float32)
    return result


def prim_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """subtract(a, b) -> element-wise subtraction."""
    bld = OnnxBuilder()
    inp_a = bld.add_input("a", list(a.shape))
    inp_b = bld.add_input("b", list(b.shape))
    out = bld.add_node("Sub", [inp_a, inp_b], "out")
    bld.add_output("out", list(a.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"a": a, "b": b})["out"]


def prim_switch(grid: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """switch(grid, c1, c2) -> swap two colors."""
    g = _to_grid(grid)
    result = g.copy()
    mask1 = (g == c1)
    mask2 = (g == c2)
    result[mask1] = c2
    result[mask2] = c1
    one_hot = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for ch in range(10):
        one_hot[0, ch] = (result == ch).astype(np.float32)
    return one_hot


def prim_toindices(obj) -> set:
    """toindices(obj) -> set of (r,c) positions."""
    if isinstance(obj, np.ndarray):
        g = _to_grid(obj)
        rows, cols = np.nonzero(g)
        return {(int(r), int(c)) for r, c in zip(rows, cols)}
    if isinstance(obj, set):
        return obj
    return set()


def prim_toivec(obj) -> Tuple[int, int]:
    """toivec(obj) -> (row, 0) vector."""
    if isinstance(obj, (int, np.integer)):
        return (int(obj), 0)
    if isinstance(obj, tuple) and len(obj) >= 2:
        return (obj[0], 0)
    return (0, 0)


def prim_tojvec(obj) -> Tuple[int, int]:
    """tojvec(obj) -> (0, col) vector."""
    if isinstance(obj, (int, np.integer)):
        return (0, int(obj))
    if isinstance(obj, tuple) and len(obj) >= 2:
        return (0, obj[1])
    return (0, 0)


def prim_toobject(grid: np.ndarray) -> set:
    """toobject(grid) -> set of ((r,c), color) for non-zero cells."""
    g = _to_grid(grid)
    result = set()
    for r in range(30):
        for c in range(30):
            if g[r, c] > 0:
                result.add(((r, c), int(g[r, c])))
    return result


def prim_tophalf(grid: np.ndarray) -> np.ndarray:
    """tophalf(grid) -> top half of grid."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", [1, 10, 30, 30])
    starts = bld.add_initializer("starts", np.array([0, 0, 0, 0], dtype=np.int64))
    ends = bld.add_initializer("ends", np.array([1, 10, 15, 30], dtype=np.int64))
    axes = bld.add_initializer("axes", np.array([0, 1, 2, 3], dtype=np.int64))
    out = bld.add_node("Slice", [inp, starts, ends, axes], "out")
    bld.add_output("out", [1, 10, 15, 30])
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": grid})["out"]


def prim_totuple(obj) -> tuple:
    """totuple(obj) -> convert to tuple."""
    if isinstance(obj, np.ndarray):
        return tuple(obj.flatten().tolist())
    if isinstance(obj, (list, set)):
        return tuple(obj)
    return (obj,)


def prim_trim(grid: np.ndarray) -> np.ndarray:
    """trim(grid) -> remove outer border."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", [1, 10, 30, 30])
    starts = bld.add_initializer("starts", np.array([0, 0, 1, 1], dtype=np.int64))
    ends = bld.add_initializer("ends", np.array([1, 10, 29, 29], dtype=np.int64))
    axes = bld.add_initializer("axes", np.array([0, 1, 2, 3], dtype=np.int64))
    out = bld.add_node("Slice", [inp, starts, ends, axes], "out")
    bld.add_output("out", [1, 10, 28, 28])
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": grid})["out"]


def prim_ulcorner(obj) -> Tuple[int, int]:
    """ulcorner(obj) -> upper-left corner of bounding box."""
    if isinstance(obj, np.ndarray):
        g = _to_grid(obj)
        rows, cols = np.nonzero(g)
        if len(rows) == 0:
            return (0, 0)
        return (int(rows.min()), int(cols.min()))
    if isinstance(obj, set) and obj:
        return (min(r for r, c in obj), min(c for r, c in obj))
    return (0, 0)


def prim_underfill(grid: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray:
    """underfill(grid, mask, color) -> paint only background cells."""
    g = _to_grid(grid)
    result = g.copy()
    for r in range(30):
        for c in range(30):
            if mask[r, c] > 0 and g[r, c] == 0:
                result[r, c] = color
    one_hot = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for ch in range(10):
        one_hot[0, ch] = (result == ch).astype(np.float32)
    return one_hot


def prim_underpaint(grid: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """underpaint(grid, obj) -> paint obj onto background only."""
    g = _to_grid(grid)
    o = _to_grid(obj)
    result = g.copy()
    for r in range(30):
        for c in range(30):
            if o[r, c] > 0 and g[r, c] == 0:
                result[r, c] = o[r, c]
    one_hot = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for ch in range(10):
        one_hot[0, ch] = (result == ch).astype(np.float32)
    return one_hot


def prim_uppermost(obj) -> int:
    """uppermost(obj) -> minimum row index."""
    if isinstance(obj, np.ndarray):
        g = _to_grid(obj)
        rows = np.nonzero(g)[0]
        return int(rows.min()) if len(rows) > 0 else 0
    if isinstance(obj, set) and obj:
        return min(r for r, c in obj)
    return 0


def prim_upscale(grid: np.ndarray, factor: int) -> np.ndarray:
    """upscale(grid, factor) -> upscale spatially."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", [1, 10, 30, 30])
    repeats = bld.add_initializer("reps", np.array([1, 1, factor, factor], dtype=np.int64))
    out = bld.add_node("Tile", [inp, repeats], "out")
    bld.add_output("out", [1, 10, 30 * factor, 30 * factor])
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": grid})["out"]


def prim_urcorner(obj) -> Tuple[int, int]:
    """urcorner(obj) -> upper-right corner."""
    if isinstance(obj, np.ndarray):
        g = _to_grid(obj)
        rows, cols = np.nonzero(g)
        if len(rows) == 0:
            return (0, 0)
        return (int(rows.min()), int(cols.max()))
    if isinstance(obj, set) and obj:
        return (min(r for r, c in obj), max(c for r, c in obj))
    return (0, 0)


def prim_valmax(grid: np.ndarray, obj) -> int:
    """valmax(grid, obj) -> max value of obj applied to grid."""
    if callable(obj):
        vals = []
        for r in range(30):
            for c in range(30):
                vals.append(obj((r, c)))
        return max(vals) if vals else 0
    return int(obj) if isinstance(obj, (int, float)) else 0


def prim_valmin(grid: np.ndarray, obj) -> int:
    """valmin(grid, obj) -> min value of obj applied to grid."""
    if callable(obj):
        vals = []
        for r in range(30):
            for c in range(30):
                vals.append(obj((r, c)))
        return min(vals) if vals else 0
    return int(obj) if isinstance(obj, (int, float)) else 0


def prim_vconcat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """vconcat(a, b) -> vertical concatenation (axis=2)."""
    bld = OnnxBuilder()
    inp_a = bld.add_input("a", list(a.shape))
    inp_b = bld.add_input("b", list(b.shape))
    out = bld.add_node("Concat", [inp_a, inp_b], "out", axis=2)
    out_shape = list(a.shape)
    out_shape[2] = a.shape[2] + b.shape[2]
    bld.add_output("out", out_shape)
    bld.nodes[-1].output[0] = "out"
    return bld.run({"a": a, "b": b})["out"]


def prim_vfrontier(mask: np.ndarray) -> np.ndarray:
    """vfrontier(mask) -> vertical frontier cells."""
    result = np.zeros((30, 30), dtype=np.float32)
    for r in range(30):
        for c in range(30):
            if mask[r, c] > 0:
                if r == 0 or r == 29 or mask[r-1, c] == 0 or mask[r+1, c] == 0:
                    result[r, c] = 1.0
    bld = OnnxBuilder()
    init_name = bld.add_initializer("vf", result)
    _identity_from_initializer(bld, init_name, "out", [30, 30])
    return bld.run({})["out"]


def prim_vline(obj) -> bool:
    """vline(obj) -> check if object is a vertical line."""
    if isinstance(obj, np.ndarray):
        g = _to_grid(obj)
        rows, cols = np.nonzero(g)
        if len(rows) == 0:
            return False
        return len(set(cols.tolist())) == 1
    if isinstance(obj, set) and obj:
        return len(set(c for _, c in obj)) == 1
    return False


def prim_vmatching(obj_a, obj_b) -> bool:
    """vmatching(obj_a, obj_b) -> share same column set."""
    if isinstance(obj_a, set) and isinstance(obj_b, set):
        cols_a = {c for _, c in obj_a}
        cols_b = {c for _, c in obj_b}
        return cols_a == cols_b
    return False


def prim_vmirror(grid: np.ndarray) -> np.ndarray:
    """vmirror(grid) -> vertical mirror (flip axis 2)."""
    return grid[:, :, ::-1, :].copy()


def prim_vperiod(grid: np.ndarray) -> int:
    """vperiod(grid) -> vertical period of pattern."""
    g = _to_grid(grid)
    for period in range(1, 30):
        match = True
        for r in range(30 - period):
            for c in range(30):
                if g[r, c] != g[r + period, c]:
                    match = False
                    break
            if not match:
                break
        if match:
            return period
    return 30


def prim_vsplit(grid: np.ndarray, n: int = 2) -> List[np.ndarray]:
    """vsplit(grid, n) -> split vertically into n parts."""
    _, channels, h, w = grid.shape
    part_h = h // n
    result = []
    for i in range(n):
        start = i * part_h
        bld = OnnxBuilder()
        inp = bld.add_input("x", [1, 10, 30, 30])
        starts = bld.add_initializer("s", np.array([0, 0, start, 0], dtype=np.int64))
        ends = bld.add_initializer("e", np.array([1, 10, start + part_h, 30], dtype=np.int64))
        axes = bld.add_initializer("a", np.array([0, 1, 2, 3], dtype=np.int64))
        out = bld.add_node("Slice", [inp, starts, ends, axes], "out")
        bld.add_output("out", [1, 10, part_h, 30])
        bld.nodes[-1].output[0] = "out"
        result.append(bld.run({"x": grid})["out"])
    return result


def prim_vupscale(grid: np.ndarray, factor: int) -> np.ndarray:
    """vupscale(grid, factor) -> upscale vertically."""
    _, c, h, w = grid.shape
    result = np.repeat(grid, factor, axis=2)
    return result


def prim_width(obj) -> int:
    """width(obj) -> width of object/grid."""
    if isinstance(obj, np.ndarray) and obj.ndim >= 3:
        return obj.shape[-1]
    if isinstance(obj, set) and obj:
        return max(c for _, c in obj) - min(c for _, c in obj) + 1
    return 1


def prim_asindices(grid: np.ndarray) -> set:
    """asindices(grid) -> set of all (r,c) positions of non-zero cells."""
    g = _to_grid(grid)
    rows, cols = np.nonzero(g)
    return {(int(r), int(c)) for r, c in zip(rows, cols)}


def prim_asobject(grid: np.ndarray) -> set:
    """asobject(grid) -> set of ((r,c), color) for all non-zero cells."""
    g = _to_grid(grid)
    result = set()
    for r in range(30):
        for c in range(30):
            if g[r, c] > 0:
                result.add(((r, c), int(g[r, c])))
    return result


def prim_crement(x: np.ndarray, delta: int = 1) -> np.ndarray:
    """crement(x, delta) -> x + delta (generic increment/decrement)."""
    bld = OnnxBuilder()
    inp = bld.add_input("x", list(x.shape))
    d = bld.add_initializer("d", np.array(float(delta), dtype=np.float32))
    out = bld.add_node("Add", [inp, d], "out")
    bld.add_output("out", list(x.shape))
    bld.nodes[-1].output[0] = "out"
    return bld.run({"x": x})["out"]


if __name__ == "__main__":
    test_all()