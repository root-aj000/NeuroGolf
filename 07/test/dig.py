# Quick shape-trace diagnostic — paste before running tests
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import arc_onnx_primitives as P

def diagnose_bbox():
    """Trace exact shapes through _bbox for RECT_GRID (4x6)."""
    nodes, inits = [], []
    h, w = 4, 6
    
    # Build: input → _reduce_occ → _bbox → lo_r → _to_scalar_f
    occ = P._reduce_occ(nodes, inits, "input", P.NUM_COLORS, "diag")
    print(f"occ node output: checking what shape this should be")
    
    # Manually trace _bbox
    occ2_name = P._op(nodes, "Reshape",
                      [occ, P._init(inits, "rs", np.array([h, w], dtype=np.int64))],
                      "diag", suffix="occ2")
    print(f"occ2 = Reshape(occ, [{h},{w}])  → expected shape ({h},{w})")
    
    # ReduceMax over axis=1 (columns) → row presence vector
    rows_f = P._reduce_max(nodes, inits, occ2_name, "diag_r", axes=[1], keepdims=0)
    print(f"rows_f = ReduceMax(occ2, axis=1, keepdims=0) → expected shape ({h},)")
    
    # ArgMax over axis=0
    lo_r = P._op(nodes, "ArgMax", [rows_f], "diag", suffix="lor", axis=0, keepdims=0)
    print(f"lo_r = ArgMax(rows_f, axis=0) → expected shape ()")
    
    # Cast + Reshape to (1,1,1,1)
    xf = P._op(nodes, "Cast", [lo_r], "diag", to=TensorProto.FLOAT)
    out = P._op(nodes, "Reshape",
                [xf, P._init(inits, "rs4", np.array([1,1,1,1], dtype=np.int64))],
                "diag", suffix="sc")
    
    # Build model and check shapes
    inp = np.zeros((1, P.NUM_COLORS, h, w), dtype=np.float32)
    X = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, P.NUM_COLORS, h, w])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 1, 1])
    nodes.append(helper.make_node("Identity", [out], ["output"]))
    graph = helper.make_graph(nodes, "diag", [X], [Y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        for n in inferred.graph.node:
            print(f"  Node {n.op_type}: outputs={list(n.output)}")
        for vi in inferred.graph.value_info:
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            print(f"  ValueInfo {vi.name}: shape={shape}")
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")

diagnose_bbox()