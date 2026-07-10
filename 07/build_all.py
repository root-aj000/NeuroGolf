"""
Strategy 07: Compile DSL solvers to ONNX.

Only handles the simplest tasks:
  - Spatial: rot90, rot180, rot270, hmirror, vmirror
  - Color: replace(a, b), switch(a, b)

Reads all 400 tasks, builds a model for each one it can handle.
"""
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
TASK_DIR = Path(__file__).parent.parent / "neurogolf-2026"
SOLVERS_PATH = Path(__file__).parent.parent / "arc-dsl" / "solvers.py"
MATCHES_PATH = Path(__file__).parent.parent / "03" / "solver_matches.json"
OUTPUT_DIR = Path(__file__).parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

N_COLORS = 10
GRID = 30

# ── Spatial permutation formulas ─────────────────────────────
# Each returns (r_in, c_in) given (r_out, c_out) for a grid of size H x W.

def perm_rot90(H, W):
    return lambda r, c: (W - 1 - c, r)

def perm_rot180(H, W):
    return lambda r, c: (H - 1 - r, W - 1 - c)

def perm_rot270(H, W):
    return lambda r, c: (c, H - 1 - r)

def perm_hmirror(H, W):
    return lambda r, c: (H - 1 - r, c)

def perm_vmirror(H, W):
    return lambda r, c: (r, W - 1 - c)

SPATIAL_PERMS = {
    "rot90": perm_rot90,
    "rot180": perm_rot180,
    "rot270": perm_rot270,
    "hmirror": perm_hmirror,
    "vmirror": perm_vmirror,
}

# ── Step 1: Get grid size from task ──────────────────────────

def get_grid_size(task_key):
    """Return (H, W) if all examples have the same input size, else None."""
    with open(TASK_DIR / f"{task_key}.json") as f:
        task = json.load(f)
    sizes = set()
    for ex in task["train"] + task.get("test", []):
        inp = ex["input"]
        sizes.add((len(inp), len(inp[0])))
    if len(sizes) == 1:
        return list(sizes)[0]
    return None

# ── Step 2: Parse DSL solver ─────────────────────────────────

def get_solver_code(solver_name):
    with open(SOLVERS_PATH) as f:
        content = f.read()
    import re
    pattern = rf"(def {solver_name}\(.*?\n(?:    .*\n)*)"
    match = re.search(pattern, content)
    return match.group(0) if match else None

def parse_solver_ops(solver_code):
    """Parse solver into list of (target, func, args)."""
    DSL_CONSTS = {
        "ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4,
        "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10,
    }
    def resolve(a):
        a = a.strip()
        if a in DSL_CONSTS:
            return DSL_CONSTS[a]
        try:
            return int(a)
        except ValueError:
            return a

    ops = []
    for line in solver_code.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("def ") or line.startswith("return"):
            continue
        if "=" in line and "(" in line:
            target = line.split("=")[0].strip()
            rest = line.split("=", 1)[1].strip()
            func = rest.split("(")[0].strip()
            args_str = rest.split("(", 1)[1].rsplit(")", 1)[0]
            args = [resolve(a) for a in args_str.split(",")]
            ops.append((target, func, args))
    return ops

# ── Step 3: Build permutation array ──────────────────────────

def build_perm_indices(H, W, perm_fn):
    """Build 900-element permutation array for a spatial transform.
    
    For each output position (r, c), store the flat input index.
    Positions outside the H x W grid map to themselves (identity).
    """
    indices = np.arange(GRID * GRID, dtype=np.int64)
    for r in range(H):
        for c in range(W):
            r_in, c_in = perm_fn(r, c)
            indices[r * GRID + c] = r_in * GRID + c_in
    return indices

# ── Step 4: Build ONNX models ────────────────────────────────

def build_perm_model(perm_indices, task_name):
    """Build ONNX model: Reshape → Gather → Reshape."""
    perm_init = numpy_helper.from_array(perm_indices, name="perm")
    rs_shape = numpy_helper.from_array(np.array([1, 10, 900], dtype=np.int64), name="rs")
    out_shape = numpy_helper.from_array(np.array([1, 10, 30, 30], dtype=np.int64), name="os")

    nodes = [
        helper.make_node("Reshape", ["input", "rs"], ["flat"]),
        helper.make_node("Gather", ["flat", "perm"], ["permuted"], axis=2),
        helper.make_node("Reshape", ["permuted", "os"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes, task_name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=[perm_init, rs_shape, out_shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def build_replace_model(old_color, new_color, task_name):
    """Build ONNX model that replaces one color with another.
    
    Logic: channel_new = clip(channel_new + mask, 0, 1)
           channel_old = channel_old * (1 - mask)
           where mask = (channel_old == 1.0)
    """
    nodes = []
    inits = []

    def make_const(arr, name):
        init = numpy_helper.from_array(arr, name=name)
        inits.append(init)
        return name

    ch_old = make_const(np.array([old_color], dtype=np.int64), "ch_old")
    ch_new = make_const(np.array([new_color], dtype=np.int64), "ch_new")
    one_f = make_const(np.array(1.0, dtype=np.float32), "one")
    zero_f = make_const(np.array(0.0, dtype=np.float32), "zero")
    one_f2 = make_const(np.array(1.0, dtype=np.float32), "one2")

    # Extract the two channels
    nodes.append(helper.make_node("Gather", ["input", ch_old], ["co"], axis=1))
    nodes.append(helper.make_node("Gather", ["input", ch_new], ["cn"], axis=1))

    # Mask: where old channel is 1
    nodes.append(helper.make_node("Cast", ["co"], ["co_int"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Equal", ["co_int", make_const(np.array(1, dtype=np.int32), "one_i")], ["mask_bool"]))
    nodes.append(helper.make_node("Cast", ["mask_bool"], ["mask"], to=TensorProto.FLOAT))

    # New channel: clip(cn + mask)
    nodes.append(helper.make_node("Add", ["cn", "mask"], ["cn_plus"]))
    nodes.append(helper.make_node("Clip", ["cn_plus", zero_f, one_f], ["cn_new"]))

    # Old channel: zeroed where mask was 1
    nodes.append(helper.make_node("Sub", [one_f2, "mask"], ["inv_mask"]))
    nodes.append(helper.make_node("Mul", ["co", "inv_mask"], ["co_zeroed"]))

    # Rebuild all 10 channels
    all_ch = []
    for c in range(N_COLORS):
        idx = make_const(np.array([c], dtype=np.int64), f"idx_{c}")
        out_ch = f"ch_{c}"
        if c == old_color:
            nodes.append(helper.make_node("Identity", ["co_zeroed"], [out_ch]))
        elif c == new_color:
            nodes.append(helper.make_node("Identity", ["cn_new"], [out_ch]))
        else:
            nodes.append(helper.make_node("Gather", ["input", idx], [out_ch], axis=1))
        all_ch.append(out_ch)

    nodes.append(helper.make_node("Concat", all_ch, ["output"], axis=1))

    graph = helper.make_graph(
        nodes, task_name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def build_switch_model(a, b, task_name):
    """Build ONNX model that swaps colors a and b."""
    nodes = []
    inits = []

    def make_const(arr, name):
        init = numpy_helper.from_array(arr, name=name)
        inits.append(init)
        return name

    ch_a = make_const(np.array([a], dtype=np.int64), "ch_a")
    ch_b = make_const(np.array([b], dtype=np.int64), "ch_b")
    zero_f = make_const(np.array(0.0, dtype=np.float32), "zero")
    one_f = make_const(np.array(1.0, dtype=np.float32), "one")
    one_i = make_const(np.array(1, dtype=np.int32), "one_i")

    # Extract channels a and b
    nodes.append(helper.make_node("Gather", ["input", ch_a], ["ca"], axis=1))
    nodes.append(helper.make_node("Gather", ["input", ch_b], ["cb"], axis=1))

    # Masks
    for c_name, m_name in [("ca", "ma"), ("cb", "mb")]:
        nodes.append(helper.make_node("Cast", [c_name], [f"{c_name}_i"], to=TensorProto.INT32))
        nodes.append(helper.make_node("Equal", [f"{c_name}_i", one_i], [f"{m_name}_bool"]))
        nodes.append(helper.make_node("Cast", [f"{m_name}_bool"], [m_name], to=TensorProto.FLOAT))

    # Rebuild: channel a gets mask_b, channel b gets mask_a
    all_ch = []
    for c in range(N_COLORS):
        idx = make_const(np.array([c], dtype=np.int64), f"idx_{c}")
        out_ch = f"ch_{c}"
        if c == a:
            nodes.append(helper.make_node("Identity", ["mb"], [out_ch]))
        elif c == b:
            nodes.append(helper.make_node("Identity", ["ma"], [out_ch]))
        else:
            nodes.append(helper.make_node("Gather", ["input", idx], [out_ch], axis=1))
        all_ch.append(out_ch)

    nodes.append(helper.make_node("Concat", all_ch, ["output"], axis=1))

    graph = helper.make_graph(
        nodes, task_name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def build_switch_chain_model(pairs, task_name):
    """Build ONNX model that applies multiple switches in sequence.
    
    pairs: list of (a, b) color pairs to swap, in order.
    """
    nodes = []
    inits = []
    current = "input"

    def make_const(arr, name):
        init = numpy_helper.from_array(arr, name=name)
        inits.append(init)
        return name

    one_i = make_const(np.array(1, dtype=np.int32), "one_i")
    one_f = make_const(np.array(1.0, dtype=np.float32), "one_f")

    for i, (a, b) in enumerate(pairs):
        tag = f"_s{i}"

        ch_a = make_const(np.array([a], dtype=np.int64), f"ch_a{tag}")
        ch_b = make_const(np.array([b], dtype=np.int64), f"ch_b{tag}")

        # Extract channels
        nodes.append(helper.make_node("Gather", [current, ch_a], [f"ca{tag}"], axis=1))
        nodes.append(helper.make_node("Gather", [current, ch_b], [f"cb{tag}"], axis=1))

        # Masks
        nodes.append(helper.make_node("Cast", [f"ca{tag}"], [f"ca_i{tag}"], to=TensorProto.INT32))
        nodes.append(helper.make_node("Equal", [f"ca_i{tag}", one_i], [f"ma_b{tag}"]))
        nodes.append(helper.make_node("Cast", [f"ma_b{tag}"], [f"ma{tag}"], to=TensorProto.FLOAT))

        nodes.append(helper.make_node("Cast", [f"cb{tag}"], [f"cb_i{tag}"], to=TensorProto.INT32))
        nodes.append(helper.make_node("Equal", [f"cb_i{tag}", one_i], [f"mb_b{tag}"]))
        nodes.append(helper.make_node("Cast", [f"mb_b{tag}"], [f"mb{tag}"], to=TensorProto.FLOAT))

        # Rebuild channels
        all_ch = []
        for c in range(N_COLORS):
            idx = make_const(np.array([c], dtype=np.int64), f"idx{tag}_{c}")
            out_ch = f"ch{tag}_{c}"
            if c == a:
                nodes.append(helper.make_node("Identity", [f"mb{tag}"], [out_ch]))
            elif c == b:
                nodes.append(helper.make_node("Identity", [f"ma{tag}"], [out_ch]))
            else:
                nodes.append(helper.make_node("Gather", [current, idx], [out_ch], axis=1))
            all_ch.append(out_ch)

        next_out = f"cat{tag}"
        nodes.append(helper.make_node("Concat", all_ch, [next_out], axis=1))
        current = next_out

    nodes.append(helper.make_node("Identity", [current], ["output"]))

    graph = helper.make_graph(
        nodes, task_name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


# ── Step 5: Determine what to build ──────────────────────────

def classify_task(solver_code):
    """Classify a task by its DSL operations.
    
    Returns one of:
      ("spatial", op_name)         — e.g. ("spatial", "rot90")
      ("replace", old, new)        — e.g. ("replace", 2, 8)
      ("switch", a, b)             — e.g. ("switch", 1, 3)
      None                         — can't handle
    """
    ops = parse_solver_ops(solver_code)
    if not ops:
        return None

    # Find the output operation (O = ...)
    output_op = None
    for target, func, args in ops:
        if target == "O":
            output_op = (func, args)
            break
    if not output_op:
        return None

    func, args = output_op

    # Simple spatial: O = rot90(I), O = hmirror(I), etc.
    if func in SPATIAL_PERMS and len(args) == 1 and args[0] == "I":
        return ("spatial", func)

    # Replace: O = replace(x1, a, b) where x1 = <spatial>(I)
    if func == "replace" and len(args) == 3:
        # Check that the input to replace is a spatial transform of I
        input_var = args[0]
        if input_var != "I":
            # Check if input_var is defined as a spatial op on I
            for target, f2, a2 in ops:
                if target == input_var and f2 in SPATIAL_PERMS and len(a2) == 1 and a2[0] == "I":
                    return ("replace", args[1], args[2])
            return None
        return ("replace", args[1], args[2])

    # Switch: could be chained — collect all switch pairs leading to O
    if func == "switch" and len(args) == 3:
        # Walk backwards: O = switch(x3, a, b), x3 = switch(x2, c, d), ...
        pairs = []
        current_var = args[0]
        pairs.append((args[1], args[2]))

        # Check if input_var is I or another switch
        while current_var != "I":
            found = False
            for target, f2, a2 in ops:
                if target == current_var and f2 == "switch" and len(a2) == 3:
                    pairs.append((a2[1], a2[2]))
                    current_var = a2[0]
                    found = True
                    break
            if not found:
                # Check if it's a spatial op on I (single switch after spatial)
                for target, f2, a2 in ops:
                    if target == current_var and f2 in SPATIAL_PERMS and len(a2) == 1 and a2[0] == "I":
                        return ("switch_spatial", f2, args[1], args[2])
                return None

        pairs.reverse()
        return ("switch_chain", pairs)

    return None


# ── Step 6: Build model for a task ───────────────────────────

def build_task_model(task_key):
    """Build and save ONNX model for a task. Returns (path, error)."""
    with open(MATCHES_PATH) as f:
        matches = json.load(f)

    if task_key not in matches:
        return None, "no solver match"

    solver_name = matches[task_key]["solver"]
    if not solver_name:
        return None, "no solver"

    solver_code = get_solver_code(solver_name)
    if not solver_code:
        return None, "can't find solver code"

    classification = classify_task(solver_code)
    if classification is None:
        return None, "can't classify"

    kind = classification[0]

    # Color ops work for any grid size (per-cell, no spatial dependency)
    if kind == "replace":
        old_color, new_color = classification[1], classification[2]
        model = build_replace_model(old_color, new_color, task_key)
        path = OUTPUT_DIR / f"{task_key}.onnx"
        onnx.save(model, path)
        return str(path), None

    if kind == "switch":
        a, b = classification[1], classification[2]
        model = build_switch_model(a, b, task_key)
        path = OUTPUT_DIR / f"{task_key}.onnx"
        onnx.save(model, path)
        return str(path), None

    if kind == "switch_chain":
        pairs = classification[1]
        model = build_switch_chain_model(pairs, task_key)
        path = OUTPUT_DIR / f"{task_key}.onnx"
        onnx.save(model, path)
        return str(path), None

    # Spatial ops need fixed grid size (permutation depends on H, W)
    size = get_grid_size(task_key)
    if size is None:
        return None, "variable grid sizes"

    H, W = size

    if kind == "spatial":
        op_name = classification[1]
        perm_fn = SPATIAL_PERMS[op_name](H, W)
        indices = build_perm_indices(H, W, perm_fn)
        model = build_perm_model(indices, task_key)
    else:
        return None, f"unknown kind: {kind}"

    path = OUTPUT_DIR / f"{task_key}.onnx"
    onnx.save(model, path)
    return str(path), None


# ── Step 7: Verify ───────────────────────────────────────────

def verify_task(task_key, model_path):
    """Check model against all train+test examples."""
    import onnxruntime as ort

    with open(TASK_DIR / f"{task_key}.json") as f:
        task = json.load(f)

    sess = ort.InferenceSession(model_path)

    for i, ex in enumerate(task["train"] + task.get("test", [])):
        inp = np.array(ex["input"], dtype=np.float32)
        outp = np.array(ex["output"], dtype=np.float32)
        H_in, W_in = inp.shape
        H_out, W_out = outp.shape

        # Encode input as one-hot
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(H_in):
            for c in range(W_in):
                x[0, int(inp[r, c]), r, c] = 1.0

        # Run model
        out = sess.run(None, {"input": x})[0]
        pred = np.argmax(out[0], axis=0)

        # Compare
        for r in range(H_out):
            for c in range(W_out):
                if pred[r, c] != int(outp[r, c]):
                    return False, f"example {i} mismatch at ({r},{c})"

    return True, None


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(MATCHES_PATH) as f:
        matches = json.load(f)

    print(f"Processing {len(matches)} tasks...")
    print()

    success = []
    failed = []

    for task_key in sorted(matches.keys()):
        path, err = build_task_model(task_key)
        if path:
            ok, verify_err = verify_task(task_key, path)
            if ok:
                success.append(task_key)
                print(f"  OK  {task_key}")
            else:
                failed.append((task_key, verify_err))
                print(f"  FAIL {task_key}: {verify_err}")
        else:
            failed.append((task_key, err))

    print()
    print(f"Success: {len(success)}")
    print(f"Failed:  {len(failed)}")
    print()
    print("Successes:", success)
