#!/usr/bin/env python3
"""
Strategy 2: Handcrafted Analytical Convolutional Gate System.

Uses Conv2d kernels to implement boolean logic (AND, OR, NOT) for ARC tasks.
Pipeline:
1. Analyze train examples to detect rules (color mapping, neighborhood patterns)
2. Build Conv2d kernels that implement the detected logic
3. Assemble ONNX graph: Conv2d -> Bias -> ReLU -> 1x1 Conv -> Output
4. Verify and output submission.zip
"""

import json
import os
import zipfile
import numpy as np

TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neurogolf-2026")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
GRID_H, GRID_W = 30, 30
CHANNELS = 10


def load_task(task_num):
    with open(os.path.join(TASKS_DIR, f"task{task_num:03d}.json")) as f:
        return json.load(f)


def grid_to_onehot(grid):
    """Convert a 2D grid (list of lists) to (1, 10, 30, 30) one-hot tensor."""
    h, w = len(grid), len(grid[0])
    t = np.zeros((1, CHANNELS, GRID_H, GRID_W), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            t[0, grid[r][c], r, c] = 1.0
    return t


# ── Rule Detection ────────────────────────────────────────────────────────────

def detect_color_mapping(examples):
    """
    Detect simple color mapping: for each non-zero input color, what output color
    does it map to? Must be consistent across all demos.
    Returns dict {input_color: output_color} or None.
    """
    mapping = {}

    for ex in examples:
        grid_in = ex["input"]
        grid_out = ex["output"]
        h, w = len(grid_in), len(grid_in[0])
        h2, w2 = len(grid_out), len(grid_out[0])
        if h != h2 or w != w2:
            return None

        for r in range(h):
            for c in range(w):
                ic = grid_in[r][c]
                oc = grid_out[r][c]
                if ic == 0 and oc == 0:
                    continue
                if ic == 0 or oc == 0:
                    return None
                if ic in mapping:
                    if mapping[ic] != oc:
                        return None
                else:
                    mapping[ic] = oc

    if not mapping:
        return None
    return mapping


def detect_neighbor_rules(examples):
    """
    Detect neighborhood-based rules. For each pixel that changes color,
    analyze what triggered the change based on neighbors.

    Returns list of rule dicts, or None.
    Each rule: {src_color, dst_color, neighbor_color, min_count, mode}
    where mode is 'at_least' or 'exactly'.
    """
    all_rules = []

    for ex in examples:
        grid_in = ex["input"]
        grid_out = ex["output"]
        h, w = len(grid_in), len(grid_in[0])
        h2, w2 = len(grid_out), len(grid_out[0])
        if h != h2 or w != w2:
            return None

        for r in range(h):
            for c in range(w):
                ic = grid_in[r][c]
                oc = grid_out[r][c]
                if ic == oc:
                    continue

                # Count 4-connected neighbors by color
                neighbor_counts = {}
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        nv = grid_in[nr][nc]
                        neighbor_counts[nv] = neighbor_counts.get(nv, 0) + 1

                all_rules.append({
                    "pos": (r, c),
                    "src": ic, "dst": oc,
                    "neighbors": neighbor_counts,
                    "grid_h": h, "grid_w": w,
                })

    if not all_rules:
        return None

    # Try to find a consistent rule among all changed pixels
    # Group by (src, dst) pair
    from collections import defaultdict
    groups = defaultdict(list)
    for rule in all_rules:
        groups[(rule["src"], rule["dst"])].append(rule)

    # For each (src, dst) group, find the most common neighbor pattern
    best_rules = []
    for (src, dst), rules in groups.items():
        # Count how many times each neighbor color appears and with what count
        neighbor_patterns = []
        for rule in rules:
            # Non-zero, non-src neighbor colors and their counts
            pattern = {}
            for nc, cnt in rule["neighbors"].items():
                if nc != 0 and nc != src:
                    pattern[nc] = cnt
            neighbor_patterns.append(tuple(sorted(pattern.items())))

        from collections import Counter
        common = Counter(neighbor_patterns).most_common(1)[0]
        pattern, count = common

        if count < len(rules) * 0.5:
            continue  # pattern not consistent enough

        # Convert pattern to rule
        for nc, min_cnt in pattern:
            best_rules.append({
                "src_color": src,
                "dst_color": dst,
                "neighbor_color": nc,
                "min_count": min_cnt,
                "mode": "at_least",
            })

    return best_rules if best_rules else None


def analyze_task(task_data):
    """Analyze a task and return the best strategy."""
    train = task_data["train"]
    test = task_data.get("test", [])

    # Try color mapping first
    all_exs = train + test
    color_map = detect_color_mapping(all_exs)
    if color_map is not None:
        return ("color_map", color_map)

    # Try neighborhood rules on train
    rules = detect_neighbor_rules(train)
    if rules:
        return ("neighborhood", rules)

    return None


# ── ONNX Model Building ──────────────────────────────────────────────────────

def build_color_map_model(color_map, grid_h, grid_w):
    """
    Build ONNX model for simple color mapping.
    Uses a 1x1 Conv2d that remaps input channels to output channels.

    color_map: {input_color: output_color}
    """
    import onnx
    from onnx import helper, TensorProto

    # Build weight matrix: output_channel <- input_channel
    W = np.zeros((CHANNELS, CHANNELS, 1, 1), dtype=np.float32)
    for ic in range(CHANNELS):
        oc = color_map.get(ic, ic)  # default: identity
        W[oc, ic, 0, 0] = 1.0

    # Verify: each output channel gets at most 1 input (valid one-hot)
    col_sums = W.reshape(CHANNELS, CHANNELS).sum(axis=0)
    if np.any(col_sums > 1.0 + 1e-6):
        return None

    weight_data = [float(x) for x in W.flatten()]
    weight_tensor = helper.make_tensor(
        "W", TensorProto.FLOAT,
        [CHANNELS, CHANNELS, 1, 1],
        weight_data,
    )

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                                  [1, CHANNELS, GRID_H, GRID_W])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                                   [1, CHANNELS, GRID_H, GRID_W])

    node = helper.make_node("Conv", ["input", "W"], ["output"],
                            kernel_shape=[1, 1], pads=[0, 0, 0, 0])

    graph = helper.make_graph([node], "color_map", [input_tensor], [output_tensor],
                               [weight_tensor])
    model = helper.make_model(graph, ir_version=10,
                               opset_imports=[helper.make_opsetid("", 10)])
    return model


def build_neighborhood_model(rules, grid_h, grid_w):
    """
    Build ONNX model for neighborhood-based rules.

    Architecture (per rule):
        Input (1, 10, 30, 30)
          -> Conv2d: count neighbor_color neighbors (cross kernel)
          -> Add: bias = -(min_count - 0.5)  (threshold gate)
          -> ReLU: clip negative
          -> Broadcast to 10 channels via 1x1 Conv
        Output (1, 10, 30, 30)

    For the trigger pixel itself, we use a large weight on the src_color
    channel center to act as a mask (only trigger pixels activate).
    """
    import onnx
    from onnx import helper, TensorProto

    if not rules:
        return None

    # Use the first rule (most common pattern)
    rule = rules[0]
    src = rule["src_color"]
    dst = rule["dst_color"]
    ncol = rule["neighbor_color"]
    min_cnt = rule["min_count"]

    # ── Conv2d kernel: detect trigger pixel with N neighbors of ncol ──
    W_conv = np.zeros((1, CHANNELS, 3, 3), dtype=np.float32)
    W_conv[0, src, 1, 1] = 100.0      # trigger mask (large = pixel must be src)
    W_conv[0, ncol, 0, 1] = 1.0       # top neighbor
    W_conv[0, ncol, 1, 0] = 1.0       # left neighbor
    W_conv[0, ncol, 1, 2] = 1.0       # right neighbor
    W_conv[0, ncol, 2, 1] = 1.0       # bottom neighbor

    # Bias: threshold so only trigger pixels with >= min_cnt neighbors pass ReLU
    bias_val = -(100.0 + min_cnt - 0.5)

    # ── Build one_hot(dst) - one_hot(src) as (1, 10, 1, 1) ──
    diff = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    diff[0, dst, 0, 0] = 1.0
    diff[0, src, 0, 0] = -1.0

    # ── Initializer tensors ──
    wconv_tensor = helper.make_tensor(
        "W_conv", TensorProto.FLOAT, [1, CHANNELS, 3, 3],
        [float(x) for x in W_conv.flatten()],
    )
    bias_tensor = helper.make_tensor(
        "bias", TensorProto.FLOAT, [1], [float(bias_val)],
    )
    diff_tensor = helper.make_tensor(
        "color_diff", TensorProto.FLOAT, [1, CHANNELS, 1, 1],
        [float(x) for x in diff.flatten()],
    )

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                                  [1, CHANNELS, GRID_H, GRID_W])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                                   [1, CHANNELS, GRID_H, GRID_W])

    # ── Graph: output = input + mask * color_diff ──
    # 1. Conv2d -> raw feature
    # 2. Add bias -> thresholded
    # 3. ReLU -> binary mask (1,1,H,W)
    # 4. Mul(mask, color_diff) -> delta (1,10,H,W) via broadcast
    # 5. Add(input, delta) -> output
    nodes = [
        helper.make_node("Conv", ["input", "W_conv"], ["raw_feat"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Add", ["raw_feat", "bias"], ["biased"]),
        helper.make_node("Relu", ["biased"], ["mask"]),
        helper.make_node("Mul", ["mask", "color_diff"], ["delta"]),
        helper.make_node("Add", ["input", "delta"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes, "neighbor_rule",
        [input_tensor], [output_tensor],
        initializer=[wconv_tensor, bias_tensor, diff_tensor],
    )

    model = helper.make_model(graph, ir_version=10,
                               opset_imports=[helper.make_opsetid("", 10)])
    return model


def build_model(task_data, strategy, grid_h, grid_w):
    """Build ONNX model based on detected strategy."""
    if strategy[0] == "color_map":
        return build_color_map_model(strategy[1], grid_h, grid_w)
    elif strategy[0] == "neighborhood":
        return build_neighborhood_model(strategy[1], grid_h, grid_w)
    return None


def verify_model(model_path, examples):
    """Verify ONNX model on examples. Returns (correct, total)."""
    import onnxruntime

    session = onnxruntime.InferenceSession(model_path)
    correct = 0
    total = 0

    for ex in examples:
        h, w = len(ex["input"]), len(ex["input"][0])
        if h > GRID_H or w > GRID_W:
            continue

        inp = grid_to_onehot(ex["input"])
        expected = grid_to_onehot(ex["output"])

        output = session.run(["output"], {"input": inp})[0]
        # Competition uses threshold at 0.0
        output_bin = (output > 0.0).astype(np.float32)

        if np.array_equal(output_bin, expected):
            correct += 1
        total += 1

    return correct, total


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    task_files = sorted(
        f for f in os.listdir(TASKS_DIR)
        if f.startswith("task") and f.endswith(".json")
    )

    solved, skipped = [], []

    for task_file in task_files:
        task_num = int(task_file.replace("task", "").replace(".json", ""))
        print(f"task {task_num:03d}: ", end="", flush=True)

        try:
            task_data = load_task(task_num)
            train = task_data["train"]
            h0 = len(train[0]["input"])
            w0 = len(train[0]["input"][0])
            if h0 > GRID_H or w0 > GRID_W:
                print(f"grid too large ({h0}x{w0})")
                skipped.append(task_num)
                continue

            strategy = analyze_task(task_data)
            if strategy is None:
                print("no detectable rule")
                skipped.append(task_num)
                continue

            stype = strategy[0]
            print(f"{stype}", end=" ")

            model = build_model(task_data, strategy, h0, w0)
            if model is None:
                print("(model build failed)")
                skipped.append(task_num)
                continue

            import onnx
            model_path = os.path.join(OUTPUT_DIR, f"task{task_num:03d}.onnx")
            onnx.save(model, model_path)

            test = task_data.get("test", [])
            correct, total = verify_model(model_path, train + test)

            if correct == total:
                print(f"OK ({correct}/{total})")
                solved.append(task_num)
            else:
                print(f"FAIL ({correct}/{total})")
                os.remove(model_path)
                skipped.append(task_num)

        except Exception as e:
            print(f"ERROR: {e}")
            skipped.append(task_num)

    print(f"\nSolved: {len(solved)} / Skipped: {len(skipped)}")
    print(f"Solved tasks: {solved}")

    if solved:
        zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for task_num in solved:
                zf.write(
                    os.path.join(OUTPUT_DIR, f"task{task_num:03d}.onnx"),
                    f"task{task_num:03d}.onnx",
                )
        print(f"Created {zip_path}")


if __name__ == "__main__":
    main()
