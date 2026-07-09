#!/usr/bin/env python3
"""
Zero-Parameter Permutation Architecture for NeuroGolf 2026.
Implements D4 symmetry detection and ONNX model generation.

Pipeline:
1. Load task data from neurogolf-2026/
2. Test D4 symmetries on train examples
3. Build permutation indices for winning transformation
4. Generate ONNX model with Reshape + Transpose + Gather ops
5. Verify on test examples
6. Output .onnx files and submission.zip
"""

import json
import os
import sys
import zipfile
import numpy as np

TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neurogolf-2026")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
STRIDE = 30  # Fixed stride for 30x30 tensor


# ── D4 Transformation functions ──────────────────────────────────────────────

def identity(r, c, H, W):
    return (r, c)

def flip_lr(r, c, H, W):
    return (r, W - 1 - c)

def flip_ud(r, c, H, W):
    return (H - 1 - r, c)

def rotate_180(r, c, H, W):
    return (H - 1 - r, W - 1 - c)

def rotate_90(r, c, H, W):
    return (c, H - 1 - r)

def rotate_270(r, c, H, W):
    return (W - 1 - c, r)

def transpose(r, c, H, W):
    return (c, r)

def anti_transpose(r, c, H, W):
    return (W - 1 - c, H - 1 - r)

TRANSFORMS = {
    "identity": identity,
    "flip_lr": flip_lr,
    "flip_ud": flip_ud,
    "rotate_180": rotate_180,
    "rotate_90": rotate_90,
    "rotate_270": rotate_270,
    "transpose": transpose,
    "anti_transpose": anti_transpose,
}


# ── Step 1: Rule Induction ───────────────────────────────────────────────────

def load_task(task_num):
    filepath = os.path.join(TASKS_DIR, f"task{task_num:03d}.json")
    with open(filepath) as f:
        return json.load(f)


def test_transformation(transform_func, input_grid, output_grid):
    H = len(input_grid)
    W = len(input_grid[0])
    for r in range(H):
        for c in range(W):
            out_r, out_c = transform_func(r, c, H, W)
            if input_grid[r][c] != output_grid[out_r][out_c]:
                return False
    return True


def find_rule(examples):
    """Find D4 symmetry rule from examples. Returns (transform_name, H, W) or None."""
    sizes = set()
    for example in examples:
        H_in = len(example["input"])
        W_in = len(example["input"][0])
        H_out = len(example["output"])
        W_out = len(example["output"][0])
        if H_in != H_out or W_in != W_out:
            return None
        sizes.add((H_in, W_in))

    if len(sizes) != 1:
        return None

    H, W = sizes.pop()

    valid_transforms = ["identity", "flip_lr", "flip_ud", "rotate_180"]
    if H == W:
        valid_transforms.extend(["rotate_90", "rotate_270", "transpose", "anti_transpose"])

    for transform_name in valid_transforms:
        transform_func = TRANSFORMS[transform_name]
        if all(test_transformation(transform_func, ex["input"], ex["output"]) for ex in examples):
            return (transform_name, H, W)

    return None


# ── Step 2: Permutation Matrix compilation ────────────────────────────────────

def build_perm_indices(transform_func, H, W):
    """Build permutation indices array for 30x30 tensor."""
    N = 900  # 30 * 30
    perm = np.arange(N, dtype=np.int64)

    for r in range(H):
        for c in range(W):
            in_idx = r * STRIDE + c
            out_r, out_c = transform_func(r, c, H, W)
            out_idx = out_r * STRIDE + out_c
            perm[out_idx] = in_idx

    return perm


# ── Step 3: Static ONNX Graph Assembly ────────────────────────────────────────

def build_onnx_model(perm_indices):
    """
    Build ONNX graph:
        Input (1,10,30,30)
          -> Reshape (10,900)
          -> Transpose (900,10)
          -> Gather (900,10)  [apply permutation to spatial positions]
          -> Transpose (10,900)
          -> Reshape (1,10,30,30)
        Output
    """
    import onnx
    from onnx import helper, TensorProto

    N = 900

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    reshape1_shape = helper.make_tensor("reshape1_shape", TensorProto.INT64, [2], [10, N])
    reshape2_shape = helper.make_tensor("reshape2_shape", TensorProto.INT64, [4], [1, 10, 30, 30])
    perm_tensor = helper.make_tensor("perm_indices", TensorProto.INT64, [N], perm_indices.tolist())

    nodes = [
        helper.make_node("Reshape", ["input", "reshape1_shape"], ["reshaped1"]),
        helper.make_node("Transpose", ["reshaped1"], ["transposed1"], perm=[1, 0]),
        helper.make_node("Gather", ["transposed1", "perm_indices"], ["gathered"], axis=0),
        helper.make_node("Transpose", ["gathered"], ["transposed2"], perm=[1, 0]),
        helper.make_node("Reshape", ["transposed2", "reshape2_shape"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes, "permutation_graph",
        [input_tensor], [output_tensor],
        [reshape1_shape, reshape2_shape, perm_tensor],
    )

    model = helper.make_model(
        graph, ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    return model


# ── Step 4: Verification & Submission ─────────────────────────────────────────

def grid_to_onehot(grid, H, W):
    tensor = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            tensor[0, grid[r][c], r, c] = 1.0
    return tensor


def verify_model(model_path, examples):
    import onnxruntime

    session = onnxruntime.InferenceSession(model_path)
    correct = 0
    total = 0

    for example in examples:
        H = len(example["input"])
        W = len(example["input"][0])
        inp = grid_to_onehot(example["input"], H, W)
        expected = grid_to_onehot(example["output"], H, W)
        output = session.run(["output"], {"input": inp})[0]
        if np.allclose(output, expected, atol=1e-6):
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
            train_examples = task_data["train"]
            rule = find_rule(train_examples)

            if rule is None:
                print("no D4 symmetry")
                skipped.append(task_num)
                continue

            transform_name, H, W = rule
            print(f"{transform_name} ({H}x{W})", end=" ")

            transform_func = TRANSFORMS[transform_name]
            perm_indices = build_perm_indices(transform_func, H, W)
            model = build_onnx_model(perm_indices)
            model_path = os.path.join(OUTPUT_DIR, f"task{task_num:03d}.onnx")

            import onnx
            onnx.save(model, model_path)

            test_examples = task_data.get("test", [])
            if test_examples:
                correct, total = verify_model(model_path, test_examples)
                if correct == total:
                    print(f"OK ({correct}/{total})")
                    solved.append(task_num)
                else:
                    print(f"FAIL ({correct}/{total})")
                    os.remove(model_path)
                    skipped.append(task_num)
            else:
                print("OK (no test)")
                solved.append(task_num)

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
