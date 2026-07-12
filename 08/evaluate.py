"""
NeuroGolf Evaluator: Test all ONNX models and compute competition metrics.

Usage:
    python evaluate.py                  # evaluate all models in onnx_output/
    python evaluate.py --tasks 1-10     # evaluate specific range
    python evaluate.py --onnx-dir DIR   # custom ONNX directory
"""

import json
import math
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import onnxruntime as ort

# ============================================================================
# Config
# ============================================================================

ROOT = Path(__file__).parent.parent
TASKS_DIR = ROOT / "07" / "tasks"
META_PATH = ROOT / "07" / "tasks_meta.json"
ONNX_DIR = Path(__file__).parent / "onnx_output"
CANVAS_SIZE = 30
NUM_COLORS = 10
EXCLUDED_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


# ============================================================================
# Encoding / Decoding
# ============================================================================

def encode_grid(grid: list) -> np.ndarray:
    """Encode Python grid to (1, 10, 30, 30) one-hot float32."""
    arr = np.array(grid, dtype=np.int64)
    H, W = arr.shape
    oh = np.zeros((1, NUM_COLORS, CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    for c in range(NUM_COLORS):
        oh[0, c, :H, :W] = (arr == c).astype(np.float32)
    return oh


def decode_output(onnx_out: np.ndarray, expected_shape: tuple = None) -> list:
    """Decode (1,10,30,30) output to Python grid, trimming to expected shape if given."""
    grid = onnx_out[0].argmax(axis=0)  # (30,30)
    if expected_shape:
        H, W = expected_shape
        grid = grid[:H, :W]
    return grid.tolist()


def grids_match(pred: np.ndarray, expected: np.ndarray) -> bool:
    """Check if two one-hot arrays match (after argmax)."""
    return (pred.argmax(axis=1) == expected.argmax(axis=1)).all()


def pixel_accuracy(pred: np.ndarray, expected: np.ndarray) -> float:
    """Per-pixel accuracy (fraction of correct cells)."""
    return (pred.argmax(axis=1) == expected.argmax(axis=1)).float().mean().item()


# ============================================================================
# Model validation helpers (competition rules)
# ============================================================================

def check_ops_safe(onnx_path: str) -> Tuple[bool, List[str]]:
    """Check model doesn't use banned ops. Returns (safe, list_of_banned_ops_found)."""
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    banned_found = []
    for node in model.graph.node:
        if node.op_type.upper() in EXCLUDED_OPS:
            banned_found.append(node.op_type)
        if "Sequence" in node.op_type:
            banned_found.append(node.op_type)
    return len(banned_found) == 0, banned_found


def check_filesize(onnx_path: str) -> Tuple[bool, int]:
    """Check file size. Returns (ok, size_bytes)."""
    sz = Path(onnx_path).stat().st_size
    return sz <= 1_440_000, sz


def count_params(onnx_path: str) -> Optional[int]:
    """Count model parameters (initializers + constants)."""
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    params = 0
    for init in model.graph.initializer:
        if any(d <= 0 for d in init.dims):
            return None
        n = 1
        for d in init.dims:
            n *= d
        params += n
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value' and attr.t.dims is not None:
                    n = 1
                    for d in attr.t.dims:
                        n *= d
                    params += n
    return params


def check_shapes_static(onnx_path: str) -> Tuple[bool, List[str]]:
    """Check all shapes are statically defined (no dynamic dims)."""
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    try:
        graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception:
        return False, ["shape inference failed"]
    problems = []
    for tensor in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not tensor.type.HasField("tensor_type"):
            continue
        tt = tensor.type.tensor_type
        if not tt.HasField("shape"):
            continue
        for dim in tt.shape.dim:
            if dim.HasField("dim_param"):
                problems.append(f"{tensor.name} has dynamic dim '{dim.dim_param}'")
    return len(problems) == 0, problems


def compute_score(memory: float, params: int) -> float:
    """Competition score: max(1, 25 - ln(memory + params))."""
    return max(1.0, 25.0 - math.log(max(1.0, memory + params)))


# ============================================================================
# Per-task evaluation
# ============================================================================

def load_meta() -> Dict:
    with open(META_PATH) as f:
        return json.load(f)


def load_task(task_num: int) -> dict:
    with open(TASKS_DIR / f"task{task_num:03d}.json") as f:
        return json.load(f)


def evaluate_task(task_num: int, onnx_path: str, task_data: dict) -> dict:
    """Full evaluation of one task. Returns results dict."""
    result = {
        "task": task_num,
        "onnx_path": str(onnx_path),
        "status": "error",
        "train_correct": 0,
        "train_total": 0,
        "test_correct": 0,
        "test_total": 0,
        "train_acc": 0.0,
        "test_acc": 0.0,
        "all_correct": False,
        "params": None,
        "filesize_ok": False,
        "filesize_bytes": 0,
        "ops_safe": False,
        "banned_ops": [],
        "shapes_static": False,
        "shape_issues": [],
        "estimated_score": 0.0,
        "error": None,
    }

    if not Path(onnx_path).exists():
        result["status"] = "missing"
        result["error"] = "ONNX file not found"
        return result

    # File size check
    fs_ok, fs_bytes = check_filesize(onnx_path)
    result["filesize_ok"] = fs_ok
    result["filesize_bytes"] = fs_bytes

    # Ops check
    ops_ok, banned = check_ops_safe(onnx_path)
    result["ops_safe"] = ops_ok
    result["banned_ops"] = banned

    # Shapes check
    shapes_ok, issues = check_shapes_static(onnx_path)
    result["shapes_static"] = shapes_ok
    result["shape_issues"] = issues

    # IO shapes from runtime
    try:
        _sess = ort.InferenceSession(onnx_path)
        result["input_shape"] = _sess.get_inputs()[0].shape
        result["output_shape"] = _sess.get_outputs()[0].shape
    except Exception:
        result["input_shape"] = None
        result["output_shape"] = None

    # Param count
    result["params"] = count_params(onnx_path)

    # Run inference on train + test
    try:
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name

        for split in ("train", "test"):
            correct = 0
            total = 0
            for example in task_data.get(split, []):
                total += 1
                input_oh = encode_grid(example["input"])
                expected_oh = encode_grid(example["output"])

                try:
                    out = sess.run(None, {input_name: input_oh})[0]
                    if grids_match(out, expected_oh):
                        correct += 1
                except Exception as e:
                    pass

            if split == "train":
                result["train_correct"] = correct
                result["train_total"] = total
                result["train_acc"] = correct / total if total > 0 else 0.0
            else:
                result["test_correct"] = correct
                result["test_total"] = total
                result["test_acc"] = correct / total if total > 0 else 0.0

        total_all = result["train_total"] + result["test_total"]
        correct_all = result["train_correct"] + result["test_correct"]
        result["all_correct"] = (correct_all == total_all and total_all > 0)

        # Estimate score (only meaningful if all correct)
        if result["all_correct"] and result["params"] is not None:
            # Memory is approximate — use filesize as proxy for runtime memory
            result["estimated_score"] = compute_score(fs_bytes, result["params"])
        elif result["all_correct"]:
            result["estimated_score"] = 1.0  # minimal score

        result["status"] = "ok"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


# ============================================================================
# Full evaluation
# ============================================================================

def evaluate_all(tasks: List[int] = None, onnx_dir: Path = ONNX_DIR) -> List[dict]:
    """Evaluate all (or specified) tasks."""
    meta = load_meta()

    if tasks is None:
        # Find all ONNX files
        onnx_files = sorted(onnx_dir.glob("task*.onnx"))
        tasks = [int(f.stem.replace("task", "")) for f in onnx_files]

    results = []
    t0 = time.time()

    for i, tn in enumerate(tasks):
        onnx_path = onnx_dir / f"task{tn:03d}.onnx"
        if not onnx_path.exists():
            results.append({
                "task": tn, "status": "missing",
                "train_acc": 0, "test_acc": 0, "all_correct": False
            })
            continue

        task_data = load_task(tn)
        r = evaluate_task(tn, str(onnx_path), task_data)
        results.append(r)

        # Print per-task
        if r["status"] == "ok":
            status = "PASS" if r["all_correct"] else "PARTIAL"
            params_str = f"{r['params']:,}" if r["params"] is not None else "N/A"
            shape_ok = "STATIC" if r["shapes_static"] else "DYNAMIC"
            ops_tag = "" if r["ops_safe"] else " [BANNED OPS!]"
            print(f"  task{tn:03d}  {status:8s}  train={r['train_correct']}/{r['train_total']}  "
                  f"test={r['test_correct']}/{r['test_total']}  "
                  f"params={params_str}  size={r['filesize_bytes']/1024:.1f}KB  "
                  f"shapes={shape_ok}{ops_tag}")
        elif r["status"] == "missing":
            print(f"  task{tn:03d}  MISSING")
        else:
            print(f"  task{tn:03d}  ERROR   {r.get('error', '')[:60]}")

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")

    total = len(results)
    missing = sum(1 for r in results if r["status"] == "missing")
    errors = sum(1 for r in results if r["status"] == "error")
    all_pass = sum(1 for r in results if r.get("all_correct"))
    partial = sum(1 for r in results if r["status"] == "ok" and not r.get("all_correct"))
    total_score = sum(r.get("estimated_score", 0) for r in results)

    print(f"  Total tasks:      {total}")
    print(f"  All correct:      {all_pass} ({100*all_pass/total:.1f}%)")
    print(f"  Partial:          {partial}")
    print(f"  Errors:           {errors}")
    print(f"  Missing ONNX:     {missing}")
    print(f"  Est. total score: {total_score:.1f} pts")
    print(f"  Time:             {elapsed:.1f}s")

    # Test accuracy breakdown
    test_results = [r for r in results if r["status"] == "ok" and r.get("test_total", 0) > 0]
    if test_results:
        test_correct_total = sum(r["test_correct"] for r in test_results)
        test_total_total = sum(r["test_total"] for r in test_results)
        print(f"\n  Test set accuracy: {test_correct_total}/{test_total_total} "
              f"({100*test_correct_total/test_total_total:.1f}%)" if test_total_total > 0 else "")

        # Per-task test accuracy distribution
        test_accs = [r["test_acc"] for r in test_results]
        perfect_test = sum(1 for a in test_accs if a == 1.0)
        print(f"  Perfect on test:  {perfect_test}/{len(test_results)}")

    # Validation issues
    ops_issues = [r["task"] for r in results if r.get("banned_ops")]
    shape_issues = [r["task"] for r in results if r.get("shape_issues")]
    size_issues = [r["task"] for r in results if not r.get("filesize_ok") and r["status"] == "ok"]

    if ops_issues:
        print(f"\n  Banned ops found: tasks {ops_issues}")
    if shape_issues:
        print(f"  Dynamic shapes:   tasks {shape_issues}")
    if size_issues:
        print(f"  Oversized files:  tasks {size_issues}")

    return results


def print_leaderboard(results: List[dict]):
    """Print sorted leaderboard of passing tasks."""
    passing = [r for r in results if r.get("all_correct")]
    if not passing:
        print("\nNo passing tasks.")
        return

    passing.sort(key=lambda r: -r.get("estimated_score", 0))

    print(f"\n{'='*70}")
    print(f"LEADERBOARD (tasks that pass all train+test examples)")
    print(f"{'='*70}")
    print(f"{'Rank':>4}  {'Task':>8}  {'Score':>7}  {'Params':>10}  {'Size':>8}")
    print(f"{'-'*4}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*8}")

    for rank, r in enumerate(passing, 1):
        params = f"{r['params']:,}" if r["params"] is not None else "N/A"
        print(f"{rank:4d}  task{r['task']:03d}  {r['estimated_score']:7.3f}  {params:>10}  {r['filesize_bytes']/1024:6.1f}KB")


# ============================================================================
# CLI
# ============================================================================

def parse_task_range(s: str) -> List[int]:
    tasks = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            tasks.extend(range(int(start), int(end) + 1))
        else:
            tasks.append(int(part))
    return sorted(set(tasks))


def main():
    parser = argparse.ArgumentParser(description="NeuroGolf Evaluator")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Task range, e.g. '1-10' or '1,5,10'")
    parser.add_argument("--onnx-dir", type=str, default=str(ONNX_DIR),
                        help="Directory containing ONNX files")

    args = parser.parse_args()
    onnx_dir = Path(args.onnx_dir)
    tasks = parse_task_range(args.tasks) if args.tasks else None

    results = evaluate_all(tasks, onnx_dir)
    print_leaderboard(results)


if __name__ == "__main__":
    main()
