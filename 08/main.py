"""
NeuroGolf Main: Compile all 400 tasks to ONNX and build submission.zip.

Strategies:
  1. Compiler: for tasks using only simple grid ops (spatial, color, crop, etc.)
  2. CNN train: for everything else — run Python solver to generate data,
     train a small CNN, export to ONNX.

Usage:
    python main.py                        # compile all tasks
    python main.py --tasks 1-10           # compile specific range
    python main.py --tasks 1,5,10         # compile specific tasks
    python main.py --verify-only          # verify existing ONNX files
    python main.py --build-zip            # build submission.zip
    python main.py --tasks 1-5 --build-zip  # compile + zip
"""

import json
import sys
import time
import zipfile
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# ============================================================================
# Config
# ============================================================================

ROOT = Path(__file__).parent.parent
TASKS_DIR = ROOT / "07" / "tasks"
META_PATH = ROOT / "07" / "tasks_meta.json"
OUTPUT_DIR = Path(__file__).parent / "onnx_output"
SUBMISSION_PATH = Path(__file__).parent / "submission.zip"
MAX_TASKS = 400
CANVAS_SIZE = 30
NUM_COLORS = 10


# ============================================================================
# Helpers
# ============================================================================

def load_meta() -> Dict:
    with open(META_PATH) as f:
        return json.load(f)


def load_task(task_num: int) -> dict:
    path = TASKS_DIR / f"task{task_num:03d}.json"
    with open(path) as f:
        return json.load(f)


def encode_grid(grid: list) -> "np.ndarray":
    """Encode Python grid to one-hot numpy array shape (1,10,30,30)."""
    import numpy as np
    arr = np.array(grid, dtype=np.int64)
    H, W = arr.shape
    oh = np.zeros((1, NUM_COLORS, CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    for c in range(NUM_COLORS):
        oh[0, c, :H, :W] = (arr == c).astype(np.float32)
    return oh


def grids_match(pred_oh, expected_oh) -> bool:
    """Check if predicted one-hot matches expected."""
    import numpy as np
    return (pred_oh.argmax(axis=1) == expected_oh.argmax(axis=1)).all()


def verify_onnx(onnx_path: str, task_data: dict) -> Tuple[bool, int, int]:
    """Verify ONNX on ALL train+test examples.
    Returns (all_match, n_pass, n_total).
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    n_pass = 0
    n_total = 0

    for split in ("train", "test"):
        for example in task_data.get(split, []):
            n_total += 1
            input_oh = encode_grid(example["input"])
            output_oh = encode_grid(example["output"])

            onnx_out = sess.run(None, {input_name: input_oh})[0]
            if grids_match(onnx_out, output_oh):
                n_pass += 1

    return n_pass == n_total, n_pass, n_total


def size_ok(onnx_path: str, max_bytes: int = 1_440_000) -> bool:
    return Path(onnx_path).stat().st_size <= max_bytes


# ============================================================================
# Strategy 1: Compiler (simple grid ops)
# ============================================================================

def get_task_grid_sizes(task_data: dict):
    """Get fixed grid sizes for a task. Returns (h_in, w_in) or (None, None)."""
    in_sizes = set()
    for split in ("train", "test"):
        for ex in task_data.get(split, []):
            in_sizes.add((len(ex["input"]), len(ex["input"][0])))
    if len(in_sizes) == 1:
        return in_sizes.pop()
    return None, None


def try_compiler(task_num: int, meta: Dict, output_dir: Path,
                  task_data: dict = None) -> Optional[str]:
    """Try to compile using the grid-level compiler."""
    try:
        from compiler import compile_solver
        from solver_parser import extract_solver

        task_key = f"task{task_num:03d}"
        solver_name = meta.get(task_key, {}).get("solver", "")
        if not solver_name:
            return None

        source = extract_solver(solver_name)

        h_in, w_in = 30, 30
        if task_data:
            h, w = get_task_grid_sizes(task_data)
            if h is not None:
                h_in, w_in = h, w

        return compile_solver(source, task_num, str(output_dir), h_in=h_in, w_in=w_in)
    except Exception:
        return None


# ============================================================================
# Strategy 2: CNN training
# ============================================================================

def try_cnn_train(task_num: int, meta: Dict, output_dir: Path,
                   max_epochs: int = 800, lr: float = 5e-3) -> Optional[str]:
    """Run Python solver → generate data → train CNN → export ONNX."""
    import torch
    import torch.nn as nn
    import numpy as np

    task_key = f"task{task_num:03d}"
    solver_name = meta.get(task_key, {}).get("solver", "")
    if not solver_name:
        return None

    try:
        from run_solver import run_solver
    except Exception:
        return None

    task_data = load_task(task_num)

    # Generate training data from ALL train+test examples
    inputs_np = []
    outputs_np = []

    for split in ("train", "test"):
        for example in task_data.get(split, []):
            input_grid = example["input"]
            expected_output = example["output"]

            try:
                result = run_solver(solver_name, input_grid)
                if result != expected_output:
                    return None
            except Exception:
                return None

            inputs_np.append(encode_grid(input_grid))
            outputs_np.append(encode_grid(expected_output))

    if not inputs_np:
        return None

    X = torch.from_numpy(np.concatenate(inputs_np, axis=0))
    Y = torch.from_numpy(np.concatenate(outputs_np, axis=0))
    N = X.shape[0]

    # ResBlock + CNN
    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        def forward(self, x):
            return x + torch.relu(self.conv2(torch.relu(self.conv1(x))))

    class TaskCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(10, 32, 3, padding=1), nn.ReLU(inplace=True),
                ResBlock(32), ResBlock(32), ResBlock(32),
                nn.Conv2d(32, 10, 1),
            )
        def forward(self, x):
            return self.enc(x)

    model = TaskCNN()

    # Data augmentation
    X_aug = [X]
    Y_aug = [Y]
    X_aug.append(torch.flip(X, [2])); Y_aug.append(torch.flip(Y, [2]))
    X_aug.append(torch.flip(X, [3])); Y_aug.append(torch.flip(Y, [3]))
    X_aug.append(torch.rot90(X, 1, [2, 3])); Y_aug.append(torch.rot90(Y, 1, [2, 3]))
    X_all = torch.cat(X_aug, dim=0)
    Y_all = torch.cat(Y_aug, dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        pred = model(X_all)
        target = Y_all.argmax(dim=1)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == max_epochs - 1:
            with torch.no_grad():
                train_pred = model(X)
                train_acc = (train_pred.argmax(dim=1) == Y.argmax(dim=1)).float().mean().item()
            print(f"    epoch {epoch:4d}/{max_epochs}  loss={loss.item():.6f}  train_acc={train_acc:.4f}")

        if loss.item() < 1e-6:
            print(f"    converged at epoch {epoch}  loss={loss.item():.8f}")
            break

    # Final accuracy check on original data
    model.eval()
    with torch.no_grad():
        pred = model(X)
        acc = (pred.argmax(dim=1) == Y.argmax(dim=1)).float().mean().item()

    if acc < 1.0:
        print(f"    FAILED: train_acc={acc:.4f} < 1.0")
        return None

    # Export to ONNX
    onnx_path = str(output_dir / f"task{task_num:03d}.onnx")
    dummy = torch.zeros(1, NUM_COLORS, CANVAS_SIZE, CANVAS_SIZE)
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    # Fix: inline external data so the .onnx is self-contained
    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.save_model(m, onnx_path)
        # Clean up any .data files
        data_file = Path(onnx_path).with_suffix(".onnx.data")
        if data_file.exists():
            data_file.unlink()
    except Exception:
        pass

    return onnx_path


# ============================================================================
# Orchestrator
# ============================================================================

def compile_task(task_num: int, meta: Dict, output_dir: Path,
                 verbose: bool = True) -> Tuple[Optional[str], str]:
    """Try all strategies for a task."""
    task_data = load_task(task_num)
    solver_name = meta.get(f"task{task_num:03d}", {}).get("solver", "")

    if verbose:
        in_sizes = set()
        out_sizes = set()
        for ex in task_data.get("train", []) + task_data.get("test", []):
            in_sizes.add(f"{len(ex['input'])}x{len(ex['input'][0])}")
            out_sizes.add(f"{len(ex['output'])}x{len(ex['output'][0])}")
        n_examples = len(task_data.get("train", [])) + len(task_data.get("test", []))
        print(f"  task{task_num:03d}  solver={solver_name}  examples={n_examples}  in={in_sizes}  out={out_sizes}")

    # Strategy 1: compiler
    if verbose:
        print(f"    trying compiler...")
    path = try_compiler(task_num, meta, output_dir, task_data=task_data)
    if path and size_ok(path):
        if verbose:
            print(f"    -> compiler OK ({Path(path).stat().st_size/1024:.1f} KB)")
        return path, "compiler"
    if verbose and path:
        print(f"    -> compiler produced model but verification will check")

    # Strategy 2: CNN training
    if verbose:
        print(f"    trying CNN training...")
    path = try_cnn_train(task_num, meta, output_dir)
    if path and size_ok(path):
        if verbose:
            print(f"    -> CNN OK ({Path(path).stat().st_size/1024:.1f} KB)")
        return path, "cnn"

    if verbose:
        print(f"    -> FAILED")
    return None, "failed"


def compile_all(tasks: List[int] = None, output_dir: Path = OUTPUT_DIR):
    """Compile all (or specified) tasks to ONNX."""
    output_dir.mkdir(exist_ok=True)
    meta = load_meta()

    if tasks is None:
        tasks = list(range(1, MAX_TASKS + 1))

    total = len(tasks)
    results = {
        "verified": [],
        "failed": [],
        "strategies": {},
    }

    t0 = time.time()

    for i, tn in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] task{tn:03d}")
        print(f"{'='*60}")

        task_t0 = time.time()

        # Skip if already have verified ONNX
        existing = output_dir / f"task{tn:03d}.onnx"
        if existing.exists():
            try:
                task_data = load_task(tn)
                ok, n_pass, n_total = verify_onnx(str(existing), task_data)
                if ok:
                    print(f"  already verified ({n_pass}/{n_total})")
                    results["verified"].append(tn)
                    continue
            except Exception:
                pass
            existing.unlink(missing_ok=True)

        # Compile
        onnx_path, strategy = compile_task(tn, meta, output_dir)

        if onnx_path is None:
            results["failed"].append(tn)
            task_elapsed = time.time() - task_t0
            print(f"  FAILED in {task_elapsed:.1f}s")
            continue

        # Verify on ALL train+test examples
        print(f"  verifying on all examples...")
        try:
            task_data = load_task(tn)
            ok, n_pass, n_total = verify_onnx(onnx_path, task_data)
            task_elapsed = time.time() - task_t0
            if ok:
                print(f"  PASS {n_pass}/{n_total}  strategy={strategy}  {task_elapsed:.1f}s")
                results["verified"].append(tn)
                results["strategies"][tn] = strategy
            else:
                print(f"  FAIL {n_pass}/{n_total}  {task_elapsed:.1f}s")
                results["failed"].append(tn)
                Path(onnx_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"  VERIFY ERROR: {e}")
            results["failed"].append(tn)
            Path(onnx_path).unlink(missing_ok=True)

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total:    {total}")
    print(f"  Verified: {len(results['verified'])}")
    print(f"  Failed:   {len(results['failed'])}")
    print(f"  Time:     {elapsed:.1f}s")

    strat_counts = {}
    for tn in results["verified"]:
        s = results["strategies"].get(tn, "unknown")
        strat_counts[s] = strat_counts.get(s, 0) + 1
    for s, c in sorted(strat_counts.items()):
        print(f"    {s}: {c}")

    if results["failed"]:
        print(f"\n  Failed tasks: {results['failed']}")

    return results


def build_submission(output_dir: Path = OUTPUT_DIR, submission_path: Path = SUBMISSION_PATH):
    """Build submission.zip from all ONNX files in output directory."""
    onnx_files = sorted(output_dir.glob("task*.onnx"))

    if not onnx_files:
        print("No ONNX files to package!")
        return

    with zipfile.ZipFile(submission_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in onnx_files:
            zf.write(f, f.name)

    print(f"\nSubmission built: {submission_path}")
    print(f"  Files: {len(onnx_files)}")
    print(f"  Size: {submission_path.stat().st_size / 1024:.1f} KB")


def verify_all(output_dir: Path = OUTPUT_DIR):
    """Verify all ONNX files in the output directory."""
    meta = load_meta()
    onnx_files = sorted(output_dir.glob("task*.onnx"))

    total = len(onnx_files)
    verified = 0
    failed = []

    for f in onnx_files:
        tn = int(f.stem.replace("task", ""))
        try:
            task_data = load_task(tn)
            ok, n_pass, n_total = verify_onnx(str(f), task_data)
            status = "PASS" if ok else "FAIL"
            print(f"  task{tn:03d}: {status} {n_pass}/{n_total}")
            if ok:
                verified += 1
            else:
                failed.append((tn, n_pass, n_total))
        except Exception as e:
            print(f"  task{tn:03d}: ERROR {e}")
            failed.append((tn, 0, 0))

    print(f"\nVerification: {verified}/{total} pass")
    return verified, total


# ============================================================================
# CLI
# ============================================================================

def parse_task_range(s: str) -> List[int]:
    """Parse task range string like '1-10', '1,5,10', '1-5,10,20-30'."""
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
    parser = argparse.ArgumentParser(description="NeuroGolf ONNX Compiler")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Task range, e.g. '1-10' or '1,5,10' or '1-5,20-30'")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing ONNX files")
    parser.add_argument("--build-zip", action="store_true",
                        help="Build submission.zip from ONNX files")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for ONNX files")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.verify_only:
        verify_all(output_dir)
        return

    tasks = parse_task_range(args.tasks) if args.tasks else None

    results = compile_all(tasks, output_dir)

    if args.build_zip:
        build_submission(output_dir)

    return results


if __name__ == "__main__":
    main()
