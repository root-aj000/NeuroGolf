#!/usr/bin/env python3
"""
Strategy 6A: Pseudo-Inverse Linear Mapping
For small-grid tasks, embed a W matrix that maps input one-hot to output one-hot.
Uses per-task hardcoded content bounds since ONNX requires static shapes.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import json
import os
import sys

SHAPE = [1, 10, 30, 30]
MAX_MODEL_BYTES = 1_440_000


def grid_to_onehot(grid, h=30, w=30):
    oh = np.zeros((10, h, w), dtype=np.float32)
    for r in range(min(len(grid), h)):
        for c in range(min(len(grid[0]), w)):
            v = grid[r][c]
            if 0 <= v < 10:
                oh[v, r, c] = 1.0
    return oh


def try_pseudo_inverse_bounded(task_id, max_h=None, max_w=None):
    """Try pseudo-inverse with a specific content bounding box."""
    td = json.load(open(f'../neurogolf-2026/{task_id}.json'))

    if max_h is None or max_w is None:
        # Auto-detect: find bounding box of all non-zero content across all examples
        max_h, max_w = 0, 0
        for ex in td['train'] + td['test']:
            for grid_key in ['input', 'output']:
                g = ex[grid_key]
                for r in range(len(g)):
                    for c in range(len(g[0])):
                        if g[r][c] != 0:
                            max_h = max(max_h, r + 1)
                            max_w = max(max_w, c + 1)
        max_h = max(max_h, 1)
        max_w = max(max_w, 1)

    n = 10 * max_h * max_w
    w_bytes = n * n * 4
    if w_bytes > MAX_MODEL_BYTES:
        return None, f'W_too_large_{w_bytes}', (max_h, max_w)

    # Build training data using bounded one-hot
    X_list, Y_list = [], []
    for ex in td['train']:
        x = grid_to_onehot(ex['input'], max_h, max_w).reshape(1, -1)
        y = grid_to_onehot(ex['output'], max_h, max_w).reshape(1, -1)
        X_list.append(x)
        Y_list.append(y)
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    # Solve for W using least-squares
    W, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)

    # Verify on all examples
    for ex in td['train'] + td['test']:
        x = grid_to_onehot(ex['input'], max_h, max_w).reshape(1, -1)
        y_pred = (x @ W).reshape(10, max_h, max_w)
        pred = np.argmax(y_pred, axis=0)
        exp = ex['output']
        h, w = len(exp), len(exp[0])
        for r in range(h):
            for c in range(w):
                if pred[r][c] != exp[r][c]:
                    return None, 'verify_fail', (max_h, max_w)

    return W, 'ok', (max_h, max_w)


def build_model_from_W(W, h, w, opset=11):
    """Build ONNX model: Slice content -> Reshape -> Gemm -> Reshape -> Pad."""
    n = 10 * h * w
    nodes = []
    inits = []

    # Slice input to content bounds: [0, 0, 0, 0] -> [1, 10, h, w]
    inits.append(numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), name='starts'))
    inits.append(numpy_helper.from_array(np.array([1, 10, h, w], dtype=np.int64), name='ends'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='axes'))
    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'axes'], ['sliced']))

    # Reshape to flat: (1, 10, h, w) -> (1, n)
    inits.append(numpy_helper.from_array(np.array([1, n], dtype=np.int64), name='shape_in'))
    nodes.append(helper.make_node('Reshape', ['sliced', 'shape_in'], ['flat_in']))

    # Gemm: flat_out = flat_in @ W + bias
    inits.append(numpy_helper.from_array(W.astype(np.float32), name='W'))
    inits.append(numpy_helper.from_array(np.zeros(n, dtype=np.float32), name='bias'))
    nodes.append(helper.make_node('Gemm', ['flat_in', 'W', 'bias'], ['flat_out']))

    # Reshape back: (1, n) -> (1, 10, h, w)
    inits.append(numpy_helper.from_array(np.array([1, 10, h, w], dtype=np.int64), name='shape_mid'))
    nodes.append(helper.make_node('Reshape', ['flat_out', 'shape_mid'], ['mid']))

    # Pad to (1, 10, 30, 30)
    ph = 30 - h
    pw = 30 - w
    inits.append(numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, ph, pw], dtype=np.int64), name='pad'))
    inits.append(numpy_helper.from_array(np.float32(0), name='pad_val'))
    nodes.append(helper.make_node('Pad', ['mid', 'pad', 'pad_val'], ['output'], mode='constant'))

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)

    graph = helper.make_graph(nodes, f'pseudo_inverse_{h}x{w}', [X], [Y], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def try_task(task_id):
    """Try pseudo-inverse on a task with auto-detected bounds."""
    W, status, (h, w) = try_pseudo_inverse_bounded(task_id)
    if status != 'ok':
        return None, status

    model = build_model_from_W(W, h, w)
    return model, 'ok'


def verify_model(task_id, model):
    """Verify model on all examples using onnxruntime."""
    import onnxruntime as ort
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(model, f.name)
        path = f.name

    try:
        sess = ort.InferenceSession(path)
        td = json.load(open(f'../neurogolf-2026/{task_id}.json'))

        for ex in td['train'] + td['test']:
            arr = np.zeros(SHAPE, dtype=np.float32)
            inp = ex['input']
            for r in range(min(30, len(inp))):
                for c in range(min(30, len(inp[0]))):
                    ch = inp[r][c]
                    if 0 <= ch < 10:
                        arr[0, ch, r, c] = 1.0
            out = sess.run(None, {'input': arr})[0]
            pred = np.argmax(out[0], axis=0).tolist()
            exp = ex['output']
            h, w = len(exp), len(exp[0])
            for r in range(h):
                for c in range(w):
                    if pred[r][c] != exp[r][c]:
                        return False
        return True
    except Exception as e:
        return False
    finally:
        os.unlink(path)


if __name__ == '__main__':
    tasks_dir = '../neurogolf-2026'
    task_files = sorted([f.replace('.json', '') for f in os.listdir(tasks_dir) if f.endswith('.json')])

    os.makedirs('models', exist_ok=True)

    solved = []
    skipped_reasons = {}

    for task_id in task_files:
        try:
            model, status = try_task(task_id)
            if status == 'ok':
                # Verify with onnxruntime
                if verify_model(task_id, model):
                    path = f'models/{task_id}.onnx'
                    onnx.save(model, path)
                    size = os.path.getsize(path)
                    solved.append(task_id)
                    print(f'{task_id}: OK ({size} bytes)')
                else:
                    print(f'{task_id}: runtime_verify_fail')
            else:
                key = status.split('_')[0] if '_' in status else status
                skipped_reasons[key] = skipped_reasons.get(key, 0) + 1
        except Exception as e:
            skipped_reasons[f'error_{str(e)[:30]}'] = skipped_reasons.get(f'error_{str(e)[:30]}', 0) + 1

    print(f'\nSolved: {len(solved)} tasks')
    if solved:
        for t in solved:
            print(f'  {t}')
    print(f'\nSkipped reasons:')
    for reason, count in sorted(skipped_reasons.items(), key=lambda x: -x[1]):
        print(f'  {reason}: {count}')
