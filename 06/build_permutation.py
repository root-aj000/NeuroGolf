#!/usr/bin/env python3
"""
Strategy 6C: Permutation/Projection Matrices
Solves tasks where output pixels are a rearrangement of input pixels.
Each output pixel copies from exactly one input pixel (possibly via a transform).
Uses a sparse permutation matrix embedded as a dense ONNX constant.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import json
import os
import sys

SHAPE = [1, 10, 30, 30]


def grid_to_onehot(grid, h=30, w=30):
    oh = np.zeros((10, h, w), dtype=np.float32)
    for r in range(min(len(grid), h)):
        for c in range(min(len(grid[0]), w)):
            v = grid[r][c]
            if 0 <= v < 10:
                oh[v, r, c] = 1.0
    return oh


def detect_permutation(task_id):
    """
    Detect if output is a pixel-wise rearrangement of input.
    For each output position, find which input position it copies from.
    Returns a mapping: (out_r, out_c) -> (in_r, in_c) if consistent across all examples.
    """
    td = json.load(open(f'../neurogolf-2026/{task_id}.json'))

    # All examples must have same input/output size
    sizes = set()
    for ex in td['train']:
        sizes.add((len(ex['input']), len(ex['input'][0]), len(ex['output']), len(ex['output'][0])))
    if len(sizes) != 1:
        return None, 'varying_sizes'

    ih, iw, oh, ow = next(iter(sizes))

    # For each output position, find the input position it copies from
    # Use first training example to establish mapping
    ex0 = td['train'][0]
    mapping = {}  # (or, oc) -> (ir, ic)

    for or_ in range(oh):
        for oc in range(ow):
            out_val = ex0['output'][or_][oc]
            # Find which input position has this value
            candidates = []
            for ir in range(ih):
                for ic in range(iw):
                    if ex0['input'][ir][ic] == out_val:
                        candidates.append((ir, ic))
            if len(candidates) == 0 and out_val == 0:
                # Output is background, input might have no corresponding pixel
                # Mark as "zero" (no input source)
                mapping[(or_, oc)] = None
            elif len(candidates) == 1:
                mapping[(or_, oc)] = candidates[0]
            else:
                # Multiple candidates - need to find which one is consistent
                # Try each candidate and check consistency
                found = False
                for cand in candidates:
                    consistent = True
                    for ex in td['train']:
                        ir, ic = cand
                        if ex['input'][ir][ic] != ex['output'][or_][oc]:
                            consistent = False
                            break
                    if consistent:
                        mapping[(or_, oc)] = cand
                        found = True
                        break
                if not found:
                    return None, 'ambiguous_mapping'

    # Verify mapping on all training examples
    for ex in td['train']:
        for or_ in range(oh):
            for oc in range(ow):
                src = mapping.get((or_, oc))
                if src is None:
                    if ex['output'][or_][oc] != 0:
                        return None, 'mapping_verify_fail'
                else:
                    ir, ic = src
                    if ex['input'][ir][ic] != ex['output'][or_][oc]:
                        return None, 'mapping_verify_fail'

    # Verify on test
    for ex in td['test']:
        for or_ in range(oh):
            for oc in range(ow):
                src = mapping.get((or_, oc))
                if src is None:
                    if ex['output'][or_][oc] != 0:
                        return None, 'test_fail'
                else:
                    ir, ic = src
                    if ex['input'][ir][ic] != ex['output'][or_][oc]:
                        return None, 'test_fail'

    return {'mapping': mapping, 'ih': ih, 'iw': iw, 'oh': oh, 'ow': ow}, 'ok'


def build_permutation_model(info, opset=11):
    """
    Build ONNX model using Gather to permute pixels.
    For each channel, gather input pixels in the order specified by the permutation.
    Uses (900,) index array (3.6KB) instead of (900,900) matrix (3.2MB).
    """
    mapping = info['mapping']
    ih, iw, oh, ow = info['ih'], info['iw'], info['oh'], info['ow']

    nodes = []
    inits = []

    # Build permutation index array: for each output position (r*30+c),
    # which input position does it copy from?
    perm = np.zeros(900, dtype=np.int64)
    for (or_, oc), src in mapping.items():
        out_idx = or_ * 30 + oc
        if src is not None:
            ir, ic = src
            in_idx = ir * 30 + ic
            perm[out_idx] = in_idx
        else:
            perm[out_idx] = 0  # will be zero since input at 0,0 is typically background

    inits.append(numpy_helper.from_array(perm, name='perm'))

    # Reshape input: (1, 10, 30, 30) -> (10, 900)
    inits.append(numpy_helper.from_array(np.array([10, 900], dtype=np.int64), name='shape_flat'))
    nodes.append(helper.make_node('Reshape', ['input', 'shape_flat'], ['flat_in']))

    # Gather: for each channel, permute the 900 positions
    # flat_in shape: (10, 900), perm shape: (900,)
    # Output shape: (10, 900)
    nodes.append(helper.make_node('Gather', ['flat_in', 'perm'], ['flat_out'], axis=1))

    # Reshape back to (1, 10, 30, 30)
    inits.append(numpy_helper.from_array(np.array(SHAPE, dtype=np.int64), name='shape_out'))
    nodes.append(helper.make_node('Reshape', ['flat_out', 'shape_out'], ['output']))

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)

    graph = helper.make_graph(nodes, 'permutation_gather', [X], [Y], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def try_permutation(task_id):
    """Try permutation approach."""
    result, status = detect_permutation(task_id)
    if status != 'ok':
        return None, status

    model = build_permutation_model(result)
    return model, 'ok'


def verify_model(task_id, model):
    """Verify model on all examples."""
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

    for task_id in task_files:
        try:
            model, status = try_permutation(task_id)
            if status == 'ok':
                if verify_model(task_id, model):
                    path = f'models/{task_id}.onnx'
                    onnx.save(model, path)
                    size = os.path.getsize(path)
                    solved.append(task_id)
                    print(f'{task_id}: OK ({size} bytes)')
        except Exception as e:
            pass

    print(f'\nSolved: {len(solved)} tasks')
    for t in solved:
        print(f'  {t}')
