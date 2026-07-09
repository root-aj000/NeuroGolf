#!/usr/bin/env python3
"""
Strategy 6B: Kronecker Product Tiling
Solves tasks where output = layout ⊗ stamp (tiling a small pattern across a grid).
Also covers tasks with channel-wise color remapping via Hadamard masks.
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


def try_color_remap(task_id):
    """
    Try pure color remapping: each pixel's color changes independently of position.
    output_color = f(input_color) for some permutation/mapping.
    This is a 10x10 color mapping matrix applied per-pixel.
    """
    td = json.load(open(f'../neurogolf-2026/{task_id}.json'))

    # Check all examples have same grid size
    sizes = set()
    for ex in td['train'] + td['test']:
        sizes.add((len(ex['input']), len(ex['input'][0])))
    if len(sizes) != 1:
        return None, 'varying_sizes'

    h, w = next(iter(sizes))

    # For each training example, extract color mapping
    # For each position (r,c), input color -> output color
    # If the mapping is consistent across all positions, it's a global color map
    color_maps = []  # list of (input_color, output_color) pairs per example
    for ex in td['train']:
        cm = {}
        for r in range(h):
            for c in range(w):
                ic = ex['input'][r][c]
                oc = ex['output'][r][c]
                if ic != 0:  # skip background
                    if ic in cm:
                        if cm[ic] != oc:
                            return None, 'inconsistent_color_map'
                    cm[ic] = oc
        color_maps.append(cm)

    # Check all training examples agree on color mapping
    unified_map = {}
    for cm in color_maps:
        for k, v in cm.items():
            if k in unified_map:
                if unified_map[k] != v:
                    return None, 'conflicting_color_map'
            unified_map[k] = v

    # Verify on test examples
    for ex in td['test']:
        for r in range(h):
            for c in range(w):
                ic = ex['input'][r][c]
                oc = ex['output'][r][c]
                expected = unified_map.get(ic, ic)  # unmapped colors stay same
                if oc != expected:
                    return None, 'test_fail'

    return unified_map, 'ok'


def build_color_remap_model(color_map, h=30, w=30, opset=11):
    """
    Build ONNX model for color remapping using channel-slicing.
    For each input color, route its one-hot channel to the output color channel.
    """
    nodes = []
    inits = []

    # For each output channel, find which input channel maps to it
    # Build inverse map: for output color c, which input color gives c?
    inv_map = {}
    for ic, oc in color_map.items():
        inv_map[oc] = ic

    # Build channel routing
    for out_c in range(10):
        if out_c in inv_map:
            in_c = inv_map[out_c]
            # Slice input channel in_c, copy to output channel out_c
            inits.append(numpy_helper.from_array(np.array([0, in_c, 0, 0], dtype=np.int64), name=f's_in_{out_c}'))
            inits.append(numpy_helper.from_array(np.array([1, in_c + 1, 30, 30], dtype=np.int64), name=f'e_in_{out_c}'))
            inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name=f'ax_{out_c}'))
            nodes.append(helper.make_node('Slice', ['input', f's_in_{out_c}', f'e_in_{out_c}', f'ax_{out_c}'], [f'ch_{out_c}']))
        else:
            # No mapping to this output color - zero it out
            inits.append(numpy_helper.from_array(np.zeros((1, 1, 30, 30), dtype=np.float32), name=f'zeros_{out_c}'))
            nodes.append(helper.make_node('Identity', [f'zeros_{out_c}'], [f'ch_{out_c}']))

    # Concat all channels
    ch_names = [f'ch_{c}' for c in range(10)]
    nodes.append(helper.make_node('Concat', ch_names, ['output'], axis=1))

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)

    graph = helper.make_graph(nodes, 'color_remap', [X], [Y], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def try_tiling(task_id):
    """
    Try Kronecker tiling: output[r*s+i, c*s+j] = input[r, c] for each (i,j) in stamp.
    For 10x10 -> 10x10, no tiling needed (identity or simple remap).
    For 3x3 -> 30x30, each cell tiles to 10x10.
    """
    td = json.load(open(f'../neurogolf-2026/{task_id}.json'))

    # Check if output is a regular tiling of input
    for ex in td['train']:
        inp = ex['input']
        out = ex['output']
        ih, iw = len(inp), len(inp[0])
        oh, ow = len(out), len(out[0])

        if oh % ih != 0 or ow % iw != 0:
            return None, 'not_tilable'

        sy = oh // ih
        sx = ow // iw

        # Check if each input cell tiles to a sx x sy block of same color
        for r in range(ih):
            for c in range(iw):
                color = inp[r][c]
                for dr in range(sy):
                    for dc in range(sx):
                        or_ = r * sy + dr
                        oc = c * sx + dc
                        if out[or_][oc] != color:
                            return None, 'not_regular_tiling'

    return {'sy': sy, 'sx': sx, 'ih': ih, 'iw': iw}, 'ok'


def build_tiling_model(info, opset=11):
    """Build ONNX model for regular tiling using Reshape + Expand."""
    ih, iw = info['ih'], info['iw']
    sy, sx = info['sy'], info['sx']
    oh, ow = ih * sy, iw * sx

    nodes = []
    inits = []

    # Slice content: (1, 10, ih, iw)
    inits.append(numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), name='starts'))
    inits.append(numpy_helper.from_array(np.array([1, 10, ih, iw], dtype=np.int64), name='ends'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='axes'))
    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'axes'], ['sliced']))

    # Reshape to (1, 10, ih, 1, iw, 1) for broadcasting
    inits.append(numpy_helper.from_array(np.array([1, 10, ih, 1, iw, 1], dtype=np.int64), name='shape_expand'))
    nodes.append(helper.make_node('Reshape', ['sliced', 'shape_expand'], ['expanded']))

    # Expand to (1, 10, ih, sy, iw, sx)
    inits.append(numpy_helper.from_array(np.array([1, 1, 1, sy, 1, sx], dtype=np.int64), name='shape_tiled'))
    nodes.append(helper.make_node('Expand', ['expanded', 'shape_tiled'], ['tiled']))

    # Reshape to (1, 10, oh, ow)
    inits.append(numpy_helper.from_array(np.array([1, 10, oh, ow], dtype=np.int64), name='shape_out'))
    nodes.append(helper.make_node('Reshape', ['tiled', 'shape_out'], ['tiled_2d']))

    # Pad to (1, 10, 30, 30) if needed
    if oh < 30 or ow < 30:
        ph = 30 - oh
        pw = 30 - ow
        inits.append(numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, ph, pw], dtype=np.int64), name='pad'))
        inits.append(numpy_helper.from_array(np.float32(0), name='pad_val'))
        nodes.append(helper.make_node('Pad', ['tiled_2d', 'pad', 'pad_val'], ['output'], mode='constant'))
    else:
        nodes.append(helper.make_node('Identity', ['tiled_2d'], ['output']))

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)

    graph = helper.make_graph(nodes, 'tiling', [X], [Y], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


if __name__ == '__main__':
    tasks_dir = '../neurogolf-2026'
    task_files = sorted([f.replace('.json', '') for f in os.listdir(tasks_dir) if f.endswith('.json')])

    os.makedirs('models', exist_ok=True)

    solved = []

    # Strategy 1: Color remapping
    print('=== Color Remapping ===')
    for task_id in task_files:
        try:
            result, status = try_color_remap(task_id)
            if status == 'ok':
                model = build_color_remap_model(result)

                # Verify
                import onnxruntime as ort
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                    onnx.save(model, f.name)
                    path = f.name
                try:
                    sess = ort.InferenceSession(path)
                    td = json.load(open(f'../neurogolf-2026/{task_id}.json'))
                    all_ok = True
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
                                    all_ok = False
                                    break
                            if not all_ok:
                                break
                        if not all_ok:
                            break

                    if all_ok:
                        path = f'models/{task_id}.onnx'
                        onnx.save(model, path)
                        size = os.path.getsize(path)
                        solved.append(task_id)
                        print(f'{task_id}: OK ({size} bytes) color_map={result}')
                    else:
                        pass  # print(f'{task_id}: verify_fail')
                finally:
                    os.unlink(path)
        except Exception as e:
            pass

    # Strategy 2: Tiling
    print('\n=== Tiling ===')
    for task_id in task_files:
        if task_id in solved:
            continue
        try:
            result, status = try_tiling(task_id)
            if status == 'ok':
                model = build_tiling_model(result)

                import onnxruntime as ort
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                    onnx.save(model, f.name)
                    path = f.name
                try:
                    sess = ort.InferenceSession(path)
                    td = json.load(open(f'../neurogolf-2026/{task_id}.json'))
                    all_ok = True
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
                                    all_ok = False
                                    break
                            if not all_ok:
                                break
                        if not all_ok:
                            break

                    if all_ok:
                        path = f'models/{task_id}.onnx'
                        onnx.save(model, path)
                        size = os.path.getsize(path)
                        solved.append(task_id)
                        print(f'{task_id}: OK ({size} bytes) tiling={result}')
                finally:
                    os.unlink(path)
        except Exception as e:
            pass

    print(f'\nTotal solved: {len(solved)}')
    for t in solved:
        print(f'  {t}')
