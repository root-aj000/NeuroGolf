#!/usr/bin/env python3
"""
Build hmirror_vconcat and vmirror_hconcat ONNX models.
Strategy: reverse full 30 rows/cols, extract relevant slices, concat, pad.
"""
import numpy as np
import onnxruntime as ort
import onnx
import os
from onnx import helper, TensorProto, numpy_helper
import json

SHAPE = [1, 10, 30, 30]
REV30 = np.arange(29, -1, -1, dtype=np.int64)


def build_hmirror_vconcat():
    """
    hmirror(I) then vconcat(x1, I):
    - Flip input vertically (reverse all 30 rows)
    - Content at original rows 0..H-1 → reversed rows (30-H)..29
    - Slice reversed rows (30-H)..30 → flipped content [1,10,H,30]
    - Slice original rows 0..H → original content [1,10,H,30]
    - Concat flipped + original along axis=2 → [1,10,2H,30]
    - Pad to [1,10,30,30]
    """
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)

    all_nodes = []
    all_inits = []

    # 1. Detect content height
    all_nodes.extend([
        helper.make_node('ReduceMax', ['input'], ['rh1'], axes=[1], keepdims=0),
        helper.make_node('ReduceMax', ['rh1'], ['row_mask'], axes=[2], keepdims=0),
        helper.make_node('Gather', ['row_mask', 'rev'], ['rev_mask'], axis=1),
        helper.make_node('ArgMax', ['rev_mask'], ['rev_idx'], axis=1, keepdims=0),
        helper.make_node('Sub', ['thirty', 'rev_idx'], ['content_h']),
    ])

    # 2. Reverse all 30 rows
    all_nodes.append(helper.make_node('Gather', ['input', 'rev'], ['reversed'], axis=2))

    # 3. Slice flipped content: reversed rows (30-content_h)..30
    all_nodes.extend([
        helper.make_node('Sub', ['thirty', 'content_h'], ['flip_start']),
        helper.make_node('Concat', ['_z', '_z', 'flip_start', '_z'], ['flip_starts'], axis=0),
        helper.make_node('Concat', ['_1', '_10', 'thirty', '_30'], ['flip_ends'], axis=0),
        helper.make_node('Slice', ['reversed', 'flip_starts', 'flip_ends', '_ax'], ['flipped']),
    ])

    # 4. Slice original content: rows 0..content_h
    all_nodes.extend([
        helper.make_node('Concat', ['_1', '_10', 'content_h', '_30'], ['orig_ends'], axis=0),
        helper.make_node('Slice', ['input', '_z4', 'orig_ends', '_ax'], ['original']),
    ])

    # 5. Concat flipped + original along height
    all_nodes.append(helper.make_node('Concat', ['flipped', 'original'], ['combined'], axis=2))

    # 6. Pad to 30x30: pad_bottom = 30 - 2*content_h
    all_nodes.extend([
        helper.make_node('Add', ['content_h', 'content_h'], ['double_h']),
        helper.make_node('Sub', ['thirty', 'double_h'], ['pad_bottom']),
        helper.make_node('Concat', ['_z4', '_z', '_z', 'pad_bottom', '_z'], ['pad_spec'], axis=0),
        helper.make_node('Pad', ['combined', 'pad_spec', '_zero'], ['output'], mode='constant'),
    ])

    all_inits.extend([
        numpy_helper.from_array(REV30, name='rev'),
        numpy_helper.from_array(np.array([30], dtype=np.int64), name='thirty'),
        numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='_z'),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name='_1'),
        numpy_helper.from_array(np.array([10], dtype=np.int64), name='_10'),
        numpy_helper.from_array(np.array([30], dtype=np.int64), name='_30'),
        numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='_z4'),
        numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='_ax'),
        numpy_helper.from_array(np.float32(0), name='_zero'),
    ])

    graph = helper.make_graph(all_nodes, 'hmirror_vconcat', [X], [Y], all_inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_vconcat_hmirror():
    """
    vconcat(I, hmirror(I)): original on top, vertically-flipped on bottom.
    Same as hmirror_vconcat but concat order is original + flipped.
    """
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)

    all_nodes = []
    all_inits = []

    # 1. Detect content height
    all_nodes.extend([
        helper.make_node('ReduceMax', ['input'], ['rh1'], axes=[1], keepdims=0),
        helper.make_node('ReduceMax', ['rh1'], ['row_mask'], axes=[2], keepdims=0),
        helper.make_node('Gather', ['row_mask', 'rev'], ['rev_mask'], axis=1),
        helper.make_node('ArgMax', ['rev_mask'], ['rev_idx'], axis=1, keepdims=0),
        helper.make_node('Sub', ['thirty', 'rev_idx'], ['content_h']),
    ])

    # 2. Reverse all 30 rows
    all_nodes.append(helper.make_node('Gather', ['input', 'rev'], ['reversed'], axis=2))

    # 3. Slice flipped content: reversed rows (30-content_h)..30
    all_nodes.extend([
        helper.make_node('Sub', ['thirty', 'content_h'], ['flip_start']),
        helper.make_node('Concat', ['_z', '_z', 'flip_start', '_z'], ['flip_starts'], axis=0),
        helper.make_node('Concat', ['_1', '_10', 'thirty', '_30'], ['flip_ends'], axis=0),
        helper.make_node('Slice', ['reversed', 'flip_starts', 'flip_ends', '_ax'], ['flipped']),
    ])

    # 4. Slice original content: rows 0..content_h
    all_nodes.extend([
        helper.make_node('Concat', ['_1', '_10', 'content_h', '_30'], ['orig_ends'], axis=0),
        helper.make_node('Slice', ['input', '_z4', 'orig_ends', '_ax'], ['original']),
    ])

    # 5. Concat original + flipped along height (original on top)
    all_nodes.append(helper.make_node('Concat', ['original', 'flipped'], ['combined'], axis=2))

    # 6. Pad to 30x30: pad_bottom = 30 - 2*content_h
    all_nodes.extend([
        helper.make_node('Add', ['content_h', 'content_h'], ['double_h']),
        helper.make_node('Sub', ['thirty', 'double_h'], ['pad_bottom']),
        helper.make_node('Concat', ['_z4', '_z', '_z', 'pad_bottom', '_z'], ['pad_spec'], axis=0),
        helper.make_node('Pad', ['combined', 'pad_spec', '_zero'], ['output'], mode='constant'),
    ])

    all_inits.extend([
        numpy_helper.from_array(REV30, name='rev'),
        numpy_helper.from_array(np.array([30], dtype=np.int64), name='thirty'),
        numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='_z'),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name='_1'),
        numpy_helper.from_array(np.array([10], dtype=np.int64), name='_10'),
        numpy_helper.from_array(np.array([30], dtype=np.int64), name='_30'),
        numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='_z4'),
        numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='_ax'),
        numpy_helper.from_array(np.float32(0), name='_zero'),
    ])

    graph = helper.make_graph(all_nodes, 'vconcat_hmirror', [X], [Y], all_inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_vmirror_hconcat():
    """
    vmirror(I) then hconcat(I, x1):
    - vmirror = flip horizontally (reverse all 30 cols)
    - Content at original cols 0..W-1 → reversed cols (30-W)..29
    - Slice reversed cols (30-W)..30 → flipped content [1,10,30,W]
    - Slice original cols 0..W → original content [1,10,30,W]
    - Concat original + flipped along axis=3 → [1,10,30,2W]
    - Pad to [1,10,30,30]
    """
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)

    all_nodes = []
    all_inits = []

    # 1. Detect content width
    all_nodes.extend([
        helper.make_node('ReduceMax', ['input'], ['cw1'], axes=[1], keepdims=0),
        helper.make_node('ReduceMax', ['cw1'], ['col_mask'], axes=[1], keepdims=0),
        helper.make_node('Gather', ['col_mask', 'rev'], ['rev_cmask'], axis=1),
        helper.make_node('ArgMax', ['rev_cmask'], ['cw_idx'], axis=1, keepdims=0),
        helper.make_node('Sub', ['thirty', 'cw_idx'], ['content_w']),
    ])

    # 2. Reverse all 30 cols
    all_nodes.append(helper.make_node('Gather', ['input', 'rev'], ['reversed'], axis=3))

    # 3. Slice flipped content: reversed cols (30-content_w)..30
    all_nodes.extend([
        helper.make_node('Sub', ['thirty', 'content_w'], ['flip_start']),
        helper.make_node('Concat', ['_z', '_z', '_z', 'flip_start'], ['flip_starts'], axis=0),
        helper.make_node('Concat', ['_1', '_10', '_30', 'thirty'], ['flip_ends'], axis=0),
        helper.make_node('Slice', ['reversed', 'flip_starts', 'flip_ends', '_ax'], ['flipped']),
    ])

    # 4. Slice original content: cols 0..content_w
    all_nodes.extend([
        helper.make_node('Concat', ['_1', '_10', '_30', 'content_w'], ['orig_ends'], axis=0),
        helper.make_node('Slice', ['input', '_z4', 'orig_ends', '_ax'], ['original']),
    ])

    # 5. Concat original + flipped along width
    all_nodes.append(helper.make_node('Concat', ['original', 'flipped'], ['combined'], axis=3))

    # 6. Pad to 30x30: pad_right = 30 - 2*content_w
    all_nodes.extend([
        helper.make_node('Add', ['content_w', 'content_w'], ['double_w']),
        helper.make_node('Sub', ['thirty', 'double_w'], ['pad_right']),
        helper.make_node('Concat', ['_z4', '_z', '_z', '_z', 'pad_right'], ['pad_spec'], axis=0),
        helper.make_node('Pad', ['combined', 'pad_spec', '_zero'], ['output'], mode='constant'),
    ])

    all_inits.extend([
        numpy_helper.from_array(REV30, name='rev'),
        numpy_helper.from_array(np.array([30], dtype=np.int64), name='thirty'),
        numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='_z'),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name='_1'),
        numpy_helper.from_array(np.array([10], dtype=np.int64), name='_10'),
        numpy_helper.from_array(np.array([30], dtype=np.int64), name='_30'),
        numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='_z4'),
        numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='_ax'),
        numpy_helper.from_array(np.float32(0), name='_zero'),
    ])

    graph = helper.make_graph(all_nodes, 'vmirror_hconcat', [X], [Y], all_inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def verify_model(model, task_data):
    """Verify model on all training examples."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(model, f.name)
        path = f.name
    try:
        sess = ort.InferenceSession(path)
        for ex in task_data['train']:
            inp = ex['input']
            exp = ex['output']
            h_in, w_in = len(inp), len(inp[0])
            h_out, w_out = len(exp), len(exp[0])

            arr = np.zeros(SHAPE, dtype=np.float32)
            for r in range(min(30, h_in)):
                for c in range(min(30, w_in)):
                    color = inp[r][c]
                    if 0 <= color < 10:
                        arr[0, color, r, c] = 1.0

            out = sess.run(None, {'input': arr})[0]
            res = np.argmax(out[0], axis=0).tolist()

            for r in range(min(h_out, 30)):
                for c in range(min(w_out, 30)):
                    if res[r][c] != exp[r][c]:
                        return False, path
        return True, path
    except Exception as e:
        return False, path


# Load tasks
tasks = {}
for i in range(1, 401):
    with open(f"../neurogolf-2026/task{i:03d}.json") as f:
        tasks[i] = json.load(f)

# Test on concat tasks
concat_tasks = {
    'task116': 'hmirror_vconcat',
    'task172': 'vconcat_hmirror',
    'task210': 'vconcat_hmirror',
    'task164': 'vmirror_hconcat',
    'task311': 'vmirror_hconcat',
}

builders = {
    'hmirror_vconcat': build_hmirror_vconcat,
    'vconcat_hmirror': build_vconcat_hmirror,
    'vmirror_hconcat': build_vmirror_hconcat,
}

for task_key, method in concat_tasks.items():
    task_num = int(task_key.replace('task', ''))
    model = builders[method]()
    ok, path = verify_model(model, tasks[task_num])
    print(f"{task_key}: {'OK' if ok else 'FAIL'} ({method})")
    if ok:
        os.makedirs('models', exist_ok=True)
        onnx.save(model, f'models/{task_key}.onnx')
    if path and os.path.exists(path):
        os.unlink(path)
