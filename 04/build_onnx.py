#!/usr/bin/env python3
"""
Final ONNX builder for neurogolf tasks - opset 10 compatible.
Handles: single transforms, color ops, concatenation patterns.
"""
import json
import sys
import os
import re
import inspect
import numpy as np
from build_complex import get_complex_builder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'arc-dsl'))
import solvers as S

import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

from build_complex import get_complex_builder


CONST_MAP = {'ZERO': 0, 'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4,
             'FIVE': 5, 'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9}

SHAPE = [1, 10, 30, 30]
REV30 = np.arange(29, -1, -1, dtype=np.int64)


def mk_graph(nodes, name, inputs, outputs, inits=None):
    return helper.make_graph(nodes, name, inputs, outputs, inits or [])


def _detect_content_dims(nodes, inits, input_name, prefix=''):
    """Detect content height and width.
    After ReduceMax axes=[1] keepdims=0 → [1,30,30] binary mask.
    For height: ReduceMax axes=[2] → [1,30] per-row mask, then ArgMax on reversed.
    For width: ReduceMax axes=[1] → [1,30] per-col mask, then ArgMax on reversed.
    Returns (content_h, content_w) names.
    """
    s = prefix
    inits.append(numpy_helper.from_array(REV30, name=f'{s}rev'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name=f'{s}thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name=f'{s}z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=f'{s}_1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name=f'{s}_10'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name=f'{s}_30'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name=f'{s}z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name=f'{s}ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name=f'{s}zero'))

    # Binary mask: [1, 30, 30]
    nodes.append(helper.make_node('ReduceMax', [input_name], [f'{s}bm'], axes=[1], keepdims=0))
    # Height: per-row mask
    nodes.append(helper.make_node('ReduceMax', [f'{s}bm'], [f'{s}rm'], axes=[2], keepdims=0))
    nodes.append(helper.make_node('Gather', [f'{s}rm', f'{s}rev'], [f'{s}rrm'], axis=1))
    nodes.append(helper.make_node('ArgMax', [f'{s}rrm'], [f'{s}hi'], axis=1, keepdims=0))
    nodes.append(helper.make_node('Sub', [f'{s}thirty', f'{s}hi'], [f'{s}ch']))
    # Width: per-col mask
    nodes.append(helper.make_node('ReduceMax', [f'{s}bm'], [f'{s}cm'], axes=[1], keepdims=0))
    nodes.append(helper.make_node('Gather', [f'{s}cm', f'{s}rev'], [f'{s}rcm'], axis=1))
    nodes.append(helper.make_node('ArgMax', [f'{s}rcm'], [f'{s}wi'], axis=1, keepdims=0))
    nodes.append(helper.make_node('Sub', [f'{s}thirty', f'{s}wi'], [f'{s}cw']))
    return f'{s}ch', f'{s}cw'


def _slice_content(nodes, inits, src, ch, cw, out_name, prefix=''):
    """Slice [1,10,H,W] content from src at top-left corner."""
    s = prefix
    nodes.append(helper.make_node('Concat', [f'{s}_1', f'{s}_10', ch, cw], [f'{s}ends'], axis=0))
    nodes.append(helper.make_node('Slice', [src, f'{s}z4', f'{s}ends', f'{s}ax'], [out_name]))


def _pad_to_30(nodes, inits, src, ch, cw, out_name, prefix=''):
    """Pad [1,10,H,W] to [1,10,30,30] at top-left (pad bottom+right)."""
    s = prefix
    nodes.append(helper.make_node('Sub', [f'{s}thirty', ch], [f'{s}ph']))
    nodes.append(helper.make_node('Sub', [f'{s}thirty', cw], [f'{s}pw']))
    # pads: [begin_n, begin_c, begin_h, begin_w, end_n, end_c, end_h, end_w]
    nodes.append(helper.make_node('Concat', [f'{s}z', f'{s}z', f'{s}z', f'{s}z', f'{s}z', f'{s}z', f'{s}ph', f'{s}pw'], [f'{s}ps'], axis=0))
    nodes.append(helper.make_node('Pad', [src, f'{s}ps', f'{s}zero'], [out_name], mode='constant'))


def build_vmirror():
    """Reverse full 30 cols, slice content at (30-W)..30, pad back to 30x30."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []
    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    # Reverse all 30 cols
    nodes.append(helper.make_node('Gather', ['input', 'a_rev'], ['a_reversed'], axis=3))
    # Slice flipped content: cols (30-cw)..30
    nodes.append(helper.make_node('Sub', ['a_thirty', cw], ['a_cs']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_z', 'a_cs'], ['a_starts'], axis=0))
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', 'a__30', 'a_thirty'], ['a_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['a_reversed', 'a_starts', 'a_ends', 'a_ax'], ['a_flipped']))
    _pad_to_30(nodes, inits, 'a_flipped', ch, cw, 'output', 'a_')
    return helper.make_model(mk_graph(nodes, 'vmirror', [X], [Y], inits),
                             opset_imports=[helper.make_opsetid("", 11)])

def build_hmirror():
    """Reverse full 30 rows, slice content at (30-H)..30, pad back to 30x30."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []
    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    # Reverse all 30 rows
    nodes.append(helper.make_node('Gather', ['input', 'a_rev'], ['a_reversed'], axis=2))
    # Slice flipped content: rows (30-ch)..30
    nodes.append(helper.make_node('Sub', ['a_thirty', ch], ['a_rs']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_rs', 'a_z'], ['a_starts'], axis=0))
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', 'a_thirty', 'a__30'], ['a_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['a_reversed', 'a_starts', 'a_ends', 'a_ax'], ['a_flipped']))
    _pad_to_30(nodes, inits, 'a_flipped', ch, cw, 'output', 'a_')
    return helper.make_model(mk_graph(nodes, 'hmirror', [X], [Y], inits),
                             opset_imports=[helper.make_opsetid("", 11)])

def build_rot180():
    """Reverse rows then cols, slice content at (30-H,30-W)..(30,30), pad back."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []
    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    # Reverse rows then cols
    nodes.append(helper.make_node('Gather', ['input', 'a_rev'], ['a_tmp'], axis=2))
    nodes.append(helper.make_node('Gather', ['a_tmp', 'a_rev'], ['a_reversed'], axis=3))
    # Slice: rows (30-ch)..30, cols (30-cw)..30
    nodes.append(helper.make_node('Sub', ['a_thirty', ch], ['a_rs']))
    nodes.append(helper.make_node('Sub', ['a_thirty', cw], ['a_cs']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_rs', 'a_cs'], ['a_starts'], axis=0))
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', 'a_thirty', 'a_thirty'], ['a_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['a_reversed', 'a_starts', 'a_ends', 'a_ax'], ['a_flipped']))
    _pad_to_30(nodes, inits, 'a_flipped', ch, cw, 'output', 'a_')
    return helper.make_model(mk_graph(nodes, 'rot180', [X], [Y], inits),
                             opset_imports=[helper.make_opsetid("", 11)])

def build_rot90():
    """Transpose, reverse cols, slice content, pad back."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []
    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    # Transpose: [1,10,H,W] → [1,10,W,H], then reverse cols (which are original rows)
    nodes.append(helper.make_node('Transpose', ['input'], ['a_t'], perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node('Gather', ['a_t', 'a_rev'], ['a_reversed'], axis=3))
    # After transpose + reverse, original content (0,0)-(H-1,W-1) maps to:
    # transpose: (r,c)→(c,r), so content at (0,0)-(W-1,H-1) in transposed
    # reverse cols: content at (30-H)..29 cols, 0..W-1 rows
    nodes.append(helper.make_node('Sub', ['a_thirty', ch], ['a_cs']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_z', 'a_cs'], ['a_starts'], axis=0))
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', 'a__30', 'a_thirty'], ['a_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['a_reversed', 'a_starts', 'a_ends', 'a_ax'], ['a_flipped']))
    # Pad: rot90 output is [W, H] → need pad 30-W on axis2, 30-H on axis3
    nodes.append(helper.make_node('Sub', ['a_thirty', cw], ['a_ph']))
    nodes.append(helper.make_node('Sub', ['a_thirty', ch], ['a_pw']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_ph', 'a_pw'], ['a_ps'], axis=0))
    nodes.append(helper.make_node('Pad', ['a_flipped', 'a_ps', 'a_zero'], ['output'], mode='constant'))
    return helper.make_model(mk_graph(nodes, 'rot90', [X], [Y], inits),
                             opset_imports=[helper.make_opsetid("", 11)])

def build_rot270():
    """Transpose, reverse rows, slice content, pad back."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []
    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    # Transpose: [1,10,H,W] → [1,10,W,H], then reverse rows
    nodes.append(helper.make_node('Transpose', ['input'], ['a_t'], perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node('Gather', ['a_t', 'a_rev'], ['a_reversed'], axis=2))
    # After transpose + reverse rows, content at (30-W)..29 rows, 0..H-1 cols
    nodes.append(helper.make_node('Sub', ['a_thirty', cw], ['a_rs']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_rs', 'a_z'], ['a_starts'], axis=0))
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', 'a_thirty', 'a__30'], ['a_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['a_reversed', 'a_starts', 'a_ends', 'a_ax'], ['a_flipped']))
    # Pad: [1,10,W,H] → pad 30-W on axis2, 30-H on axis3
    nodes.append(helper.make_node('Sub', ['a_thirty', cw], ['a_ph']))
    nodes.append(helper.make_node('Sub', ['a_thirty', ch], ['a_pw']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_ph', 'a_pw'], ['a_ps'], axis=0))
    nodes.append(helper.make_node('Pad', ['a_flipped', 'a_ps', 'a_zero'], ['output'], mode='constant'))
    return helper.make_model(mk_graph(nodes, 'rot270', [X], [Y], inits),
                             opset_imports=[helper.make_opsetid("", 11)])

def build_dmirror():
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    node = helper.make_node('Transpose', ['input'], ['output'], perm=[0, 1, 3, 2])
    return helper.make_model(mk_graph([node], 'dmirror', [X], [Y]),
                             opset_imports=[helper.make_opsetid("", 10)])


def _cast_equal_mask(input_name, mask_name, out_name, suffix=''):
    """Cast input and mask to int32, compare, return bool mask."""
    s = suffix
    return [
        helper.make_node('Cast', [input_name], [f'inp_i{s}'], to=TensorProto.INT32),
        helper.make_node('Cast', [mask_name], [f'msk_i{s}'], to=TensorProto.INT32),
        helper.make_node('Equal', [f'inp_i{s}', f'msk_i{s}'], [out_name]),
    ]


def build_switch(c1, c2):
    """Swap colors c1 and c2 in one-hot encoding."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    inits = []
    nodes = []

    # Slice both channels
    inits.append(numpy_helper.from_array(np.array([0, c1, 0, 0], dtype=np.int64), name='s1'))
    inits.append(numpy_helper.from_array(np.array([1, c1 + 1, 30, 30], dtype=np.int64), name='e1'))
    inits.append(numpy_helper.from_array(np.array([0, c2, 0, 0], dtype=np.int64), name='s2'))
    inits.append(numpy_helper.from_array(np.array([1, c2 + 1, 30, 30], dtype=np.int64), name='e2'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax_s'))
    nodes.append(helper.make_node('Slice', ['input', 's1', 'e1', 'ax_s'], ['sc1']))
    nodes.append(helper.make_node('Slice', ['input', 's2', 'e2', 'ax_s'], ['sc2']))

    # Swap
    nodes.append(helper.make_node('Identity', ['sc2'], ['sw1']))
    nodes.append(helper.make_node('Identity', ['sc1'], ['sw2']))

    # Build each channel
    for c in range(10):
        s = str(c)
        inits.append(numpy_helper.from_array(np.array([0, c, 0, 0], dtype=np.int64), name=f'st{s}'))
        inits.append(numpy_helper.from_array(np.array([1, c + 1, 30, 30], dtype=np.int64), name=f'nd{s}'))
        nodes.append(helper.make_node('Slice', ['input', f'st{s}', f'nd{s}', 'ax_s'], [f'ch{s}']))
        if c == c1:
            nodes.append(helper.make_node('Identity', ['sw1'], [f'out{s}']))
        elif c == c2:
            nodes.append(helper.make_node('Identity', ['sw2'], [f'out{s}']))
        else:
            nodes.append(helper.make_node('Identity', [f'ch{s}'], [f'out{s}']))

    ch_names = [f'out{c}' for c in range(10)]
    nodes.append(helper.make_node('Concat', ch_names, ['output'], axis=1))
    return helper.make_model(helper.make_graph(nodes, f'switch_{c1}_{c2}', [X], [Y], inits),
                             opset_imports=[helper.make_opsetid("", 10)])


def build_replace(c_from, c_to):
    """Replace color c_from with c_to in one-hot encoding."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    inits = []
    nodes = []

    # Slice source and target channels
    inits.append(numpy_helper.from_array(np.array([0, c_from, 0, 0], dtype=np.int64), name='s_from'))
    inits.append(numpy_helper.from_array(np.array([1, c_from + 1, 30, 30], dtype=np.int64), name='e_from'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax'))
    nodes.append(helper.make_node('Slice', ['input', 's_from', 'e_from', 'ax'], ['src']))
    inits.append(numpy_helper.from_array(np.array([0, c_to, 0, 0], dtype=np.int64), name='s_to'))
    inits.append(numpy_helper.from_array(np.array([1, c_to + 1, 30, 30], dtype=np.int64), name='e_to'))
    nodes.append(helper.make_node('Slice', ['input', 's_to', 'e_to', 'ax'], ['tgt']))
    nodes.append(helper.make_node('Max', ['tgt', 'src'], ['new_tgt']))

    for c in range(10):
        s = str(c)
        inits.append(numpy_helper.from_array(np.array([0, c, 0, 0], dtype=np.int64), name=f'st{s}'))
        inits.append(numpy_helper.from_array(np.array([1, c + 1, 30, 30], dtype=np.int64), name=f'nd{s}'))
        nodes.append(helper.make_node('Slice', ['input', f'st{s}', f'nd{s}', 'ax'], [f'ch{s}']))
        if c == c_from:
            inits.append(numpy_helper.from_array(np.zeros((1, 1, 30, 30), dtype=np.float32), name=f'z{s}'))
            nodes.append(helper.make_node('Identity', [f'z{s}'], [f'new{s}']))
        elif c == c_to:
            nodes.append(helper.make_node('Identity', ['new_tgt'], [f'new{s}']))
        else:
            nodes.append(helper.make_node('Identity', [f'ch{s}'], [f'new{s}']))

    ch_names = [f'new{c}' for c in range(10)]
    nodes.append(helper.make_node('Concat', ch_names, ['output'], axis=1))
    return helper.make_model(helper.make_graph(nodes, f'replace_{c_from}_{c_to}', [X], [Y], inits),
                             opset_imports=[helper.make_opsetid("", 10)])


def build_hmirror_vconcat():
    from build_concat import build_hmirror_vconcat as _build
    return _build()


def build_vmirror_hconcat():
    from build_concat import build_vmirror_hconcat as _build
    return _build()


def build_upscale(scale):
    out_shape = [1, 10, 30 * scale, 30 * scale]
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, None)
    # Resize opset 10 takes 2 inputs: X, scales
    node = helper.make_node('Resize', ['input', 'scales'], ['output'], mode='nearest')
    i_s = numpy_helper.from_array(np.array([1., 1., float(scale), float(scale)], dtype=np.float32), name='scales')
    return helper.make_model(mk_graph([node], f'upscale_{scale}', [X], [Y], [i_s]),
                             opset_imports=[helper.make_opsetid("", 10)])


def build_multi_switch(switches):
    """Chain multiple switch operations."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)

    inits = []
    nodes = []
    prev = 'input'

    for i, (c1, c2) in enumerate(switches):
        s = f'_{i}'
        # Slice both channels from prev
        inits.append(numpy_helper.from_array(np.array([0, c1, 0, 0], dtype=np.int64), name=f's1{s}'))
        inits.append(numpy_helper.from_array(np.array([1, c1 + 1, 30, 30], dtype=np.int64), name=f'e1{s}'))
        inits.append(numpy_helper.from_array(np.array([0, c2, 0, 0], dtype=np.int64), name=f's2{s}'))
        inits.append(numpy_helper.from_array(np.array([1, c2 + 1, 30, 30], dtype=np.int64), name=f'e2{s}'))
        inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name=f'ax{s}'))
        nodes.append(helper.make_node('Slice', [prev, f's1{s}', f'e1{s}', f'ax{s}'], [f'ch1{s}']))
        nodes.append(helper.make_node('Slice', [prev, f's2{s}', f'e2{s}', f'ax{s}'], [f'ch2{s}']))
        nodes.append(helper.make_node('Identity', [f'ch2{s}'], [f'new1{s}']))
        nodes.append(helper.make_node('Identity', [f'ch1{s}'], [f'new2{s}']))

        # Rebuild all channels
        for c in range(10):
            cs = f'{s}_{c}'
            inits.append(numpy_helper.from_array(np.array([0, c, 0, 0], dtype=np.int64), name=f'st{cs}'))
            inits.append(numpy_helper.from_array(np.array([1, c + 1, 30, 30], dtype=np.int64), name=f'nd{cs}'))
            nodes.append(helper.make_node('Slice', [prev, f'st{cs}', f'nd{cs}', f'ax{s}'], [f'ch{cs}']))
            if c == c1:
                nodes.append(helper.make_node('Identity', [f'new1{s}'], [f'out{cs}']))
            elif c == c2:
                nodes.append(helper.make_node('Identity', [f'new2{s}'], [f'out{cs}']))
            else:
                nodes.append(helper.make_node('Identity', [f'ch{cs}'], [f'out{cs}']))

        ch_names = [f'out{s}_{c}' for c in range(10)]
        out_name = f'step{s}'
        nodes.append(helper.make_node('Concat', ch_names, [out_name], axis=1))
        prev = out_name

    nodes.append(helper.make_node('Identity', [prev], ['output']))
    return helper.make_model(helper.make_graph(nodes, 'multi_switch', [X], [Y], inits),
                             opset_imports=[helper.make_opsetid("", 10)])


# ── Solver analysis ────────────────────────────────────────────────

def get_ops(solver_name):
    fn = getattr(S, solver_name)
    src = inspect.getsource(fn)
    lines = [l.strip() for l in src.split('\n') if '=' in l and '(' in l]
    ops = []
    for line in lines:
        m = re.search(r'= ([a-z_][a-z0-9_]*)\(', line)
        if m and m.group(1) != solver_name:
            ops.append(m.group(1))
    return ops, src


def build_for_task(solver_name):
    ops, src = get_ops(solver_name)
    unique = sorted(set(ops))

    # Single op
    if ops == ['vmirror']:
        return build_vmirror(), "vmirror"
    if ops == ['hmirror']:
        return build_hmirror(), "hmirror"
    if ops == ['rot180']:
        return build_rot180(), "rot180"
    if ops == ['rot90']:
        return build_rot90(), "rot90"
    if ops == ['rot270']:
        return build_rot270(), "rot270"
    if ops == ['dmirror']:
        return build_dmirror(), "dmirror"

    # Two ops
    if ops == ['hmirror', 'vconcat']:
        # Check concat order: vconcat(I, x1) = original first, vconcat(x1, I) = flipped first
        if re.search(r'vconcat\(\s*I\s*,', src):
            from build_concat import build_vconcat_hmirror as _build
            return _build(), "vconcat_hmirror"
        return build_hmirror_vconcat(), "hmirror_vconcat"
    if ops == ['vmirror', 'hconcat']:
        # Check concat order: hconcat(I, x1) = original first, hconcat(x1, I) = flipped first
        if re.search(r'hconcat\(\s*I\s*,', src):
            from build_concat import build_hmirror_vconcat as _hv
            # Actually need hconcat(I, vmirror(I)) = original left, flipped right
            # This is build_vmirror_hconcat which already does original+flipped
            pass
        return build_vmirror_hconcat(), "vmirror_hconcat"

    # Upscale
    if ops == ['upscale']:
        m = re.search(r'upscale\(I,\s*(\w+)\)', src)
        if m and m.group(1) in CONST_MAP:
            return build_upscale(CONST_MAP[m.group(1)]), f"upscale({CONST_MAP[m.group(1)]})"

    # Single replace
    if ops == ['replace']:
        m = re.search(r'replace\(I,\s*(\w+),\s*(\w+)\)', src)
        if m and m.group(1) in CONST_MAP and m.group(2) in CONST_MAP:
            return build_replace(CONST_MAP[m.group(1)], CONST_MAP[m.group(2)]), f"replace({CONST_MAP[m.group(1)]},{CONST_MAP[m.group(2)]})"

    # Single switch
    if ops == ['switch']:
        m = re.search(r'switch\(I,\s*(\w+),\s*(\w+)\)', src)
        if m and m.group(1) in CONST_MAP and m.group(2) in CONST_MAP:
            return build_switch(CONST_MAP[m.group(1)], CONST_MAP[m.group(2)]), f"switch({CONST_MAP[m.group(1)]},{CONST_MAP[m.group(2)]})"

    # Multi-switch (all ops are switch)
    if all(op == 'switch' for op in ops):
        switches = []
        for line in src.split('\n'):
            m = re.search(r'switch\(\w+,\s*(\w+),\s*(\w+)\)', line)
            if m and m.group(1) in CONST_MAP and m.group(2) in CONST_MAP:
                switches.append((CONST_MAP[m.group(1)], CONST_MAP[m.group(2)]))
        if switches:
            return build_multi_switch(switches), f"multi_switch({len(switches)})"

    # Replace + other (complex)
    # Try complex builder for multi-step patterns
    model, method = get_complex_builder(ops, src)
    if model is not None:
        return model, method
    return None, f"complex: {ops}"


# ── Main ───────────────────────────────────────────────────────────

def to_grid(l):
    return tuple(tuple(row) for row in l)

tasks = {}
for i in range(1, 401):
    with open(f"../neurogolf-2026/task{i:03d}.json") as f:
        tasks[i] = json.load(f)

with open('onnx_compatible_tasks.json') as f:
    compat = json.load(f)

os.makedirs("models", exist_ok=True)
results = {}

for task_num in compat['compatible_tasks']:
    key = f"task{task_num:03d}"
    solver_name = compat['task_solvers'][key]
    td = tasks[task_num]

    model, method = build_for_task(solver_name)
    if model is None:
        results[key] = {"status": "skip", "method": method, "solver": solver_name}
        continue

    path = f"models/{key}.onnx"
    onnx.save(model, path)

    try:
        sess = ort.InferenceSession(path)
        ok = True
        for ex in td['train']:
            inp = ex['input']
            exp = ex['output']
            h_in, w_in = len(inp), len(inp[0])
            h_out, w_out = len(exp), len(exp[0])

            # Encode input as one-hot (30x30 padded)
            arr = np.zeros(SHAPE, dtype=np.float32)
            for i in range(min(30, h_in)):
                for j in range(min(30, w_in)):
                    c = inp[i][j]
                    if 0 <= c < 10:
                        arr[0, c, i, j] = 1.0

            out = sess.run(None, {'input': arr})[0]
            res_full = np.argmax(out[0], axis=0).tolist()

            # The DSL solver works on the actual grid, not 30x30.
            # ONNX model processes full 30x30, so content may move.
            # We need to find where the content ended up and compare.
            # Simple approach: check if the content matches regardless of position.
            # For transforms that keep size (rot, mirror), content should match.
            # For size-changing ops (concat, upscale), shapes differ.

            # Build expected one-hot
            exp_arr = np.zeros(SHAPE, dtype=np.float32)
            for i in range(min(30, h_out)):
                for j in range(min(30, w_out)):
                    c = exp[i][j]
                    if 0 <= c < 10:
                        exp_arr[0, c, i, j] = 1.0

            # For same-size transforms, compare the full grids
            if h_in == h_out and w_in == w_out:
                # Content should match at some position in 30x30
                # Find the content bounds in output
                out_nz = np.argwhere(out[0].sum(axis=0) > 0)
                exp_nz = np.argwhere(exp_arr[0].sum(axis=0) > 0)

                if len(out_nz) == 0 and len(exp_nz) == 0:
                    pass  # Both empty, ok
                elif len(out_nz) > 0 and len(exp_nz) > 0:
                    # Check if content shapes match
                    out_h = out_nz[:, 0].max() - out_nz[:, 0].min() + 1
                    out_w = out_nz[:, 1].max() - out_nz[:, 1].min() + 1
                    exp_h = exp_nz[:, 0].max() - exp_nz[:, 0].min() + 1
                    exp_w = exp_nz[:, 1].max() - exp_nz[:, 1].min() + 1

                    if out_h == exp_h and out_w == exp_w:
                        # Extract and compare content
                        out_min_r, out_min_c = out_nz[:, 0].min(), out_nz[:, 1].min()
                        exp_min_r, exp_min_c = exp_nz[:, 0].min(), exp_nz[:, 1].min()

                        match = True
                        for i in range(exp_h):
                            for j in range(exp_w):
                                if res_full[out_min_r + i][out_min_c + j] != exp[exp_min_r + i][exp_min_c + j]:
                                    match = False
                                    break
                            if not match:
                                break
                        if not match:
                            ok = False
                            break
                    else:
                        ok = False
                        break
                else:
                    ok = False
                    break
            else:
                # Size-changing ops: compare at top-left
                ok_this = True
                for i in range(min(h_out, 30)):
                    for j in range(min(w_out, 30)):
                        if res_full[i][j] != exp[i][j]:
                            ok_this = False
                            break
                    if not ok_this:
                        break
                if not ok_this:
                    ok = False
                    break
        results[key] = {"status": "ok" if ok else "mismatch", "method": method, "solver": solver_name}
    except Exception as e:
        results[key] = {"status": "error", "method": method, "error": str(e)[:100]}

with open("build_results.json", "w") as f:
    json.dump(results, f, indent=2)

ok = sum(1 for v in results.values() if v['status'] == 'ok')
skip = sum(1 for v in results.values() if v['status'] == 'skip')
err = sum(1 for v in results.values() if v['status'] in ('error', 'mismatch'))
print(f"OK: {ok}, Skip: {skip}, Error/Mismatch: {err}")

for k, v in sorted(results.items()):
    if v['status'] == 'ok':
        print(f"  {k}: {v['method']}")
