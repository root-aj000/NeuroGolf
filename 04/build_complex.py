#!/usr/bin/env python3
"""
Build specific ONNX models for complex neurogolf tasks.
Uses opset 11 with Resize (2-input format) and Pad (3-input format).
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

SHAPE = [1, 10, 30, 30]
REV30 = np.arange(29, -1, -1, dtype=np.int64)


def build_task067_hsplit_first():
    """task067: hsplit(I, THREE) then first -> left 1/3 of 3x9 input = 3x3 output."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    inits.append(numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), name='starts'))
    inits.append(numpy_helper.from_array(np.array([1, 10, 3, 3], dtype=np.int64), name='ends'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax'))
    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'ax'], ['sliced']))

    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.array([3], dtype=np.int64), name='a_ch'))
    inits.append(numpy_helper.from_array(np.array([3], dtype=np.int64), name='a_cw'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero'))
    nodes.append(helper.make_node('Sub', ['a_thirty', 'a_ch'], ['a_ph']))
    nodes.append(helper.make_node('Sub', ['a_thirty', 'a_cw'], ['a_pw']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_ph', 'a_pw'], ['a_ps'], axis=0))
    nodes.append(helper.make_node('Pad', ['sliced', 'a_ps', 'a_zero'], ['output'], mode='constant'))

    graph = helper.make_graph(nodes, 'task067', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_task326_crop():
    """task326: crop(I, ORIGIN, TWO_BY_TWO) -> 2x2 at origin."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    inits.append(numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), name='starts'))
    inits.append(numpy_helper.from_array(np.array([1, 10, 2, 2], dtype=np.int64), name='ends'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax'))
    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'ax'], ['cropped']))

    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name='a_ch'))
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name='a_cw'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero'))
    nodes.append(helper.make_node('Sub', ['a_thirty', 'a_ch'], ['a_ph']))
    nodes.append(helper.make_node('Sub', ['a_thirty', 'a_cw'], ['a_pw']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_ph', 'a_pw'], ['a_ps'], axis=0))
    nodes.append(helper.make_node('Pad', ['cropped', 'a_ps', 'a_zero'], ['output'], mode='constant'))

    graph = helper.make_graph(nodes, 'task326', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_task108_rot180_down2_rot180_up4():
    """task108: rot180 -> downscale(2) -> rot180 -> upscale(4). 10x10 -> 20x20."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='_1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name='_10'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='zero'))
    inits.append(numpy_helper.from_array(np.arange(9, -1, -1, dtype=np.int64), name='rev10'))
    inits.append(numpy_helper.from_array(np.arange(4, -1, -1, dtype=np.int64), name='rev5'))
    inits.append(numpy_helper.from_array(np.arange(19, -1, -1, dtype=np.int64), name='rev20'))

    # Content detection (10x10 at top-left)
    inits.append(numpy_helper.from_array(np.arange(29, -1, -1, dtype=np.int64), name='rev30'))
    nodes.append(helper.make_node('ReduceMax', ['input'], ['bm'], axes=[1], keepdims=0))
    nodes.append(helper.make_node('ReduceMax', ['bm'], ['rm'], axes=[2], keepdims=0))
    nodes.append(helper.make_node('Gather', ['rm', 'rev30'], ['rrm'], axis=1))
    nodes.append(helper.make_node('ArgMax', ['rrm'], ['hi'], axis=1, keepdims=0))
    nodes.append(helper.make_node('Sub', ['thirty', 'hi'], ['ch']))
    nodes.append(helper.make_node('ReduceMax', ['bm'], ['cm'], axes=[1], keepdims=0))
    nodes.append(helper.make_node('Gather', ['cm', 'rev30'], ['rcm'], axis=1))
    nodes.append(helper.make_node('ArgMax', ['rcm'], ['wi'], axis=1, keepdims=0))
    nodes.append(helper.make_node('Sub', ['thirty', 'wi'], ['cw']))

    # Slice content: 10x10
    nodes.append(helper.make_node('Concat', ['_1', '_10', 'ch', 'cw'], ['c_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'z4', 'c_ends', 'ax'], ['content']))

    # rot180 on 10x10
    nodes.append(helper.make_node('Gather', ['content', 'rev10'], ['tmp'], axis=2))
    nodes.append(helper.make_node('Gather', ['tmp', 'rev10'], ['rot180_1'], axis=3))

    # downscale by 2: 10x10 -> 5x5 (opset 11 Resize: X, roi, scales)
    inits.append(numpy_helper.from_array(np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32), name='scales1'))
    inits.append(numpy_helper.from_array(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32), name='roi1'))
    nodes.append(helper.make_node('Resize', ['rot180_1', 'roi1', 'scales1'], ['down'], mode='nearest', coordinate_transformation_mode='asymmetric'))

    # rot180 on 5x5
    nodes.append(helper.make_node('Gather', ['down', 'rev5'], ['tmp2'], axis=2))
    nodes.append(helper.make_node('Gather', ['tmp2', 'rev5'], ['rot180_2'], axis=3))

    # upscale by 4: 5x5 -> 20x20 (opset 11 Resize: X, roi, scales)
    inits.append(numpy_helper.from_array(np.array([1.0, 1.0, 4.0, 4.0], dtype=np.float32), name='scales2'))
    inits.append(numpy_helper.from_array(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32), name='roi2'))
    nodes.append(helper.make_node('Resize', ['rot180_2', 'roi2', 'scales2'], ['up'], mode='nearest', coordinate_transformation_mode='asymmetric'))

    # Pad 20x20 to 30x30 (opset 10 Pad takes 3 inputs)
    inits.append(numpy_helper.from_array(np.array([20], dtype=np.int64), name='out_h'))
    inits.append(numpy_helper.from_array(np.array([20], dtype=np.int64), name='out_w'))
    nodes.append(helper.make_node('Sub', ['thirty', 'out_h'], ['ph']))
    nodes.append(helper.make_node('Sub', ['thirty', 'out_w'], ['pw']))
    nodes.append(helper.make_node('Concat', ['z', 'z', 'z', 'z', 'z', 'z', 'ph', 'pw'], ['ps'], axis=0))
    nodes.append(helper.make_node('Pad', ['up', 'ps', 'zero'], ['output'], mode='constant'))

    graph = helper.make_graph(nodes, 'task108',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)],
        inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_task130_replace_downscale():
    """task130: replace(5,0) then downscale(3). 9x9 -> 3x3."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    inits.append(numpy_helper.from_array(REV30, name='rev'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='_1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name='_10'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='_30'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='zero'))

    # Content detection (9x9)
    nodes.append(helper.make_node('ReduceMax', ['input'], ['bm'], axes=[1], keepdims=0))
    nodes.append(helper.make_node('ReduceMax', ['bm'], ['rm'], axes=[2], keepdims=0))
    nodes.append(helper.make_node('Gather', ['rm', 'rev'], ['rrm'], axis=1))
    nodes.append(helper.make_node('ArgMax', ['rrm'], ['hi'], axis=1, keepdims=0))
    nodes.append(helper.make_node('Sub', ['thirty', 'hi'], ['ch']))
    nodes.append(helper.make_node('ReduceMax', ['bm'], ['cm'], axes=[1], keepdims=0))
    nodes.append(helper.make_node('Gather', ['cm', 'rev'], ['rcm'], axis=1))
    nodes.append(helper.make_node('ArgMax', ['rcm'], ['wi'], axis=1, keepdims=0))
    nodes.append(helper.make_node('Sub', ['thirty', 'wi'], ['cw']))

    # Slice content: 9x9
    nodes.append(helper.make_node('Concat', ['_1', '_10', 'ch', 'cw'], ['c_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'z4', 'c_ends', 'ax'], ['content']))

    # Replace 5->0 in content
    inits.append(numpy_helper.from_array(np.array([0, 5, 0, 0], dtype=np.int64), name='s1'))
    inits.append(numpy_helper.from_array(np.array([1, 6, 30, 30], dtype=np.int64), name='e1'))
    inits.append(numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), name='s2'))
    inits.append(numpy_helper.from_array(np.array([1, 1, 30, 30], dtype=np.int64), name='e2'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax_s'))
    nodes.append(helper.make_node('Slice', ['content', 's1', 'e1', 'ax_s'], ['ch5']))
    nodes.append(helper.make_node('Slice', ['content', 's2', 'e2', 'ax_s'], ['ch0']))

    for c in range(10):
        if c == 0 or c == 5:
            continue
        cs = str(c)
        inits.append(numpy_helper.from_array(np.array([0, c, 0, 0], dtype=np.int64), name=f'st{cs}'))
        inits.append(numpy_helper.from_array(np.array([1, c + 1, 30, 30], dtype=np.int64), name=f'nd{cs}'))
        nodes.append(helper.make_node('Slice', ['content', f'st{cs}', f'nd{cs}', 'ax_s'], [f'ch{cs}']))
    # Replace: 5->0 (merge ch5 into ch0, zero out ch5)
    inits.append(numpy_helper.from_array(np.float32(0), name='zero_f'))
    ch_names = []
    for c in range(10):
        if c == 0:
            nodes.append(helper.make_node('Add', ['ch0', 'ch5'], [f'out{c}']))
        elif c == 5:
            nodes.append(helper.make_node('Mul', ['ch5', 'zero_f'], [f'out{c}']))
        else:
            nodes.append(helper.make_node('Identity', [f'ch{c}'], [f'out{c}']))
        ch_names.append(f'out{c}')
    nodes.append(helper.make_node('Concat', ch_names, ['replaced'], axis=1))

    # Downscale by 3: 9x9 -> 3x3 (opset 11 Resize: X, roi, scales)
    inits.append(numpy_helper.from_array(np.array([1.0, 1.0, 1.0/3, 1.0/3], dtype=np.float32), name='scales'))
    inits.append(numpy_helper.from_array(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32), name='roi'))
    nodes.append(helper.make_node('Resize', ['replaced', 'roi', 'scales'], ['resized'], mode='nearest', coordinate_transformation_mode='asymmetric'))

    # Pad 3x3 to 30x30 (opset 10 Pad takes 3 inputs)
    inits.append(numpy_helper.from_array(np.array([3], dtype=np.int64), name='out_h'))
    inits.append(numpy_helper.from_array(np.array([3], dtype=np.int64), name='out_w'))
    nodes.append(helper.make_node('Sub', ['thirty', 'out_h'], ['ph']))
    nodes.append(helper.make_node('Sub', ['thirty', 'out_w'], ['pw']))
    nodes.append(helper.make_node('Concat', ['z', 'z', 'z', 'z', 'z', 'z', 'ph', 'pw'], ['ps'], axis=0))
    nodes.append(helper.make_node('Pad', ['resized', 'ps', 'zero'], ['output'], mode='constant'))

    graph = helper.make_graph(nodes, 'task130',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)],
        inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def get_complex_builder(solver_name, src):
    """Match solver to complex builder."""
    import re

    if solver_name == 'solve_2dee498d':  # task067: hsplit(I, THREE) then first
        return build_task067_hsplit_first(), "hsplit_first"

    if solver_name == 'solve_d10ecb37':  # task326: crop
        return build_task326_crop(), "crop"

    if solver_name == 'solve_5614dbcf':  # task130: replace then downscale
        m = re.search(r'replace\(I,\s*(\w+),\s*(\w+)\)', src)
        m2 = re.search(r'downscale\([^,]+,\s*(\w+)\)', src)
        if m and m2 and m.group(1) in {'ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT','NINE'}:
            c_from = {'ZERO':0,'ONE':1,'TWO':2,'THREE':3,'FOUR':4,'FIVE':5,'SIX':6,'SEVEN':7,'EIGHT':8,'NINE':9}[m.group(1)]
            c_to = {'ZERO':0,'ONE':1,'TWO':2,'THREE':3,'FOUR':4,'FIVE':5,'SIX':6,'SEVEN':7,'EIGHT':8,'NINE':9}[m.group(2)]
            down = {'ZERO':0,'ONE':1,'TWO':2,'THREE':3,'FOUR':4,'FIVE':5,'SIX':6,'SEVEN':7,'EIGHT':8,'NINE':9}[m2.group(1)]
            return build_task130_replace_downscale(), "replace(5,0)_downscale(3)"

    if solver_name == 'solve_46f33fce':  # task108: rot180 -> downscale(2) -> rot180 -> upscale(4)
        return build_task108_rot180_down2_rot180_up4(), "rot180_down2_rot180_up4"

    return None, f"complex: {solver_name}"