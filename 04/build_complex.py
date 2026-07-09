#!/usr/bin/env python3
"""
Build complex multi-step ONNX models for neurogolf tasks.
Handles: crop, hsplit/first, downscale+replace, rot180+downscale+rot180+upscale, etc.
All models use opset 11 with Resize (2-input format) and Pad (3-input format).
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

SHAPE = [1, 10, 30, 30]
REV30 = np.arange(29, -1, -1, dtype=np.int64)


def _detect_content_dims(nodes, inits, input_name, prefix=''):
    """Detect content height and width from binary mask."""
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

    nodes.append(helper.make_node('ReduceMax', [input_name], [f'{s}bm'], axes=[1], keepdims=0))
    nodes.append(helper.make_node('ReduceMax', [f'{s}bm'], [f'{s}rm'], axes=[2], keepdims=0))
    nodes.append(helper.make_node('Gather', [f'{s}rm', f'{s}rev'], [f'{s}rrm'], axis=1))
    nodes.append(helper.make_node('ArgMax', [f'{s}rrm'], [f'{s}hi'], axis=1, keepdims=0))
    nodes.append(helper.make_node('Sub', [f'{s}thirty', f'{s}hi'], [f'{s}ch']))

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
    nodes.append(helper.make_node('Concat', [f'{s}z', f'{s}z', f'{s}z', f'{s}z', f'{s}z', f'{s}z', f'{s}ph', f'{s}pw'], [f'{s}ps'], axis=0))
    nodes.append(helper.make_node('Pad', [src, f'{s}ps', f'{s}zero'], [out_name], mode='constant'))


def build_crop():
    """Crop to 2x2 at origin (ORIGIN, TWO_BY_TWO)."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    inits.append(numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), name='starts'))
    inits.append(numpy_helper.from_array(np.array([1, 10, 2, 2], dtype=np.int64), name='ends'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='zero'))

    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'ax'], ['cropped']))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name='a_ch'))
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name='a_cw'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))
    nodes.append(helper.make_node('Sub', ['a_thirty', 'a_ch'], ['a_ph']))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))
    nodes.append(helper.make_node('Sub', ['a_thirty', 'a_cw'], ['a_pw']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_ph', 'a_pw'], ['a_ps'], axis=0))
    nodes.append(helper.make_node('Pad', ['cropped', 'a_ps', 'a_zero'], ['output'], mode='constant'))

    graph = helper.make_graph([], 'crop', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_hsplit_first():
    """hsplit(I, THREE) then first -> left 1/3 of input."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    # Slice full content first
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, cw], ['a_content_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'a_z4', 'a_content_ends', 'a_ax'], ['a_content']))

    # hsplit by 3: take first third of width
    inits.append(numpy_helper.from_array(np.array([3], dtype=np.int64), name='a_three'))
    nodes.append(helper.make_node('Div', [cw, 'a_three'], ['a_cw_div3']))

    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, 'a_cw_div3'], ['a_ends'], axis=0))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    nodes.append(helper.make_node('Slice', ['a_content', 'a_z4', 'a_ends', 'a_ax'], ['a_sliced']))

    _pad_to_30(nodes, [], 'a_sliced', ch, 'a_cw_div3', 'output', 'a_')
    inits.append(numpy_helper.from_array(REV30, name='a_rev'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='a__1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name='a__10'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a__30'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))

    graph = helper.make_graph(nodes, 'hsplit_first', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_replace_downscale(c_from, c_to, downscale_factor):
    """Replace color, then downscale."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, cw], ['a_content_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'a_z4', 'a_content_ends', 'a_ax'], ['a_content']))

    # Replace c_from with c_to in the sliced content
    inits.append(numpy_helper.from_array(np.array([0, c_from, 0, 0], dtype=np.int64), name='s1'))
    inits.append(numpy_helper.from_array(np.array([1, c_from + 1, 30, 30], dtype=np.int64), name='e1'))
    inits.append(numpy_helper.from_array(np.array([0, c_to, 0, 0], dtype=np.int64), name='s2'))
    inits.append(numpy_helper.from_array(np.array([1, c_to + 1, 30, 30], dtype=np.int64), name='e2'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax_s'))
    nodes.append(helper.make_node('Slice', ['a_content', 's1', 'e1', 'ax_s'], ['sc1']))
    nodes.append(helper.make_node('Slice', ['a_content', 's2', 'e2', 'ax_s'], ['sc2']))

    if c_from != c_to:
        for c in range(10):
            cs = str(c)
            inits.append(numpy_helper.from_array(np.array([0, c, 0, 0], dtype=np.int64), name=f'st{cs}'))
            inits.append(numpy_helper.from_array(np.array([1, c + 1, 30, 30], dtype=np.int64), name=f'nd{cs}'))
            nodes.append(helper.make_node('Slice', ['a_content', f'st{cs}', f'nd{cs}', 'ax_s'], [f'ch{cs}']))
            if c == c_from:
                nodes.append(helper.make_node('Identity', ['sc2'], [f'out{cs}']))
            elif c == c_to:
                nodes.append(helper.make_node('Identity', ['sc1'], [f'out{cs}']))
            else:
                nodes.append(helper.make_node('Identity', [f'ch{cs}'], [f'out{cs}']))
        ch_names = [f'out{c}' for c in range(10)]
        nodes.append(helper.make_node('Concat', ch_names, ['replaced'], axis=1))
    else:
        nodes.append(helper.make_node('Identity', ['a_content'], ['replaced']))

    # Downscale
    inits.append(numpy_helper.from_array(np.array([1.0, 1.0, 1.0/downscale_factor, 1.0/downscale_factor], dtype=np.float32), name='scales'))
    nodes.append(helper.make_node('Resize', ['replaced', 'scales'], ['resized'], mode='nearest'))

    # Pad: output is ch/downscale x cw/downscale
    inits.append(numpy_helper.from_array(np.array([downscale_factor], dtype=np.int64), name='a_down'))
    nodes.append(helper.make_node('Div', [ch, 'a_down'], ['a_ch_out']))
    nodes.append(helper.make_node('Div', [cw, 'a_down'], ['a_cw_out']))
    _pad_to_30(nodes, [], 'resized', 'a_ch_out', 'a_cw_out', 'output', 'a_')
    inits.append(numpy_helper.from_array(REV30, name='a_rev'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='a__1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name='a__10'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a__30'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))

    graph = helper.make_graph(nodes, 'replace_downscale', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_rot180_downscale_rot180_upscale(down_scale, up_scale):
    """rot180 -> downscale -> rot180 -> upscale."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, cw], ['a_content_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'a_z4', 'a_content_ends', 'a_ax'], ['a_content']))

    # rot180: reverse rows then cols
    nodes.append(helper.make_node('Gather', ['a_content', 'a_rev'], ['a_tmp'], axis=2))
    nodes.append(helper.make_node('Gather', ['a_tmp', 'a_rev'], ['a_rot180_1'], axis=3))

    # Downscale
    inits.append(numpy_helper.from_array(np.array([1.0, 1.0, 1.0/down_scale, 1.0/down_scale], dtype=np.float32), name='scales1'))
    nodes.append(helper.make_node('Resize', ['a_rot180_1', 'scales1'], ['a_down'], mode='nearest'))

    # rot180 again
    nodes.append(helper.make_node('Gather', ['a_down', 'a_rev'], ['a_tmp2'], axis=2))
    nodes.append(helper.make_node('Gather', ['a_tmp2', 'a_rev'], ['a_rot180_2'], axis=3))

    # Upscale
    inits.append(numpy_helper.from_array(np.array([1.0, 1.0, float(up_scale), float(up_scale)], dtype=np.float32), name='scales2'))
    nodes.append(helper.make_node('Resize', ['a_rot180_2', 'scales2'], ['a_up'], mode='nearest'))

    # Pad using original content dimensions (restored after sequence)
    _pad_to_30(nodes, [], 'a_up', ch, cw, 'output', 'a_')
    inits.append(numpy_helper.from_array(REV30, name='a_rev'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='a__1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name='a__10'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a__30'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))

    graph = helper.make_graph(nodes, f'rot180_down{down_scale}_rot180_up{up_scale}', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_crop():
    """Crop to 2x2 at origin (ORIGIN, TWO_BY_TWO)."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    inits.append(numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), name='starts'))
    inits.append(numpy_helper.from_array(np.array([1, 10, 2, 2], dtype=np.int64), name='ends'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='zero'))

    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'ax'], ['cropped']))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name='a_ch'))
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name='a_cw'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))
    nodes.append(helper.make_node('Sub', ['a_thirty', 'a_ch'], ['a_ph']))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))
    nodes.append(helper.make_node('Sub', ['a_thirty', 'a_cw'], ['a_pw']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_ph', 'a_pw'], ['a_ps'], axis=0))
    nodes.append(helper.make_node('Pad', ['cropped', 'a_ps', 'a_zero'], ['output'], mode='constant'))

    graph = helper.make_graph(nodes, 'crop', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_hsplit_first():
    """hsplit(I, THREE) then first -> left 1/3 of input."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, cw], ['a_content_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'a_z4', 'a_content_ends', 'a_ax'], ['a_content']))

    inits.append(numpy_helper.from_array(np.array([3], dtype=np.int64), name='a_three'))
    nodes.append(helper.make_node('Div', [cw, 'a_three'], ['a_cw_div3']))

    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, 'a_cw_div3'], ['a_ends'], axis=0))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    nodes.append(helper.make_node('Slice', ['a_content', 'a_z4', 'a_ends', 'a_ax'], ['a_sliced']))

    _pad_to_30(nodes, [], 'a_sliced', ch, 'a_cw_div3', 'output', 'a_')
    inits.append(numpy_helper.from_array(REV30, name='a_rev'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='a__1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name='a__10'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a__30'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))

    graph = helper.make_graph(nodes, 'hsplit_first', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_crop():
    """Crop to 2x2 at origin (ORIGIN, TWO_BY_TWO)."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    inits.append(numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), name='starts'))
    inits.append(numpy_helper.from_array(np.array([1, 10, 2, 2], dtype=np.int64), name='ends'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='zero'))

    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'ax'], ['cropped']))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name='a_ch'))
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name='a_cw'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))
    nodes.append(helper.make_node('Sub', ['a_thirty', 'a_ch'], ['a_ph']))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))
    nodes.append(helper.make_node('Sub', ['a_thirty', 'a_cw'], ['a_pw']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_ph', 'a_pw'], ['a_ps'], axis=0))
    nodes.append(helper.make_node('Pad', ['cropped', 'a_ps', 'a_zero'], ['output'], mode='constant'))

    graph = helper.make_graph(nodes, 'crop', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_hsplit_first():
    """hsplit(I, THREE) then first -> left 1/3 of input."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, cw], ['a_content_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'a_z4', 'a_content_ends', 'a_ax'], ['a_content']))

    inits.append(numpy_helper.from_array(np.array([3], dtype=np.int64), name='a_three'))
    nodes.append(helper.make_node('Div', [cw, 'a_three'], ['a_cw_div3']))

    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, 'a_cw_div3'], ['a_ends'], axis=0))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    nodes.append(helper.make_node('Slice', ['a_content', 'a_z4', 'a_ends', 'a_ax'], ['a_sliced']))

    _pad_to_30(nodes, [], 'a_sliced', ch, 'a_cw_div3', 'output', 'a_')
    inits.append(numpy_helper.from_array(REV30, name='a_rev'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='a__1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name='a__10'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a__30'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))

    graph = helper.make_graph(nodes, 'hsplit_first', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_replace_downscale(c_from, c_to, downscale_factor):
    """Replace color, then downscale."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, cw], ['a_content_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'a_z4', 'a_content_ends', 'a_ax'], ['a_content']))

    inits.append(numpy_helper.from_array(np.array([0, c_from, 0, 0], dtype=np.int64), name='s1'))
    inits.append(numpy_helper.from_array(np.array([1, c_from + 1, 30, 30], dtype=np.int64), name='e1'))
    inits.append(numpy_helper.from_array(np.array([0, c_to, 0, 0], dtype=np.int64), name='s2'))
    inits.append(numpy_helper.from_array(np.array([1, c_to + 1, 30, 30], dtype=np.int64), name='e2'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax_s'))
    nodes.append(helper.make_node('Slice', ['a_content', 's1', 'e1', 'ax_s'], ['sc1']))
    nodes.append(helper.make_node('Slice', ['a_content', 's2', 'e2', 'ax_s'], ['sc2']))

    if c_from != c_to:
        for c in range(10):
            cs = str(c)
            inits.append(numpy_helper.from_array(np.array([0, c, 0, 0], dtype=np.int64), name=f'st{cs}'))
            inits.append(numpy_helper.from_array(np.array([1, c + 1, 30, 30], dtype=np.int64), name=f'nd{cs}'))
            nodes.append(helper.make_node('Slice', ['a_content', f'st{cs}', f'nd{cs}', 'ax_s'], [f'ch{cs}']))
            if c == c_from:
                nodes.append(helper.make_node('Identity', ['sc2'], [f'out{cs}']))
            elif c == c_to:
                nodes.append(helper.make_node('Identity', ['sc1'], [f'out{cs}']))
            else:
                nodes.append(helper.make_node('Identity', [f'ch{cs}'], [f'out{cs}']))
        ch_names = [f'out{c}' for c in range(10)]
        nodes.append(helper.make_node('Concat', ch_names, ['replaced'], axis=1))
    else:
        nodes.append(helper.make_node('Identity', ['a_content'], ['replaced']))

    inits.append(numpy_helper.from_array(np.array([1.0, 1.0, 1.0/downscale_factor, 1.0/downscale_factor], dtype=np.float32), name='scales'))
    nodes.append(helper.make_node('Resize', ['replaced', 'scales'], ['resized'], mode='nearest'))

    inits.append(numpy_helper.from_array(np.array([downscale_factor], dtype=np.int64), name='a_down'))
    nodes.append(helper.make_node('Div', [ch, 'a_down'], ['a_ch_out']))
    nodes.append(helper.make_node('Div', [cw, 'a_down'], ['a_cw_out']))
    _pad_to_30(nodes, [], 'resized', 'a_ch_out', 'a_cw_out', 'output', 'a_')
    inits.append(numpy_helper.from_array(REV30, name='a_rev'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='a__1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name='a__10'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a__30'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))

    graph = helper.make_graph(nodes, 'replace_downscale', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_rot180_downscale_rot180_upscale(down_scale, up_scale):
    """rot180 -> downscale -> rot180 -> upscale."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, cw], ['a_content_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'a_z4', 'a_content_ends', 'a_ax'], ['a_content']))

    nodes.append(helper.make_node('Gather', ['a_content', 'a_rev'], ['a_tmp'], axis=2))
    nodes.append(helper.make_node('Gather', ['a_tmp', 'a_rev'], ['a_rot180_1'], axis=3))

    inits.append(numpy_helper.from_array(np.array([1.0, 1.0, 1.0/down_scale, 1.0/down_scale], dtype=np.float32), name='scales1'))
    nodes.append(helper.make_node('Resize', ['a_rot180_1', 'scales1'], ['a_down'], mode='nearest'))

    nodes.append(helper.make_node('Gather', ['a_down', 'a_rev'], ['a_tmp2'], axis=2))
    nodes.append(helper.make_node('Gather', ['a_tmp2', 'a_rev'], ['a_rot180_2'], axis=3))

    inits.append(numpy_helper.from_array(np.array([1.0, 1.0, float(up_scale), float(up_scale)], dtype=np.float32), name='scales2'))
    nodes.append(helper.make_node('Resize', ['a_rot180_2', 'scales2'], ['a_up'], mode='nearest'))

    _pad_to_30(nodes, [], 'a_up', ch, cw, 'output', 'a_')
    inits.append(numpy_helper.from_array(REV30, name='a_rev'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='a__1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name='a__10'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a__30'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))

    graph = helper.make_graph(nodes, f'rot180_down{down_scale}_rot180_up{up_scale}', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_crop():
    """Crop to 2x2 at origin (ORIGIN, TWO_BY_TWO)."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    inits.append(numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), name='starts'))
    inits.append(numpy_helper.from_array(np.array([1, 10, 2, 2], dtype=np.int64), name='ends'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='zero'))

    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'ax'], ['cropped']))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name='a_ch'))
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name='a_cw'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))
    nodes.append(helper.make_node('Sub', ['a_thirty', 'a_ch'], ['a_ph']))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero')))
    nodes.append(helper.make_node('Sub', ['a_thirty', 'a_cw'], ['a_pw']))
    nodes.append(helper.make_node('Concat', ['a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_z', 'a_ph', 'a_pw'], ['a_ps'], axis=0))
    nodes.append(helper.make_node('Pad', ['cropped', 'a_ps', 'a_zero'], ['output'], mode='constant'))

    graph = helper.make_graph(nodes, 'crop', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_hsplit_first():
    """hsplit(I, THREE) then first -> left 1/3 of input."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, cw], ['a_content_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'a_z4', 'a_content_ends', 'a_ax'], ['a_content']))

    inits.append(numpy_helper.from_array(np.array([3], dtype=np.int64), name='a_three'))
    nodes.append(helper.make_node('Div', [cw, 'a_three'], ['a_cw_div3']))

    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, 'a_cw_div3'], ['a_ends'], axis=0))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    nodes.append(helper.make_node('Slice', ['a_content', 'a_z4', 'a_ends', 'a_ax'], ['a_sliced']))

    _pad_to_30(nodes, [], 'a_sliced', ch, 'a_cw_div3', 'output', 'a_')
    inits.append(numpy_helper.from_array(REV30, name='a_rev'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='a__1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name='a__10'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a__30'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero'))

    graph = helper.make_graph(nodes, 'hsplit_first', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_replace_downscale(c_from, c_to, downscale_factor):
    """Replace color, then downscale."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, cw], ['a_content_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'a_z4', 'a_content_ends', 'a_ax'], ['a_content']))

    inits.append(numpy_helper.from_array(np.array([0, c_from, 0, 0], dtype=np.int64), name='s1'))
    inits.append(numpy_helper.from_array(np.array([1, c_from + 1, 30, 30], dtype=np.int64), name='e1'))
    inits.append(numpy_helper.from_array(np.array([0, c_to, 0, 0], dtype=np.int64), name='s2'))
    inits.append(numpy_helper.from_array(np.array([1, c_to + 1, 30, 30], dtype=np.int64), name='e2'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='ax_s'))
    nodes.append(helper.make_node('Slice', ['a_content', 's1', 'e1', 'ax_s'], ['sc1']))
    nodes.append(helper.make_node('Slice', ['a_content', 's2', 'e2', 'ax_s'], ['sc2']))

    if c_from != c_to:
        for c in range(10):
            cs = str(c)
            inits.append(numpy_helper.from_array(np.array([0, c, 0, 0], dtype=np.int64), name=f'st{cs}'))
            inits.append(numpy_helper.from_array(np.array([1, c + 1, 30, 30], dtype=np.int64), name=f'nd{cs}'))
            nodes.append(helper.make_node('Slice', ['a_content', f'st{cs}', f'nd{cs}', 'ax_s'], [f'ch{cs}']))
            if c == c_from:
                nodes.append(helper.make_node('Identity', ['sc2'], [f'out{cs}']))
            elif c == c_to:
                nodes.append(helper.make_node('Identity', ['sc1'], [f'out{cs}']))
            else:
                nodes.append(helper.make_node('Identity', [f'ch{cs}'], [f'out{cs}']))
        ch_names = [f'out{c}' for c in range(10)]
        nodes.append(helper.make_node('Concat', ch_names, ['replaced'], axis=1))
    else:
        nodes.append(helper.make_node('Identity', ['a_content'], ['replaced']))

    inits.append(numpy_helper.from_array(np.array([1.0, 1.0, 1.0/downscale_factor, 1.0/downscale_factor], dtype=np.float32), name='scales'))
    nodes.append(helper.make_node('Resize', ['replaced', 'scales'], ['resized'], mode='nearest'))

    inits.append(numpy_helper.from_array(np.array([downscale_factor], dtype=np.int64), name='a_down'))
    nodes.append(helper.make_node('Div', [ch, 'a_down'], ['a_ch_out']))
    nodes.append(helper.make_node('Div', [cw, 'a_down'], ['a_cw_out']))
    _pad_to_30(nodes, [], 'resized', 'a_ch_out', 'a_cw_out', 'output', 'a_')
    inits.append(numpy_helper.from_array(REV30, name='a_rev'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='a__1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name='a__10'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a__30'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero'))

    graph = helper.make_graph(nodes, 'replace_downscale', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def build_rot180_downscale_rot180_upscale(down_scale, up_scale):
    """rot180 -> downscale -> rot180 -> upscale."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)
    nodes, inits = [], []

    ch, cw = _detect_content_dims(nodes, inits, 'input', 'a_')
    nodes.append(helper.make_node('Concat', ['a__1', 'a__10', ch, cw], ['a_content_ends'], axis=0))
    nodes.append(helper.make_node('Slice', ['input', 'a_z4', 'a_content_ends', 'a_ax'], ['a_content']))

    nodes.append(helper.make_node('Gather', ['a_content', 'a_rev'], ['a_tmp'], axis=2))
    nodes.append(helper.make_node('Gather', ['a_tmp', 'a_rev'], ['a_rot180_1'], axis=3))

    inits.append(numpy_helper.from_array(np.array([1.0, 1.0, 1.0/down_scale, 1.0/down_scale], dtype=np.float32), name='scales1'))
    nodes.append(helper.make_node('Resize', ['a_rot180_1', 'scales1'], ['a_down'], mode='nearest'))

    nodes.append(helper.make_node('Gather', ['a_down', 'a_rev'], ['a_tmp2'], axis=2))
    nodes.append(helper.make_node('Gather', ['a_tmp2', 'a_rev'], ['a_rot180_2'], axis=3))

    inits.append(numpy_helper.from_array(np.array([1.0, 1.0, float(up_scale), float(up_scale)], dtype=np.float32), name='scales2'))
    nodes.append(helper.make_node('Resize', ['a_rot180_2', 'scales2'], ['a_up'], mode='nearest'))

    _pad_to_30(nodes, [], 'a_up', ch, cw, 'output', 'a_')
    inits.append(numpy_helper.from_array(REV30, name='a_rev'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a_thirty'))
    inits.append(numpy_helper.from_array(np.zeros(1, dtype=np.int64), name='a_z'))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='a__1'))
    inits.append(numpy_helper.from_array(np.array([10], dtype=np.int64), name='a__10'))
    inits.append(numpy_helper.from_array(np.array([30], dtype=np.int64), name='a__30'))
    inits.append(numpy_helper.from_array(np.zeros(4, dtype=np.int64), name='a_z4'))
    inits.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name='a_ax'))
    inits.append(numpy_helper.from_array(np.float32(0), name='a_zero'))

    graph = helper.make_graph(nodes, f'rot180_down{down_scale}_rot180_up{up_scale}', [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)], [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def get_complex_builder(solver_name, src):
    """Match solver to complex builder."""
    import re

    if solver_name == 'solve_2dee498d':  # task067: hsplit(I, THREE) then first
        return build_hsplit_first(), "hsplit_first"

    if solver_name == 'solve_d10ecb37':  # task326: crop
        return build_crop(), "crop"

    if solver_name == 'solve_5614dbcf':  # task130: replace then downscale
        m = re.search(r'replace\(I,\s*(\w+),\s*(\w+)\)', src)
        m2 = re.search(r'downscale\([^,]+,\s*(\w+)\)', src)
        if m and m2 and m.group(1) in {'ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT','NINE'}:
            c_from = {'ZERO':0,'ONE':1,'TWO':2,'THREE':3,'FOUR':4,'FIVE':5,'SIX':6,'SEVEN':7,'EIGHT':8,'NINE':9}[m.group(1)]
            c_to = {'ZERO':0,'ONE':1,'TWO':2,'THREE':3,'FOUR':4,'FIVE':5,'SIX':6,'SEVEN':7,'EIGHT':8,'NINE':9}[m.group(2)]
            down = {'ZERO':0,'ONE':1,'TWO':2,'THREE':3,'FOUR':4,'FIVE':5,'SIX':6,'SEVEN':7,'EIGHT':8,'NINE':9}[m2.group(1)]
            return build_replace_downscale(c_from, c_to, down), f"replace({c_from},{c_to})_downscale({down})"

    if solver_name == 'solve_46f33fce':  # task108: rot180 -> downscale(2) -> rot180 -> upscale(4)
        return build_rot180_downscale_rot180_upscale(2, 4), "rot180_down2_rot180_up4"

    return None, f"complex: {ops}"