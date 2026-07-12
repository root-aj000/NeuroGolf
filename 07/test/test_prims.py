"""Tests for ONNX primitive implementations — all 151 primitives."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from onnx_prims import (
    OnnxBuilder,
    _make_input_array,
    _to_grid,
    prim_add,
    prim_argmax,
    prim_argmin,
    prim_backdrop,
    prim_both,
    prim_bottomhalf,
    prim_box,
    prim_branch,
    prim_canvas,
    prim_cellwise,
    prim_center,
    prim_centerofmass,
    prim_cmirror,
    prim_color,
    prim_colorcount,
    prim_colorfilter,
    prim_combine,
    prim_compress,
    prim_connect,
    prim_contained,
    prim_corners,
    prim_cover,
    prim_increment,
    prim_decrement,
    prim_crop,
    prim_apply,
    prim_chain,
    prim_compose,
    prim_astuple,
    prim_initset,
    prim_dedupe,
    prim_delta,
    prim_difference,
    prim_divide,
    prim_dmirror,
    prim_dneighbors,
    prim_double,
    prim_downscale,
    prim_equality,
    prim_even,
    prim_extract,
    prim_fgpartition,
    prim_fill,
    prim_first,
    prim_flip,
    prim_fork,
    prim_frontiers,
    prim_gravitate,
    prim_greater,
    prim_halve,
    prim_hconcat,
    prim_height,
    prim_hfrontier,
    prim_hline,
    prim_hmirror,
    prim_hperiod,
    prim_hsplit,
    prim_hupscale,
    prim_inbox,
    prim_index,
    prim_insert,
    prim_intersection,
    prim_interval,
    prim_invert,
    prim_last,
    prim_lbind,
    prim_leastcolor,
    prim_leastcommon,
    prim_lefthalf,
    prim_leftmost,
    prim_llcorner,
    prim_lowermost,
    prim_lrcorner,
    prim_mapply,
    prim_matcher,
    prim_maximum,
    prim_merge,
    prim_mfilter,
    prim_minimum,
    prim_mostcolor,
    prim_mostcommon,
    prim_move,
    prim_mpapply,
    prim_multiply,
    prim_neighbors,
    prim_normalize,
    prim_numcolors,
    prim_objects,
    prim_occurrences,
    prim_ofcolor,
    prim_order,
    prim_other,
    prim_outbox,
    prim_paint,
    prim_pair,
    prim_palette,
    prim_papply,
    prim_partition,
    prim_portrait,
    prim_position,
    prim_positive,
    prim_power,
    prim_prapply,
    prim_product,
    prim_rapply,
    prim_rbind,
    prim_recolor,
    prim_remove,
    prim_repeat,
    prim_replace,
    prim_righthalf,
    prim_rot90,
    prim_rot180,
    prim_rot270,
    prim_sfilter,
    prim_shape,
    prim_shift,
    prim_shoot,
    prim_sign,
    prim_size,
    prim_sizefilter,
    prim_subgrid,
    prim_subtract,
    prim_switch,
    prim_toindices,
    prim_toivec,
    prim_tojvec,
    prim_toobject,
    prim_tophalf,
    prim_totuple,
    prim_trim,
    prim_ulcorner,
    prim_underfill,
    prim_underpaint,
    prim_uppermost,
    prim_upscale,
    prim_urcorner,
    prim_valmax,
    prim_valmin,
    prim_vconcat,
    prim_vfrontier,
    prim_vline,
    prim_vmatching,
    prim_vmirror,
    prim_vperiod,
    prim_vsplit,
    prim_vupscale,
    prim_width,
    prim_asindices,
    prim_asobject,
    prim_crement,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_grid():
    grid = np.zeros((30, 30), dtype=np.int64)
    grid[5, 10] = 3
    grid[5, 11] = 3
    grid[6, 10] = 3
    grid[15, 20] = 7
    grid[20, 5] = 2
    grid[25, 25] = 9
    return grid

@pytest.fixture
def test_onehot(test_grid):
    return _make_input_array(test_grid)

@pytest.fixture
def mask30(test_grid):
    return test_grid.astype(np.float32)


# ============================================================================
# OnnxBuilder
# ============================================================================

class TestOnnxBuilder:
    def test_basic_graph(self):
        bld = OnnxBuilder()
        a = bld.add_input("a", [2, 3])
        b = bld.add_input("b", [2, 3])
        bld.add_node("Add", [a, b], "out")
        bld.add_output("out", [2, 3])
        assert bld.build_model() is not None

    def test_run(self):
        bld = OnnxBuilder()
        a = bld.add_input("a", [2, 2])
        b = bld.add_input("b", [2, 2])
        bld.add_node("Add", [a, b], "out")
        bld.add_output("out", [2, 2])
        result = bld.run({
            "a": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "b": np.array([[5, 6], [7, 8]], dtype=np.float32)
        })
        np.testing.assert_array_equal(result["out"], [[6, 8], [10, 12]])


# ============================================================================
# Helpers
# ============================================================================

class TestHelpers:
    def test_make_input_array(self):
        padded = np.zeros((30, 30), dtype=np.int64)
        padded[0, 0] = 0; padded[0, 1] = 1; padded[1, 0] = 2; padded[1, 1] = 3
        onehot = _make_input_array(padded)
        assert onehot.shape == (1, 10, 30, 30)
        assert onehot[0, 1, 0, 1] == 1.0
        assert onehot[0, 3, 1, 1] == 1.0

    def test_to_grid(self):
        onehot = np.zeros((1, 10, 30, 30), dtype=np.float32)
        onehot[0, 3, 5, 10] = 1.0
        onehot[0, 7, 15, 20] = 1.0
        grid = _to_grid(onehot)
        assert grid[5, 10] == 3
        assert grid[15, 20] == 7


# ============================================================================
# Primitives 1-30: add through crop + combinators
# ============================================================================

class TestPrims1_30:
    def test_add(self):
        a = np.random.rand(1, 10, 30, 30).astype(np.float32)
        b = np.random.rand(1, 10, 30, 30).astype(np.float32)
        np.testing.assert_allclose(prim_add(a, b), a + b)

    def test_argmax(self):
        x = np.random.rand(1, 10, 30, 30).astype(np.float32)
        assert prim_argmax(x, axis=2).shape[0] == 1

    def test_argmin(self):
        x = np.random.rand(1, 10, 30, 30).astype(np.float32)
        assert prim_argmin(x, axis=2).shape[0] == 1

    def test_backdrop(self):
        mask = np.zeros((30, 30), dtype=np.float32)
        mask[5, 10] = 1; mask[15, 20] = 1
        r = prim_backdrop(mask)
        assert r[5, 10] == 1.0 and r[15, 20] == 1.0 and r[10, 15] == 1.0

    def test_both(self):
        a = np.array([[[[1, 0], [1, 1]]]], dtype=np.float32)
        b = np.array([[[[1, 1], [0, 1]]]], dtype=np.float32)
        r = prim_both(a, b)
        assert r[0, 0, 0, 0] == 1 and r[0, 0, 0, 1] == 0

    def test_bottomhalf(self, test_onehot):
        assert prim_bottomhalf(test_onehot).shape == (1, 10, 15, 30)

    def test_box(self):
        mask = np.zeros((30, 30), dtype=np.float32)
        mask[5:20, 10:25] = 1
        r = prim_box(mask)
        assert r[5, 10] == 1.0 and r[10, 15] == 0.0

    def test_branch(self):
        cond = np.array([[[[True, False]]]])
        t = np.array([[[[1.0, 2.0]]]])
        f = np.array([[[[3.0, 4.0]]]])
        r = prim_branch(cond, t, f)
        assert r[0, 0, 0, 0] == 1.0 and r[0, 0, 0, 1] == 4.0

    def test_canvas(self):
        r = prim_canvas(5)
        assert r.shape == (1, 10, 30, 30) and r[0, 5].sum() == 900

    def test_cellwise(self):
        g1 = np.random.rand(1, 10, 30, 30).astype(np.float32)
        g2 = np.random.rand(1, 10, 30, 30).astype(np.float32)
        np.testing.assert_allclose(prim_cellwise(g1, g2, "add"), g1 + g2)

    def test_center(self):
        mask = np.zeros((30, 30), dtype=np.float32)
        mask[10, 15] = 1; mask[20, 25] = 1
        assert prim_center(mask) == (15, 20)

    def test_centerofmass(self, test_onehot):
        r, c = prim_centerofmass(test_onehot)
        assert isinstance(r, int) and isinstance(c, int)

    def test_cmirror(self, test_onehot):
        assert prim_cmirror(test_onehot).shape == (1, 10, 30, 30)

    def test_color(self):
        oh = np.zeros((1, 10, 30, 30), dtype=np.float32)
        oh[0, 3] = 1.0
        assert prim_color(oh) == 3

    def test_colorcount(self, test_onehot):
        assert prim_colorcount(test_onehot, 3) == 3

    def test_colorfilter(self):
        obj1 = np.zeros((1, 10, 30, 30), dtype=np.float32); obj1[0, 3] = 1.0
        obj2 = np.zeros((1, 10, 30, 30), dtype=np.float32); obj2[0, 5] = 1.0
        assert len(prim_colorfilter([obj1, obj2], 3)) == 1

    def test_combine(self):
        a = np.random.rand(5, 10, 30, 30).astype(np.float32)
        b = np.random.rand(3, 10, 30, 30).astype(np.float32)
        assert prim_combine(a, b).shape == (8, 10, 30, 30)

    def test_compress(self):
        grid = np.random.rand(1, 10, 30, 30).astype(np.float32)
        mask = np.zeros((30,), dtype=np.float32); mask[5:10] = 1
        assert prim_compress(grid, mask, axis=2).shape[2] == 5

    def test_connect(self):
        r = prim_connect((0, 0), (29, 29))
        assert r[0, 0] == 1.0 and r[29, 29] == 1.0

    def test_contained(self):
        assert prim_contained(3, [1, 2, 3]) == True
        assert prim_contained(4, [1, 2, 3]) == False

    def test_corners(self):
        mask = np.zeros((30, 30), dtype=np.float32)
        mask[5, 10] = 1; mask[15, 20] = 1
        assert len(prim_corners(mask)) == 4

    def test_cover(self, test_onehot):
        mask = np.zeros((30, 30), dtype=np.float32); mask[5, 10] = 1
        r = prim_cover(test_onehot, mask, 8)
        assert _to_grid(r)[5, 10] == 8

    def test_increment(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(prim_increment(x), [2.0, 3.0, 4.0])

    def test_decrement(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(prim_decrement(x), [0.0, 1.0, 2.0])

    def test_crop(self, test_onehot):
        assert prim_crop(test_onehot, 5, 10, 10, 10).shape == (1, 10, 10, 10)

    def test_apply(self):
        np.testing.assert_array_equal(prim_apply(lambda x: x+1, np.array([1,2,3])), [2,3,4])

    def test_chain(self):
        assert prim_chain(lambda x: x*2, lambda x: x+1, 5) == 11

    def test_compose(self):
        assert prim_compose(lambda x: x+1, lambda x: x*2, 5) == 11

    def test_astuple(self):
        assert prim_astuple(1, 2) == (1, 2)

    def test_initset(self):
        assert 5 in prim_initset(5)


# ============================================================================
# Primitives 31-60: dedupe through inbox
# ============================================================================

class TestPrims31_60:
    def test_dedupe(self):
        assert prim_dedupe([1, 2, 2, 3]) == [1, 2, 3]

    def test_delta(self):
        mask = np.zeros((30, 30), dtype=np.float32)
        mask[10, 10] = 1; mask[10, 11] = 1
        assert isinstance(prim_delta([mask]), set)

    def test_difference(self):
        assert prim_difference({1, 2, 3}, {2}) == {1, 3}

    def test_divide(self):
        a = np.array([10.0, 20.0], dtype=np.float32)
        b = np.array([2.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(prim_divide(a, b), [5.0, 5.0])

    def test_dmirror(self, test_onehot):
        assert prim_dmirror(test_onehot).shape == (1, 10, 30, 30)

    def test_dneighbors(self):
        m = np.zeros((30, 30), dtype=np.float32); m[10, 10] = 1
        assert prim_dneighbors(m).shape == (30, 30)

    def test_double(self):
        np.testing.assert_array_equal(prim_double(np.array([3.0], dtype=np.float32)), [6.0])

    def test_downscale(self, test_onehot):
        assert prim_downscale(test_onehot, 2).shape == (1, 10, 15, 15)

    def test_equality(self):
        assert prim_equality(5, 5) == True
        assert prim_equality(5, 6) == False

    def test_even(self):
        assert prim_even(4) == True and prim_even(3) == False

    def test_fgpartition(self, test_onehot):
        assert len(prim_fgpartition(test_onehot)) > 0

    def test_fill(self, test_onehot):
        mask = np.zeros((30, 30), dtype=np.float32); mask[10, 10] = 1
        assert prim_fill(test_onehot, mask, 5).shape == (1, 10, 30, 30)

    def test_first(self):
        assert prim_first([10, 20, 30]) == 10

    def test_flip(self, test_onehot):
        assert prim_flip(test_onehot).shape == (1, 10, 30, 30)

    def test_fork(self):
        assert prim_fork(lambda x: x+1, lambda x: x*2, 5) == (6, 10)

    def test_frontiers(self, test_onehot):
        assert prim_frontiers(test_onehot).shape == (30, 30)

    def test_gravitate(self, test_onehot):
        prim_gravitate(test_onehot, (1, 0))

    def test_greater(self):
        a = np.array([5.0], dtype=np.float32)
        assert prim_greater(a, 3)[0] == 1.0

    def test_halve(self):
        np.testing.assert_array_equal(prim_halve(np.array([10.0], dtype=np.float32)), [5.0])

    def test_hconcat(self, test_onehot):
        assert prim_hconcat(test_onehot, test_onehot).shape[3] == 60

    def test_height(self, test_onehot):
        assert prim_height(test_onehot) == 30

    def test_hfrontier(self, mask30):
        assert prim_hfrontier(mask30).shape == (30, 30)

    def test_hline(self):
        g = np.zeros((30, 30), dtype=np.int64); g[15, 5:25] = 3
        assert prim_hline(_make_input_array(g)) == True

    def test_hmirror(self, test_onehot):
        assert prim_hmirror(test_onehot).shape == (1, 10, 30, 30)

    def test_hperiod(self, test_onehot):
        assert prim_hperiod(test_onehot) >= 1

    def test_hsplit(self, test_onehot):
        assert len(prim_hsplit(test_onehot, 2)) == 2

    def test_hupscale(self, test_onehot):
        assert prim_hupscale(test_onehot, 2).shape[3] == 60

    def test_inbox(self, test_onehot):
        assert prim_inbox(test_onehot) == False


# ============================================================================
# Primitives 61-90: index through numcolors
# ============================================================================

class TestPrims61_90:
    def test_index(self, test_onehot):
        assert prim_index(test_onehot, (5, 10)) == 3

    def test_insert(self):
        assert 3 in prim_insert({1, 2}, 3)

    def test_intersection(self):
        assert prim_intersection({1, 2, 3}, {2, 3, 4}) == {2, 3}

    def test_interval(self):
        assert prim_interval(0, 10, 2) == [0, 2, 4, 6, 8]

    def test_invert(self):
        x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        np.testing.assert_allclose(prim_invert(x), [-1.0, 2.0, -3.0])

    def test_last(self):
        assert prim_last([1, 2, 3]) == 3

    def test_lbind(self):
        assert prim_lbind(lambda a, b: a + b, 10)(5) == 15

    def test_leastcolor(self, test_onehot):
        assert isinstance(prim_leastcolor(test_onehot), int)

    def test_leastcommon(self):
        assert prim_leastcommon([1, 1, 2, 3]) == 3

    def test_lefthalf(self, test_onehot):
        assert prim_lefthalf(test_onehot).shape == (1, 10, 30, 15)

    def test_leftmost(self, test_onehot):
        assert isinstance(prim_leftmost(test_onehot), int)

    def test_llcorner(self):
        assert prim_llcorner({(10, 5), (20, 15)}) == (20, 5)

    def test_lowermost(self):
        assert prim_lowermost({(10, 5), (20, 15)}) == 20

    def test_lrcorner(self):
        assert prim_lrcorner({(10, 5), (20, 15)}) == (20, 15)

    def test_mapply(self):
        assert prim_mapply(lambda x: [x, x*2], [1, 2, 3]) == [1, 2, 2, 4, 3, 6]

    def test_matcher(self):
        f = prim_matcher(5)
        assert f(5) == True and f(3) == False

    def test_maximum(self):
        a = np.array([1.0, 5.0, 3.0], dtype=np.float32)
        np.testing.assert_allclose(prim_maximum(a, 3.0), [3.0, 5.0, 3.0])

    def test_merge(self):
        assert prim_merge([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_mfilter(self):
        assert prim_mfilter(lambda x: x > 2, [1, 2, 3, 4]) == [3, 4]

    def test_minimum(self):
        a = np.array([5.0, 2.0, 8.0], dtype=np.float32)
        np.testing.assert_allclose(prim_minimum(a, 3.0), [3.0, 2.0, 3.0])

    def test_mostcolor(self, test_onehot):
        assert isinstance(prim_mostcolor(test_onehot), int)

    def test_mostcommon(self):
        assert prim_mostcommon([1, 1, 2, 3]) == 1

    def test_move(self, test_onehot):
        m = np.zeros((30, 30), dtype=np.float32); m[10, 10] = 1
        assert prim_move(test_onehot, m, (1, 0)).shape == (1, 10, 30, 30)

    def test_mpapply(self):
        assert prim_mpapply(lambda x, y: x+y, [1, 2], [10, 20]) == [11, 22]

    def test_multiply(self):
        a = np.array([3.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(prim_multiply(a, 2.0), [6.0, 8.0])

    def test_neighbors(self):
        m = np.zeros((30, 30), dtype=np.float32); m[10, 10] = 1
        assert prim_neighbors(m).shape == (30, 30)

    def test_normalize(self, test_onehot):
        assert prim_normalize(test_onehot).shape == (1, 10, 30, 30)

    def test_numcolors(self, test_onehot):
        assert prim_numcolors(test_onehot) >= 1


# ============================================================================
# Primitives 91-120: objects through shift
# ============================================================================

class TestPrims91_120:
    def test_objects(self, test_onehot):
        assert len(prim_objects(test_onehot)) > 0

    def test_occurrences(self, test_onehot):
        assert isinstance(prim_occurrences(test_onehot, test_onehot), list)

    def test_ofcolor(self, test_onehot):
        assert prim_ofcolor(test_onehot, 3).shape == (30, 30)

    def test_order(self):
        assert prim_order([3, 1, 2]) == [1, 2, 3]

    def test_other(self):
        assert prim_other({1, 2, 3}, 2) == {1, 3}

    def test_outbox(self, mask30):
        assert prim_outbox(mask30).shape == (30, 30)

    def test_paint(self, test_onehot):
        assert prim_paint(test_onehot, test_onehot).shape == (1, 10, 30, 30)

    def test_pair(self):
        assert prim_pair([1, 2], [3, 4]) == [(1, 3), (2, 4)]

    def test_palette(self, test_onehot):
        assert 0 in prim_palette(test_onehot)

    def test_papply(self):
        assert prim_papply(lambda x, y: x+y, [1, 2], [10, 20]) == [11, 22]

    def test_partition(self, test_onehot):
        assert len(prim_partition(test_onehot)) > 0

    def test_portrait(self, test_onehot):
        assert prim_portrait(test_onehot) == False

    def test_position(self):
        r = prim_position({(5, 10)}, {(20, 20)})
        assert isinstance(r, tuple)

    def test_positive(self):
        assert prim_positive(5) == True and prim_positive(0) == False

    def test_power(self):
        assert prim_power(lambda x: x*2, 3)(1) == 8

    def test_prapply(self):
        assert prim_prapply(lambda x, y: x*y, [1, 2], [3, 4]) == [3, 4, 6, 8]

    def test_product(self):
        assert prim_product([1, 2], [3, 4]) == [(1, 3), (1, 4), (2, 3), (2, 4)]

    def test_rapply(self):
        assert prim_rapply([lambda x: x+1, lambda x: x+2], [10, 20]) == [11, 22]

    def test_rbind(self):
        assert prim_rbind(lambda a, b: a + b, 10)(5) == 15

    def test_recolor(self, test_onehot):
        r = prim_recolor(test_onehot, 5, np.ones((30, 30), dtype=np.float32))
        assert r.shape == (1, 10, 30, 30)

    def test_remove(self):
        assert prim_remove([1, 2, 3], 2) == [1, 3]

    def test_repeat(self):
        assert prim_repeat("a", 3) == ["a", "a", "a"]

    def test_replace(self, test_onehot):
        assert prim_replace(test_onehot, 3, 7).shape == (1, 10, 30, 30)

    def test_righthalf(self, test_onehot):
        assert prim_righthalf(test_onehot).shape == (1, 10, 30, 15)

    def test_rot90(self, test_onehot):
        assert prim_rot90(test_onehot).shape == (1, 10, 30, 30)

    def test_rot180(self, test_onehot):
        assert prim_rot180(test_onehot).shape == (1, 10, 30, 30)

    def test_rot270(self, test_onehot):
        assert prim_rot270(test_onehot).shape == (1, 10, 30, 30)

    def test_sfilter(self):
        assert prim_sfilter(lambda x: x > 0, [0, 1, 2]) == 1

    def test_shape(self, test_onehot):
        assert prim_shape(test_onehot) == (30, 30)

    def test_shift(self, test_onehot):
        assert prim_shift(test_onehot, (2, 3)).shape == (1, 10, 30, 30)


# ============================================================================
# Primitives 121-152: shoot through width
# ============================================================================

class TestPrims121_152:
    def test_shoot(self, test_onehot):
        assert prim_shoot(test_onehot, (0, 0), (1, 1)).shape == (30, 30)

    def test_sign(self):
        assert prim_sign(5) == 1 and prim_sign(-3) == -1 and prim_sign(0) == 0

    def test_size(self, test_onehot):
        assert prim_size(test_onehot) == 6

    def test_sizefilter(self, test_onehot):
        assert len(prim_sizefilter([test_onehot], 6)) == 1

    def test_subgrid(self, test_grid, test_onehot):
        assert prim_subgrid(test_grid, test_onehot).ndim == 4

    def test_subtract(self):
        a = np.array([5.0, 3.0], dtype=np.float32)
        b = np.array([2.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(prim_subtract(a, b), [3.0, 2.0])

    def test_switch(self, test_onehot):
        assert prim_switch(test_onehot, 3, 7).shape == (1, 10, 30, 30)

    def test_toindices(self, test_onehot):
        assert len(prim_toindices(test_onehot)) > 0

    def test_toivec(self):
        assert prim_toivec(5) == (5, 0)

    def test_tojvec(self):
        assert prim_tojvec(5) == (0, 5)

    def test_toobject(self, test_onehot):
        assert len(prim_toobject(test_onehot)) > 0

    def test_tophalf(self, test_onehot):
        assert prim_tophalf(test_onehot).shape == (1, 10, 15, 30)

    def test_totuple(self):
        assert prim_totuple([1, 2, 3]) == (1, 2, 3)

    def test_trim(self, test_onehot):
        assert prim_trim(test_onehot).shape == (1, 10, 28, 28)

    def test_ulcorner(self):
        assert prim_ulcorner({(10, 5), (20, 15)}) == (10, 5)

    def test_underfill(self, test_onehot):
        mask = np.zeros((30, 30), dtype=np.float32); mask[10, 5] = 1
        assert prim_underfill(test_onehot, mask, 5).shape == (1, 10, 30, 30)

    def test_underpaint(self, test_onehot):
        assert prim_underpaint(test_onehot, test_onehot).shape == (1, 10, 30, 30)

    def test_uppermost(self):
        assert prim_uppermost({(10, 5), (20, 15)}) == 10

    def test_upscale(self, test_onehot):
        assert prim_upscale(test_onehot, 2).shape[2] == 60

    def test_urcorner(self):
        assert prim_urcorner({(10, 5), (20, 15)}) == (10, 15)

    def test_valmax(self, test_onehot):
        assert isinstance(prim_valmax(test_onehot, lambda p: p[0]), int)

    def test_valmin(self, test_onehot):
        assert isinstance(prim_valmin(test_onehot, lambda p: p[0]), int)

    def test_vconcat(self, test_onehot):
        assert prim_vconcat(test_onehot, test_onehot).shape[2] == 60

    def test_vfrontier(self, mask30):
        assert prim_vfrontier(mask30).shape == (30, 30)

    def test_vline(self):
        g = np.zeros((30, 30), dtype=np.int64); g[5:25, 15] = 3
        assert prim_vline(_make_input_array(g)) == True

    def test_vmatching(self):
        assert prim_vmatching({(1, 2)}, {(3, 2)}) == True
        assert prim_vmatching({(1, 2)}, {(1, 3)}) == False

    def test_vmirror(self, test_onehot):
        assert prim_vmirror(test_onehot).shape == (1, 10, 30, 30)

    def test_vperiod(self, test_onehot):
        assert prim_vperiod(test_onehot) >= 1

    def test_vsplit(self, test_onehot):
        assert len(prim_vsplit(test_onehot, 2)) == 2

    def test_vupscale(self, test_onehot):
        assert prim_vupscale(test_onehot, 2).shape[2] == 60

    def test_width(self, test_onehot):
        assert prim_width(test_onehot) == 30

    def test_asindices(self, test_onehot):
        assert len(prim_asindices(test_onehot)) > 0

    def test_asobject(self, test_onehot):
        assert len(prim_asobject(test_onehot)) > 0

    def test_crement(self):
        x = np.array([5.0], dtype=np.float32)
        np.testing.assert_array_equal(prim_crement(x, 3), [8.0])
        np.testing.assert_array_equal(prim_crement(x, -2), [3.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
