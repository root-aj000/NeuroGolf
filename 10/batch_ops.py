"""
Batch-aware tensor operations for higher-order DSL primitives.

Objects are represented as (MAX_OBJ, 10, H, W) one-hot tensors
with a validity mask (MAX_OBJ,).

CRITICAL: Every function must be torch.jit.trace compatible.
- No Python if-checks on tensor values
- No .item() calls
- No data-dependent loops (all loops unrolled at trace time)
- Use always-compute-then-mask pattern

IMPORTANT: Color 0 (background) is a valid color. To detect content pixels
use (objs.sum(dim=1) > 0), NOT (oh_decode(objs) > 0) which conflates
color 0 with empty.
"""

import torch
import torch.nn.functional as F

NUM_COLORS = 10
MAX_OBJ = 20


def oh_decode(x):
    """(B, 10, H, W) one-hot -> (B, 1, H, W) scalar."""
    return x.argmax(dim=1, keepdim=True).float()


def content_mask(objs):
    """(MAX_OBJ, 10, H, W) -> (MAX_OBJ, 1, H, W) float {0,1}. True where any color channel is set."""
    return (objs.sum(dim=1, keepdim=True) > 0).float()


# ============================================================================
# Frontiers extraction (DSL's frontiers: uniform rows/columns)
# ============================================================================

def frontiers_batch(x):
    """Extract uniform rows and columns as objects (DSL's frontiers).

    x: (1, 10, H, W) one-hot
    Returns: (MAX_OBJ, 10, H, W) one-hot masks + (MAX_OBJ,) valid mask

    A frontier is a row or column where all cells have the same color.
    Fully vectorized for torch.jit.trace compatibility.
    """
    d = oh_decode(x)  # (1, 1, H, W)
    H, W = x.shape[2], x.shape[3]
    color_idx = d.long().squeeze(0).squeeze(0)  # (H, W)

    # Find uniform rows: all cells in row equal to first cell in row
    first_col = color_idx[:, 0:1]  # (H, 1)
    row_uniform_mask = (color_idx == first_col).all(dim=1)  # (H,)
    row_uniform = row_uniform_mask.nonzero(as_tuple=True)[0]  # indices
    row_color = color_idx[row_uniform, 0]  # colors of uniform rows

    # Find uniform columns: all cells in col equal to first cell in col
    first_row = color_idx[0:1, :]  # (1, W)
    col_uniform_mask = (color_idx == first_row).all(dim=0)  # (W,)
    col_uniform = col_uniform_mask.nonzero(as_tuple=True)[0]  # indices
    col_color = color_idx[0, col_uniform]  # colors of uniform cols

    n_rows = row_uniform.shape[0]
    n_cols = col_uniform.shape[0]
    total = n_rows + n_cols
    # Cap at MAX_OBJ
    n_rows_capped = min(n_rows, MAX_OBJ)
    n_cols_capped = min(n_cols, MAX_OBJ - n_rows_capped)

    batch = torch.zeros(MAX_OBJ, NUM_COLORS, H, W, device=x.device)
    valid = torch.zeros(MAX_OBJ, device=x.device)

    if n_rows_capped > 0:
        # Horizontal frontiers (rows)
        idxs = torch.arange(n_rows_capped, device=x.device)
        c = row_color[idxs]
        r = row_uniform[idxs]
        batch[idxs, c, r, :] = 1.0
        valid[idxs] = 1.0

    if n_cols_capped > 0:
        # Vertical frontiers (columns)
        start = n_rows_capped
        idxs = torch.arange(n_cols_capped, device=x.device)
        c = col_color[idxs]
        c_idx = col_uniform[idxs]
        batch[start:start+n_cols_capped, c, :, c_idx] = 1.0
        valid[start:start+n_cols_capped] = 1.0

    return batch, valid


# ============================================================================
# Object extraction (connected components via min-label propagation)
# ============================================================================

def extract_objects_batch(x, univalued=True, diagonal=False, without_bg=True):
    """Extract connected components via iterative min-label propagation.

    x: (1, 10, H, W) one-hot
    Returns: (MAX_OBJ, 10, H, W) one-hot masks + (MAX_OBJ,) valid mask

    First detects actual grid size by finding content bounding box,
    then applies connected components on that region.
    """
    d = oh_decode(x)  # (1, 1, H, W)
    H, W = x.shape[2], x.shape[3]
    color_idx = d.long()

    # Detect actual grid size: find bounding box of all content (including bg=0)
    is_any = (color_idx >= 0).float()
    row_has_any = is_any.squeeze().sum(dim=1) > 0
    col_has_any = is_any.squeeze().sum(dim=0) > 0
    row_cumsum = torch.cumsum(row_has_any.float(), dim=0)
    col_cumsum = torch.cumsum(col_has_any.float(), dim=0)
    last_row = (row_cumsum == row_cumsum[-1]).float().argmax()
    last_col = (col_cumsum == col_cumsum[-1]).float().argmax()
    h_actual = last_row + 1
    w_actual = last_col + 1

    # Slice to actual grid
    grid = x[:, :, :h_actual, :w_actual]
    d_grid = oh_decode(grid)
    H_g, W_g = grid.shape[2], grid.shape[3]
    color_idx_grid = d_grid.long()

    if without_bg:
        mask = (d_grid > 0).float()
    else:
        mask = torch.ones_like(d_grid)

    flat_idx = torch.arange(H_g * W_g, dtype=torch.float32, device=x.device)
    labels = flat_idx.view(1, 1, H_g, W_g).expand_as(mask) * mask

    padded_d = F.pad(color_idx_grid, (1, 1, 1, 1), value=-1)

    if univalued:
        same_color_masks = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_d = padded_d[:, :, 1+dr:H_g+1+dr, 1+dc:W_g+1+dc]
            same_color_masks.append((color_idx_grid == neighbor_d))
        inf_tensor = torch.tensor(float('inf'), device=x.device)

    n_iters = 60  # fixed for trace stability
    for _ in range(n_iters):
        padded = F.pad(labels, (1, 1, 1, 1), value=float('inf'))
        min_label = labels
        if univalued:
            for i, (dr, dc) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
                neighbor = padded[:, :, 1+dr:H_g+1+dr, 1+dc:W_g+1+dc]
                neighbor = torch.where(same_color_masks[i], neighbor, inf_tensor)
                min_label = torch.min(min_label, neighbor)
        else:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = padded[:, :, 1+dr:H_g+1+dr, 1+dc:W_g+1+dc]
                min_label = torch.min(min_label, neighbor)
        labels = torch.where(mask > 0, min_label, labels)

    flat_labels = labels.view(-1).long()
    counts = torch.zeros(H_g * W_g, device=x.device)
    counts.scatter_add_(0, flat_labels, torch.ones(H_g * W_g, device=x.device))
    exists = (counts > 0).float()
    mapping = torch.cumsum(exists, dim=0) - 1
    new_labels = mapping[flat_labels].view(1, 1, H_g, W_g)

    obj_ids = torch.arange(0, MAX_OBJ, dtype=torch.float32, device=x.device).view(MAX_OBJ, 1, 1, 1)
    object_masks = (new_labels == obj_ids).float()

    valid = (object_masks.sum(dim=(1, 2, 3)) > 0).float()

    if without_bg:
        valid[0] = 0.0
        object_masks[0] = 0.0

    # Pad back to original H, W
    batch = torch.zeros(MAX_OBJ, NUM_COLORS, H, W, device=x.device)
    for c in range(NUM_COLORS):
        color_mask = (d == c).float()
        # Use object_masks from actual grid, pad to full size
        batch[:, c:c+1, :h_actual, :w_actual] = (object_masks * color_mask[:, :, :h_actual, :w_actual])

    return batch, valid


# ============================================================================
# Batch predicates (per-object -> boolean) — all trace-compatible
# ============================================================================

def bordering_batch(objs, valid, grid):
    """Check if each object touches the grid border.
    Returns: (MAX_OBJ,) float {0, 1}
    """
    H, W = objs.shape[2], objs.shape[3]
    is_content = content_mask(objs)

    border = torch.zeros(1, 1, H, W, device=objs.device)
    border[0, 0, 0, :] = 1
    border[0, 0, H - 1, :] = 1
    border[0, 0, :, 0] = 1
    border[0, 0, :, W - 1] = 1

    on_border = is_content * border
    result = on_border.sum(dim=(1, 2, 3))
    result = (result > 0).float() * valid
    return result


def hline_batch(objs, valid):
    """Check if each object is a horizontal line (height == 1).
    Returns: (MAX_OBJ,) float {0, 1}
    """
    is_content = content_mask(objs)
    row_has = is_content.sum(dim=3).squeeze(-1)
    h = (row_has > 0).float().sum(dim=2).squeeze(-1)
    return (h <= 1).float() * valid


def vline_batch(objs, valid):
    """Check if each object is a vertical line (width == 1).
    Returns: (MAX_OBJ,) float {0, 1}
    """
    is_content = content_mask(objs)
    col_has = is_content.sum(dim=2).squeeze(-1)
    w = (col_has > 0).float().sum(dim=2).squeeze(-1)
    return (w <= 1).float() * valid


def color_batch(objs, valid):
    """Get dominant non-background color per object.
    Returns: (MAX_OBJ, 10, 1, 1) one-hot
    """
    B = objs.shape[0]
    is_c = content_mask(objs)

    pixel_counts = torch.zeros(B, NUM_COLORS, device=objs.device)
    for c in range(NUM_COLORS):
        pixel_counts[:, c] = (objs[:, c, :, :] * is_c.squeeze(1)).sum(dim=(1, 2))

    pixel_counts[:, 0] = -1
    best = pixel_counts.argmax(dim=1)

    out = torch.zeros(B, NUM_COLORS, 1, 1, device=objs.device)
    idx = torch.arange(B, device=objs.device)
    out[idx, best, 0, 0] = 1.0
    out = out * valid.view(-1, 1, 1, 1)
    return out


def size_batch(objs, valid):
    """Count pixels per object.
    Returns: (MAX_OBJ, 10, 1, 1) one-hot encoded count
    """
    is_c = content_mask(objs)
    counts = is_c.sum(dim=(1, 2, 3)).long().clamp(0, 9)

    B = objs.shape[0]
    out = torch.zeros(B, NUM_COLORS, 1, 1, device=objs.device)
    idx = torch.arange(B, device=objs.device)
    out[idx, counts, 0, 0] = 1.0
    out = out * valid.view(-1, 1, 1, 1)
    return out


def equality_scalar(a, b):
    """Element-wise equality between two tensors.
    Returns: (MAX_OBJ,) float {0, 1}
    """
    return (a == b).float().squeeze(-1)


def numcolors_batch(objs, valid):
    """Count unique non-background colors per object.
    Returns: (MAX_OBJ, 10, 1, 1) one-hot
    """
    B = objs.shape[0]
    is_c = content_mask(objs)

    has_color = torch.zeros(B, NUM_COLORS, device=objs.device)
    for c in range(1, NUM_COLORS):
        has_color[:, c] = ((objs[:, c, :, :] * is_c.squeeze(1)).sum(dim=(1, 2)) > 0).float()

    n_colors = has_color.sum(dim=1).long().clamp(0, 9)

    out = torch.zeros(B, NUM_COLORS, 1, 1, device=objs.device)
    idx = torch.arange(B, device=objs.device)
    out[idx, n_colors, 0, 0] = 1.0
    out = out * valid.view(-1, 1, 1, 1)
    return out


def centerofmass_batch(objs, valid):
    """Center of mass per object.
    Returns: (MAX_OBJ, 2) tensor of (row, col)
    """
    H, W = objs.shape[2], objs.shape[3]
    is_c = content_mask(objs)
    rows = torch.arange(H, device=objs.device).float().view(1, 1, H, 1).expand_as(is_c)
    cols = torch.arange(W, device=objs.device).float().view(1, 1, 1, W).expand_as(is_c)

    total = is_c.sum(dim=(2, 3)).clamp(min=1)
    r = (is_c * rows).sum(dim=(2, 3)) / total
    c = (is_c * cols).sum(dim=(2, 3)) / total

    return torch.cat([r, c], dim=1)


# ============================================================================
# Batch filter operations
# ============================================================================

def colorfilter_batch(objs, valid, color):
    """Filter objects: keep only those containing the given color.
    Returns: updated valid mask (MAX_OBJ,)
    """
    is_c = content_mask(objs)
    pixel_count = is_c.sum(dim=(1, 2, 3))
    has_color = (objs[:, color, :, :] * is_c.squeeze(1)).sum(dim=(1, 2))
    matches = ((has_color > 0) & (pixel_count > 0)).float()
    return valid * matches


def sizefilter_batch(objs, valid, n):
    """Filter objects: keep only those with exactly n pixels.
    Returns: updated valid mask (MAX_OBJ,)
    """
    is_c = content_mask(objs)
    counts = is_c.sum(dim=(1, 2, 3))
    match = (counts == n).float()
    return valid * match


def mfilter_merge(objs, valid, pred_mask):
    """Filter objects by predicate and merge survivors.
    Returns: (1, 10, H, W) merged one-hot grid
    """
    mask = valid * pred_mask
    filtered = objs * mask.view(-1, 1, 1, 1)
    merged = filtered.max(dim=0, keepdim=True).values
    return merged


# ============================================================================
# Set operations — fully vectorized
# ============================================================================

def difference_batch(objs_a, valid_a, objs_b, valid_b):
    """Set difference: objects in A whose pixel pattern is NOT in B.
    Returns: updated valid_a mask (MAX_OBJ,)
    """
    is_a = content_mask(objs_a).view(objs_a.shape[0], -1)
    is_b = content_mask(objs_b).view(objs_b.shape[0], -1)

    eq_matrix = (is_a.unsqueeze(1) == is_b.unsqueeze(0)).float()
    shape_match = eq_matrix.prod(dim=2)
    any_match = (shape_match * valid_b.unsqueeze(0)).max(dim=1).values
    any_match = torch.clamp(any_match, 0, 1)

    return valid_a * (1.0 - any_match)


def merge_batch(objs, valid):
    """Merge all valid objects into a single grid.
    Returns: (1, 10, H, W)
    """
    filtered = objs * valid.view(-1, 1, 1, 1)
    merged = filtered.max(dim=0, keepdim=True).values
    return merged


def combine_batch(valid_a, valid_b):
    """Union of two valid masks.
    Returns: (MAX_OBJ,) float
    """
    return torch.clamp(valid_a + valid_b, 0, 1)


# ============================================================================
# Dispatch
# ============================================================================

BATCH_PRED_DISPATCH = {
    "bordering": bordering_batch,
    "hline": hline_batch,
    "vline": vline_batch,
    "color": color_batch,
    "size": size_batch,
    "numcolors": numcolors_batch,
    "equality": equality_scalar,
    "centerofmass": centerofmass_batch,
}

BATCH_FN_DISPATCH = {
    "centerofmass": centerofmass_batch,
}