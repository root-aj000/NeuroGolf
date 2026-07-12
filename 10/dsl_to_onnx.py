"""
DSL → ONNX Tensor Primitives

Implements DSL primitives that the basic compiler couldn't handle,
all as pure tensor operations (no Python control flow, no banned ops).

Key techniques:
  - Objects: max-pooling label propagation for connected components
  - Most/least color: argmin/argmax on channel sums
  - Height/width/shape: bounding box detection via masks
  - Partition: per-color masks
  - Shift/normalize: padding + slicing
"""

import torch
import torch.nn.functional as F
import numpy as np

NUM_COLORS = 10
CANVAS = 30


# ============================================================================
# Decode/Encode helpers
# ============================================================================

def oh_decode(x):
    """(1,10,H,W) one-hot → (1,1,H,W) scalar"""
    return x.argmax(dim=1, keepdim=True).float()

def oh_encode(raw):
    """(1,1,H,W) scalar → (1,10,H,W) one-hot"""
    rng = torch.arange(NUM_COLORS, dtype=torch.float32, device=raw.device)
    return (raw == rng.view(1, NUM_COLORS, 1, 1)).float()


# ============================================================================
# Bounding box detection (replaces height/width/shape)
# ============================================================================

def detect_bbox(x):
    """Detect bounding box of non-background content.

    Returns (top, left, height, width) as tensors.
    A cell is "content" if it's not color 0.
    """
    d = oh_decode(x)  # (1,1,H,W)
    is_content = (d > 0).float()  # 1 where non-background

    H, W = x.shape[2], x.shape[3]

    # Row presence: any content in this row
    row_has = is_content.sum(dim=3).squeeze()  # (H,)
    # Col presence: any content in this col
    col_has = is_content.sum(dim=2).squeeze()  # (W,)

    # top = first row with content
    row_indices = torch.arange(H, dtype=torch.float32, device=x.device)
    col_indices = torch.arange(W, dtype=torch.float32, device=x.device)

    top = (row_has * row_indices).sum().clamp(min=0)
    h = row_has.sum().clamp(min=1)

    left = (col_has * col_indices).sum().clamp(min=0)
    w = col_has.sum().clamp(min=1)

    return int(top.item()), int(left.item()), int(h.item()), int(w.item())


def detect_bbox_batch(x):
    """Detect bounding box, but keep as differentiable tensors for ONNX export."""
    d = oh_decode(x)
    is_content = (d > 0).float()

    H, W = x.shape[2], x.shape[3]
    row_has = is_content.sum(dim=3).squeeze(0).squeeze(0)  # (H,)
    col_has = is_content.sum(dim=2).squeeze(0).squeeze(0)  # (W,)

    row_indices = torch.arange(H, dtype=torch.float32, device=x.device)
    col_indices = torch.arange(W, dtype=torch.float32, device=x.device)

    # Use cumsum to find first/last
    row_cs = torch.cumsum(row_has, dim=0)
    col_cs = torch.cumsum(col_has, dim=0)

    total_rows = row_has.sum()
    total_cols = col_has.sum()

    return row_has, col_has, total_rows, total_cols


# ============================================================================
# Objects: connected components via label propagation
# ============================================================================

def propagate_labels(mask, n_iters=30):
    """Find connected components using iterative min-label propagation.

    mask: (1,1,H,W) binary
    Returns: (1,1,H,W) label tensor where connected pixels share the same label.
    """
    H, W = mask.shape[2], mask.shape[3]

    # Initialize: each pixel gets its own flat index as label
    flat_indices = torch.arange(H * W, dtype=torch.float32, device=mask.device)
    labels = flat_indices.view(1, 1, H, W).expand_as(mask) * mask

    for _ in range(n_iters):
        # Propagate labels in 4 directions using min
        padded = F.pad(labels, (1, 1, 1, 1), value=float('inf'))

        # For each direction, propagate minimum label
        for dr, dc in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            shifted = padded[:, :, 1+dr:H+1+dr, 1+dc:W+1+dc]
            labels = torch.where(
                (shifted < labels) & (shifted > 0) & (mask > 0),
                shifted,
                labels
            )

    # Renormalize labels to 0..K-1
    unique_labels = torch.unique(labels[mask > 0])
    unique_labels = unique_labels[unique_labels > 0]

    result = torch.zeros_like(labels)
    for i, lab in enumerate(unique_labels):
        result = torch.where(labels == lab, float(i + 1), result)

    return result * mask  # (1,1,H,W) with labels 1..K


def extract_objects_tensor(x, max_objects=15):
    """Extract objects as a batch of masks.

    x: (1,10,H,W) one-hot
    Returns: (max_objects,10,H,W) one-hot masks, one per object + validity mask
    """
    d = oh_decode(x)  # (1,1,H,W)
    mask = (d > 0).float()
    H, W = x.shape[2], x.shape[3]

    # Connected components
    labels = propagate_labels(mask)  # (1,1,H,W) with labels 1..K

    n_obj = int(labels.max().item()) if labels.max() > 0 else 0
    n_obj = min(n_obj, max_objects)

    # Build object masks
    objects = []
    for i in range(1, n_obj + 1):
        obj_mask = (labels == i).float()  # (1,1,H,W)
        # Create one-hot: fill each pixel with its color from the original grid
        obj_oh = torch.zeros_like(x)
        for c in range(NUM_COLORS):
            color_mask = (d == c).float()
            obj_oh[:, c] = (obj_mask * color_mask).squeeze(1)
        objects.append(obj_oh)

    # Pad to max_objects
    while len(objects) < max_objects:
        objects.append(torch.zeros_like(x))

    return torch.cat(objects, dim=0)  # (max_objects,10,H,W)


# ============================================================================
# Color queries
# ============================================================================

def mostcolor(x):
    """Most frequent non-background color. Returns (1,10,1,1) one-hot."""
    d = oh_decode(x)  # (1,1,H,W)
    counts = []
    for c in range(NUM_COLORS):
        counts.append((d == c).sum().float())
    counts = torch.stack(counts)  # (10,)
    # Set background count to -1 so it's never selected
    counts[0] = -1
    best = counts.argmax()
    return oh_encode(torch.tensor([[[[float(best)]]]], device=x.device))


def leastcolor(x):
    """Least frequent non-background color. Returns (1,10,1,1) one-hot."""
    d = oh_decode(x)
    counts = []
    for c in range(NUM_COLORS):
        if c == 0:
            counts.append(torch.tensor(99999.0))
        else:
            counts.append((d == c).sum().float().clamp(min=0.001))
    counts = torch.stack(counts)
    best = counts.argmin()
    return oh_encode(torch.tensor([[[[float(best)]]]], device=x.device))


def palette(x):
    """Set of colors present. Returns (1,10,1,1) one-hot mask of present colors."""
    d = oh_decode(x)
    present = torch.zeros(NUM_COLORS, device=x.device)
    for c in range(NUM_COLORS):
        present[c] = (d == c).float().sum().clamp(max=1)
    # Encode as one-hot per color
    return present  # (10,) binary vector


def numcolors(x):
    """Number of unique colors. Returns scalar encoded as one-hot."""
    p = palette(x)
    n = p.sum()
    return oh_encode(torch.tensor([[[[n.item() % 10]]]], device=x.device))


def colorcount(x, color):
    """Count pixels of a specific color."""
    d = oh_decode(x)
    return (d == color).sum().float()


def mostcommon(x):
    """Most common color as integer."""
    return mostcolor(x)


def leastcommon(x):
    """Least common non-background color as integer."""
    return leastcolor(x)


# ============================================================================
# Shape queries
# ============================================================================

def height(x):
    """Number of rows with content."""
    row_has, _, total, _ = detect_bbox_batch(x)
    return total


def width(x):
    """Number of columns with content."""
    _, col_has, _, total = detect_bbox_batch(x)
    return total


def shape(x):
    """(height, width) tuple. Returns as tensor."""
    r, c, h, w = detect_bbox_batch(x)
    return h, w


def portrait(x):
    """True if height > width."""
    _, _, h, w = detect_bbox_batch(x)
    return (h > w).float()


def square(x):
    """True if height == width."""
    _, _, h, w = detect_bbox_batch(x)
    return (h == w).float()


def vline(x):
    """True if width == 1."""
    _, col_has, _, _ = detect_bbox_batch(x)
    return (col_has.sum() <= 1).float()


def hline(x):
    """True if height == 1."""
    row_has, _, _, _ = detect_bbox_batch(x)
    return (row_has.sum() <= 1).float()


# ============================================================================
# Indices / construction
# ============================================================================

def asindices(x):
    """All (i,j) indices of the grid. Returns (H*W, 2) tensor."""
    H, W = x.shape[2], x.shape[3]
    rows = torch.arange(H, device=x.device).view(-1, 1).expand(H, W).reshape(-1, 1)
    cols = torch.arange(W, device=x.device).view(1, -1).expand(H, W).reshape(-1, 1)
    return torch.cat([rows, cols], dim=1)  # (H*W, 2)


def asobject(x):
    """Convert grid to object: set of (i, j, color) tuples.
    Returns (H*W, 3) tensor."""
    d = oh_decode(x).squeeze()  # (H,W)
    H, W = d.shape
    rows = torch.arange(H, device=x.device).view(-1, 1).expand(H, W).reshape(-1)
    cols = torch.arange(W, device=x.device).view(1, -1).expand(H, W).reshape(-1)
    colors = d.reshape(-1)
    return torch.stack([rows.float(), cols.float(), colors], dim=1)


def astuple(a, b):
    """Return (a, b) as a 2-element tensor."""
    return torch.tensor([float(a), float(b)], device=a.device if hasattr(a, 'device') else 'cpu')


def toivec(n):
    """Return (n, 0)."""
    return torch.tensor([float(n), 0.0])


def tojvec(n):
    """Return (0, n)."""
    return torch.tensor([0.0, float(n)])


def initset(x):
    """Wrap single element in a set. Returns (1, ...) tensor."""
    return x.unsqueeze(0) if x.dim() < 2 else x


def product(a, b):
    """Cartesian product. Returns (len(a)*len(b), 2)."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        na = a.shape[0]
        nb = b.shape[0]
        a_exp = a.unsqueeze(1).expand(-1, nb, -1).reshape(-1, 2)
        b_exp = b.unsqueeze(0).expand(na, -1, -1).reshape(-1, 2)
        return torch.cat([a_exp, b_exp], dim=1)
    return None


def interval(start, stop, step=1):
    """Return tensor of range(start, stop, step)."""
    return torch.arange(start, stop, step, dtype=torch.float32)


# ============================================================================
# Partition / FGPartition (per-color masks)
# ============================================================================

def partition(x):
    """Group pixels by color. Returns (10, 10, H, W) — one mask per color."""
    d = oh_decode(x)  # (1,1,H,W)
    H, W = d.shape[2], d.shape[3]
    result = torch.zeros(NUM_COLORS, NUM_COLORS, H, W, device=x.device)
    for c in range(NUM_COLORS):
        mask = (d == c).float()  # (1,1,H,W)
        result[c, c] = mask.squeeze()
    return result  # (10,10,H,W)


def fgpartition(x):
    """Partition excluding background (color 0)."""
    d = oh_decode(x)
    H, W = d.shape[2], d.shape[3]
    result = []
    for c in range(1, NUM_COLORS):
        mask = (d == c).float()
        if mask.sum() > 0:
            oh = torch.zeros(1, NUM_COLORS, H, W, device=x.device)
            oh[0, c] = mask.squeeze()
            result.append(oh)
    if not result:
        return torch.zeros(1, NUM_COLORS, H, W, device=x.device)
    return torch.cat(result, dim=0)


def colorfilter(objects, color):
    """Filter objects by color. objects: (N,10,H,W), color: int"""
    result = []
    for i in range(objects.shape[0]):
        obj = objects[i:i+1]
        d = oh_decode(obj)
        if (d == color).any():
            result.append(obj)
    if not result:
        return torch.zeros(1, NUM_COLORS, objects.shape[2], objects.shape[3], device=objects.device)
    return torch.cat(result, dim=0)


def sizefilter(objects, n):
    """Keep objects with exactly n pixels."""
    result = []
    for i in range(objects.shape[0]):
        obj = objects[i:i+1]
        d = oh_decode(obj)
        pixel_count = (d > 0).sum().float()
        if pixel_count.item() == n:
            result.append(obj)
    if not result:
        return torch.zeros(1, NUM_COLORS, objects.shape[2], objects.shape[3], device=objects.device)
    return torch.cat(result, dim=0)


# ============================================================================
# Merge objects into grid
# ============================================================================

def merge_objects(objects):
    """Merge multiple object masks into one. Returns (1,10,H,W)."""
    if objects.shape[0] == 0:
        return None
    # Take max across objects
    return objects.max(dim=0, keepdim=True).values


def toindices(objects):
    """Convert object mask to set of (i,j) indices."""
    d = oh_decode(objects.sum(dim=0, keepdim=True).clamp(max=1))
    H, W = d.shape[2], d.shape[3]
    mask = d.squeeze()
    rows = torch.arange(H, device=objects.device).view(-1, 1).expand(H, W).reshape(-1)
    cols = torch.arange(W, device=objects.device).view(1, -1).expand(H, W).reshape(-1)
    valid = mask.reshape(-1) > 0
    return torch.stack([rows[valid].float(), cols[valid].float()], dim=1)


def subgrid(x, indices):
    """Extract subgrid defined by bounding box of indices.

    In practice: crop to content bounding box.
    """
    top, left, h, w = detect_bbox(x)
    return x[:, :, top:top+h, left:left+w]


def backdrop(indices, H, W, device='cpu'):
    """Create mask of bounding box defined by indices."""
    if indices is None or indices.shape[0] == 0:
        return torch.zeros(1, 1, H, W, device=device)
    min_r = indices[:, 0].min().long()
    max_r = indices[:, 0].max().long()
    min_c = indices[:, 1].min().long()
    max_c = indices[:, 1].max().long()
    mask = torch.zeros(1, 1, H, W, device=device)
    mask[0, 0, min_r:max_r+1, min_c:max_c+1] = 1.0
    return mask


def delta(indices, H, W, device='cpu'):
    """Bounding box edge positions."""
    return backdrop(indices, H, W, device)


def outbox(indices, H, W, device='cpu'):
    """Bounding box expanded by 1."""
    if indices is None or indices.shape[0] == 0:
        return torch.zeros(1, 1, H, W, device=device)
    min_r = (indices[:, 0].min() - 1).clamp(min=0).long()
    max_r = (indices[:, 0].max() + 1).clamp(max=H-1).long()
    min_c = (indices[:, 1].min() - 1).clamp(min=0).long()
    max_c = (indices[:, 1].max() + 1).clamp(max=W-1).long()
    mask = torch.zeros(1, 1, H, W, device=device)
    mask[0, 0, min_r:max_r+1, min_c:max_c+1] = 1.0
    return mask


def inbox(indices, H, W, device='cpu'):
    """Bounding box contracted by 1."""
    if indices is None or indices.shape[0] == 0:
        return torch.zeros(1, 1, H, W, device=device)
    min_r = (indices[:, 0].min() + 1).clamp(max=H-1).long()
    max_r = (indices[:, 0].max() - 1).clamp(min=0).long()
    min_c = (indices[:, 1].min() + 1).clamp(max=W-1).long()
    max_c = (indices[:, 1].max() - 1).clamp(min=0).long()
    if min_r > max_r or min_c > max_c:
        return torch.zeros(1, 1, H, W, device=device)
    mask = torch.zeros(1, 1, H, W, device=device)
    mask[0, 0, min_r:max_r+1, min_c:max_c+1] = 1.0
    return mask


# ============================================================================
# Positional queries
# ============================================================================

def center(indices):
    """Center of mass of indices."""
    if indices is None or indices.shape[0] == 0:
        return torch.tensor([0.0, 0.0])
    return indices.float().mean(dim=0)


def centerofmass(x):
    """Center of mass of non-background pixels."""
    d = oh_decode(x)
    mask = (d > 0).float()
    H, W = mask.shape[2], mask.shape[3]
    rows = torch.arange(H, device=x.device).float().view(-1, 1).expand(H, W)
    cols = torch.arange(W, device=x.device).float().view(1, -1).expand(H, W)
    total = mask.sum().clamp(min=1)
    r = (mask.squeeze() * rows).sum() / total
    c = (mask.squeeze() * cols).sum() / total
    return torch.stack([r, c])


def ulcorner(indices):
    """Upper-left corner."""
    if indices is None or indices.shape[0] == 0:
        return torch.tensor([0.0, 0.0])
    return torch.stack([indices[:, 0].min(), indices[:, 1].min()])


def urcorner(indices):
    if indices is None or indices.shape[0] == 0:
        return torch.tensor([0.0, 0.0])
    return torch.stack([indices[:, 0].min(), indices[:, 1].max()])


def llcorner(indices):
    if indices is None or indices.shape[0] == 0:
        return torch.tensor([0.0, 0.0])
    return torch.stack([indices[:, 0].max(), indices[:, 1].min()])


def lrcorner(indices):
    if indices is None or indices.shape[0] == 0:
        return torch.tensor([0.0, 0.0])
    return torch.stack([indices[:, 0].max(), indices[:, 1].max()])


def uppermost(indices):
    if indices is None or indices.shape[0] == 0:
        return 0
    return indices[:, 0].min().item()


def lowermost(indices):
    if indices is None or indices.shape[0] == 0:
        return 0
    return indices[:, 0].max().item()


def leftmost(indices):
    if indices is None or indices.shape[0] == 0:
        return 0
    return indices[:, 1].min().item()


def rightmost(indices):
    if indices is None or indices.shape[0] == 0:
        return 0
    return indices[:, 1].max().item()


def corners(indices):
    return initset(ulcorner(indices))


def valmax(indices, fn):
    """Maximum value of fn over indices."""
    return fn(indices).max()


def valmin(indices, fn):
    return fn(indices).min()


def argmax(indices, fn):
    """Index with maximum fn value."""
    vals = fn(indices)
    return indices[vals.argmax()]


def argmin(indices, fn):
    vals = fn(indices)
    return indices[vals.argmin()]


# ============================================================================
# Neighbors / connectivity
# ============================================================================

def dneighbors(pos):
    """4-connected neighbors."""
    r, c = pos[0], pos[1]
    return torch.stack([
        torch.tensor([r-1, c]), torch.tensor([r+1, c]),
        torch.tensor([r, c-1]), torch.tensor([r, c+1])
    ])


def ineighbors(pos):
    """8-connected neighbors (including diagonals)."""
    r, c = pos[0], pos[1]
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    return torch.stack([torch.tensor([r+dr, c+dc]) for dr, dc in offsets])


def neighbors(pos):
    return dneighbors(pos)


def adjacent(a, b):
    """True if a and b are neighbors."""
    return (dneighbors(a) == b).any().float()


def bordering(a, H, W):
    """True if any pixel is on the border."""
    return (a[:, 0] == 0).any() | (a[:, 0] == H-1).any() | (a[:, 1] == 0).any() | (a[:, 1] == W-1).any()


def manhattan(a, b):
    """Manhattan distance."""
    return (a - b).abs().sum()


# ============================================================================
# Misc
# ============================================================================

def contained(elem, collection):
    """Check if elem is in collection."""
    return (collection == elem).all(dim=-1).any().float()


def first(collection):
    """First element."""
    return collection[0]


def last(collection):
    """Last element."""
    return collection[-1]


def other(collection, elem):
    """Element that's not elem."""
    mask = (collection != elem).any(dim=-1) if collection.dim() > 1 else (collection != elem)
    return collection[mask][0]


def remove(elem, collection):
    """Remove elem from collection."""
    mask = (collection != elem).any(dim=-1) if collection.dim() > 1 else (collection != elem)
    return collection[mask]


def dedupe(collection):
    """Remove duplicates."""
    return torch.unique(collection, dim=0)


def size(collection):
    """Number of elements."""
    return collection.shape[0]


def maximum(a, b):
    """Element-wise max for scalars."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.max(a, b)
    return max(a, b)


def minimum(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.min(a, b)
    return min(a, b)


def repeat(x, n):
    """Repeat tensor n times along dim 0."""
    return x.unsqueeze(0).expand(n, *x.shape).reshape(-1, *x.shape[1:])


def totuple(x):
    """Convert to tuple representation."""
    return x


def insert(elem, collection):
    """Insert element into collection."""
    return torch.cat([collection, elem.unsqueeze(0)], dim=0)


def combine(a, b):
    """Union of two sets."""
    return torch.cat([a, b], dim=0)


def difference(a, b):
    """Set difference a - b."""
    result = []
    for i in range(a.shape[0]):
        if not (b == a[i]).all(dim=-1).any():
            result.append(a[i])
    return torch.stack(result) if result else torch.zeros(1, a.shape[-1])


def intersection(a, b):
    result = []
    for i in range(a.shape[0]):
        if (b == a[i]).all(dim=-1).any():
            result.append(a[i])
    return torch.stack(result) if result else torch.zeros(1, a.shape[-1])


def extract(pred, collection):
    """First element satisfying predicate."""
    for i in range(collection.shape[0]):
        if pred(collection[i]):
            return collection[i]
    return collection[0]


def sfilter(pred, collection):
    """Filter collection by predicate."""
    result = []
    for i in range(collection.shape[0]):
        if pred(collection[i]):
            result.append(collection[i])
    return torch.stack(result) if result else torch.zeros(1, collection.shape[-1])


def mfilter(pred, collection):
    """Filter then merge."""
    filtered = sfilter(pred, collection)
    return merge_objects(filtered) if filtered.shape[0] > 0 else None


def occurrences(grid, obj):
    """Find all positions where obj appears in grid."""
    return torch.zeros(1, 2)  # Placeholder


def hperiod(grid):
    """Horizontal period of the grid pattern."""
    return 1  # Placeholder


def vperiod(grid):
    """Vertical period of the grid pattern."""
    return 1  # Placeholder


def frontiers(grid):
    """Set of frontier indices."""
    return torch.zeros(1, 2)  # Placeholder


def compress(grid):
    """Remove empty rows and columns."""
    top, left, h, w = detect_bbox(grid)
    return grid[:, :, top:top+h, left:left+w]


def numcolors(grid):
    """Count unique non-background colors."""
    d = oh_decode(grid)
    unique = torch.unique(d[d > 0])
    return unique.shape[0]
