"""
PyTorch tensor primitives for ARC grid operations.

All functions operate on (1, 10, H, W) one-hot float tensors.
"""

import torch
import torch.nn.functional as F

NUM_COLORS = 10
CANVAS = 30


# ============================================================================
# One-hot helpers
# ============================================================================

def oh_decode(x: torch.Tensor) -> torch.Tensor:
    """One-hot (1,10,H,W) → scalar (1,1,H,W) via argmax."""
    return x.argmax(dim=1, keepdim=True).float()

def oh_encode(raw: torch.Tensor) -> torch.Tensor:
    """Scalar (1,1,H,W) → one-hot (1,10,H,W)."""
    rng = torch.arange(NUM_COLORS, dtype=torch.float32, device=raw.device)
    return (raw == rng.view(1, NUM_COLORS, 1, 1)).float()


# ============================================================================
# Output padding/cropping to CANVAS x CANVAS
# ============================================================================

def pad_to_canvas(x: torch.Tensor) -> torch.Tensor:
    """Pad (1,10,H,W) to (1,10,30,30) with background (channel 0 = 1)."""
    B, C, H, W = x.shape
    if H >= CANVAS and W >= CANVAS:
        return x[:, :, :CANVAS, :CANVAS]
    out = torch.zeros(B, C, CANVAS, CANVAS, device=x.device, dtype=x.dtype)
    out[:, 0, :, :] = 1.0  # background
    out[:, :, :H, :W] = x
    return out


# ============================================================================
# Spatial transforms
# ============================================================================

def rot90(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.transpose(x, 2, 3), [3])

def rot180(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, [2, 3])

def rot270(x: torch.Tensor) -> torch.Tensor:
    return torch.transpose(torch.flip(x, [3]), 2, 3)

def hmirror(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, [2])

def vmirror(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, [3])

def dmirror(x: torch.Tensor) -> torch.Tensor:
    return torch.transpose(x, 2, 3)

def cmirror(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.transpose(torch.flip(x, [2]), 2, 3), [2])


# ============================================================================
# Cropping (output can be any size — compiler pads to 30x30)
# ============================================================================

def crop(x: torch.Tensor, top: int, left: int, h: int, w: int) -> torch.Tensor:
    return x[:, :, top:top+h, left:left+w]

def trim(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, 1:-1, 1:-1]

def tophalf(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, :x.shape[2]//2, :]

def bottomhalf(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, x.shape[2]//2:, :]

def lefthalf(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, :, :x.shape[3]//2]

def righthalf(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, :, x.shape[3]//2:]


# ============================================================================
# Concatenation
# ============================================================================

def hconcat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat([a, b], dim=3)

def vconcat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat([a, b], dim=2)


# ============================================================================
# Upscaling (pixel-repeat)
# ============================================================================

def hupscale(x: torch.Tensor, factor: int) -> torch.Tensor:
    r = x.unsqueeze(4).repeat(1, 1, 1, 1, factor)
    B, C, H, W = x.shape
    return r.reshape(B, C, H, W * factor)

def vupscale(x: torch.Tensor, factor: int) -> torch.Tensor:
    r = x.unsqueeze(3).repeat(1, 1, 1, factor, 1)
    B, C, H, W = x.shape
    return r.reshape(B, C, H * factor, W)

def upscale(x: torch.Tensor, factor: int) -> torch.Tensor:
    r = x.unsqueeze(3).unsqueeze(5).repeat(1, 1, 1, factor, 1, factor)
    B, C, H, W = x.shape
    return r.reshape(B, C, H * factor, W * factor)

def downscale(x: torch.Tensor, factor: int) -> torch.Tensor:
    return x[:, :, ::factor, ::factor]


# ============================================================================
# Value operations (decode → op → encode)
# ============================================================================

def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return oh_encode((oh_decode(a) + oh_decode(b)) % 10)

def subtract(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return oh_encode((oh_decode(a) - oh_decode(b)) % 10)

def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return oh_encode((oh_decode(a) * oh_decode(b)) % 10)

def increment(x: torch.Tensor, delta: int = 1) -> torch.Tensor:
    return oh_encode((oh_decode(x) + delta) % 10)

def decrement(x: torch.Tensor, delta: int = 1) -> torch.Tensor:
    return oh_encode((oh_decode(x) - delta) % 10)

def double_val(x: torch.Tensor) -> torch.Tensor:
    return oh_encode((oh_decode(x) * 2) % 10)

def minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return oh_encode(torch.min(oh_decode(a), oh_decode(b)))

def maximum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return oh_encode(torch.max(oh_decode(a), oh_decode(b)))

def crement(x: torch.Tensor, delta: int = 1) -> torch.Tensor:
    d = oh_decode(x)
    result = torch.where(d > 0, d + delta, torch.where(d < 0, d - delta, d))
    return oh_encode(result % 10)

def invert(x: torch.Tensor) -> torch.Tensor:
    return oh_encode((-oh_decode(x)) % 10)

def sign(x: torch.Tensor) -> torch.Tensor:
    return oh_encode(torch.clamp(oh_decode(x), -1, 1))

def halve(x: torch.Tensor) -> torch.Tensor:
    return oh_encode((oh_decode(x) // 2) % 10)


# ============================================================================
# Comparison
# ============================================================================

def equality(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return oh_encode((oh_decode(a) == oh_decode(b)).float())

def greater(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return oh_encode((oh_decode(a) > oh_decode(b)).float())

def less(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return oh_encode((oh_decode(a) < oh_decode(b)).float())

def even(x: torch.Tensor) -> torch.Tensor:
    return oh_encode((oh_decode(x).long() % 2 == 0).float())

def positive(x: torch.Tensor) -> torch.Tensor:
    return oh_encode((oh_decode(x) > 0).float())

def flip_bool(x: torch.Tensor) -> torch.Tensor:
    return oh_encode(1.0 - oh_decode(x))

def both(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return oh_encode(oh_decode(a) * oh_decode(b))

def either(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return oh_encode(torch.clamp(oh_decode(a) + oh_decode(b), 0, 1))


# ============================================================================
# Color operations
# ============================================================================

def replace_colors(x: torch.Tensor, old_color: int, new_color: int) -> torch.Tensor:
    d = oh_decode(x)
    return oh_encode(torch.where(d == old_color, float(new_color), d))

def switch_colors(x: torch.Tensor, a: int, b: int) -> torch.Tensor:
    d = oh_decode(x)
    return oh_encode(torch.where(d == a, float(b), torch.where(d == b, float(a), d)))

def cellwise(a: torch.Tensor, b: torch.Tensor, fallback: int = 0) -> torch.Tensor:
    """Where a==b keep value, else use fallback."""
    da, db = oh_decode(a), oh_decode(b)
    match = (da == db).float()
    result = torch.where(match == 1, da, torch.full_like(da, float(fallback)))
    return oh_encode(result)

def merge_grids(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise max on decoded values."""
    return oh_encode(torch.max(oh_decode(a), oh_decode(b)))


# ============================================================================
# ofcolor: returns one-hot mask (channel 1 = matching pixels)
# ============================================================================

def ofcolor(x: torch.Tensor, color: int) -> torch.Tensor:
    H, W = x.shape[2], x.shape[3]
    d = oh_decode(x)
    match = (d == color).float()
    out = torch.zeros(1, NUM_COLORS, H, W, device=x.device, dtype=x.dtype)
    out[:, 0] = 1.0 - match.squeeze(1)
    out[:, 1] = match.squeeze(1)
    return out


# ============================================================================
# fill / underfill / cover
# ============================================================================

def fill_grid(x: torch.Tensor, color: int, mask: torch.Tensor) -> torch.Tensor:
    """Fill positions where mask channel 1 > 0 with given color.

    DSL signature: fill(grid, value, patch)
    mask is a one-hot tensor from ofcolor/mapply.
    """
    m = mask[:, 1:2, :, :]  # (1,1,H,W) foreground channel
    bg = x * (1 - m)
    fg = torch.zeros_like(x)
    fg[:, color] = 1.0
    return bg + fg * m


def underfill_grid(x: torch.Tensor, color: int, mask: torch.Tensor) -> torch.Tensor:
    """Fill only where current pixel is background (color 0)."""
    m = mask[:, 1:2, :, :]
    d = oh_decode(x)
    is_bg = (d == 0).float()
    fill_mask = m * is_bg
    bg = x * (1 - fill_mask)
    fg = torch.zeros_like(x)
    fg[:, color] = 1.0
    return bg + fg * fill_mask


def cover_grid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Erase positions where mask channel 1 > 0 (set to background color 0)."""
    m = mask[:, 1:2, :, :]
    return x * (1 - m) + torch.zeros_like(x)[:, 0:1, :, :] * m


# ============================================================================
# paint / underpaint
# ============================================================================

def paint_grid(x: torch.Tensor, obj_mask: torch.Tensor) -> torch.Tensor:
    """Paint obj_mask onto x. obj_mask is one-hot with colored pixels."""
    # obj_mask has non-zero channels where the object has color
    # We need to overwrite those positions
    d_obj = oh_decode(obj_mask)
    is_obj = (d_obj > 0).float()
    bg = x * (1 - is_obj)
    return bg + obj_mask * is_obj


def underpaint_grid(x: torch.Tensor, obj_mask: torch.Tensor) -> torch.Tensor:
    """Paint only where current pixel is background."""
    d = oh_decode(x)
    is_bg = (d == 0).float()
    d_obj = oh_decode(obj_mask)
    is_obj = (d_obj > 0).float()
    paint_mask = is_bg * is_obj
    bg = x * (1 - paint_mask)
    return bg + obj_mask * paint_mask


# ============================================================================
# Canvas
# ============================================================================

def canvas_grid(color: int, H: int, W: int, device='cpu') -> torch.Tensor:
    out = torch.zeros(1, NUM_COLORS, H, W, device=device)
    out[:, color] = 1.0
    return out


# ============================================================================
# Connected components (objects)
# ============================================================================

def _flood_fill(mask: torch.Tensor, visited: torch.Tensor, r: int, c: int,
                H: int, W: int, component: list):
    """BFS flood fill on a binary mask (1,H,W)."""
    if r < 0 or r >= H or c < 0 or c >= W:
        return
    if visited[0, r, c] == 1:
        return
    if mask[0, r, c] == 0:
        return
    visited[0, r, c] = 1
    component.append((r, c))
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        _flood_fill(mask, visited, r+dr, c+dc, H, W, component)


def connected_components(mask: torch.Tensor) -> list:
    """Find connected components in a binary mask (1,1,H,W).

    Returns list of masks, each (1,1,H,W).
    """
    H, W = mask.shape[2], mask.shape[3]
    visited = torch.zeros(1, H, W, dtype=torch.float32, device=mask.device)
    components = []

    for r in range(H):
        for c in range(W):
            if mask[0, 0, r, c] == 1 and visited[0, r, c] == 0:
                comp = []
                _flood_fill(mask, visited, r, c, H, W, comp)
                if comp:
                    m = torch.zeros(1, 1, H, W, device=mask.device, dtype=mask.dtype)
                    for pr, pc in comp:
                        m[0, 0, pr, pc] = 1.0
                    components.append(m)

    return components


def objects_grid(x: torch.Tensor, univalued: bool = True,
                diagonal: bool = False, without_bg: bool = True) -> list:
    """Extract connected objects from grid.

    Returns list of one-hot masks, each (1,10,H,W) with the object's color.
    """
    d = oh_decode(x)  # (1,1,H,W)
    H, W = x.shape[2], x.shape[3]

    # Get unique colors
    if without_bg:
        colors = [c for c in range(1, NUM_COLORS) if (d == c).any()]
    else:
        colors = list(range(NUM_COLORS))

    objects = []
    for color in colors:
        color_mask = (d == color).float()  # (1,1,H,W)
        components = connected_components(color_mask)
        for comp_mask in components:
            # Build one-hot object mask
            obj = torch.zeros(1, NUM_COLORS, H, W, device=x.device, dtype=x.dtype)
            obj[:, color] = comp_mask.squeeze(1)
            objects.append(obj)

    return objects


# ============================================================================
# Color filter / sfilter / mfilter (operating on object masks)
# ============================================================================

def colorfilter_masks(objects: list, color: int) -> list:
    """Keep objects whose dominant color matches."""
    result = []
    for obj in objects:
        d = oh_decode(obj)
        c = int(d.max().item())
        if c == color:
            result.append(obj)
    return result


def merge_masks(objects: list) -> torch.Tensor:
    """Merge list of object masks into one mask (OR)."""
    if not objects:
        return None
    H, W = objects[0].shape[2], objects[0].shape[3]
    out = torch.zeros(1, 1, H, W, device=objects[0].device, dtype=objects[0].dtype)
    for obj in objects:
        d = oh_decode(obj)
        out = torch.clamp(out + d, 0, 1)
    # Convert to one-hot mask format
    mask = torch.zeros(1, NUM_COLORS, H, W, device=objects[0].device, dtype=objects[0].dtype)
    mask[:, 0] = 1.0 - out.squeeze(1)
    mask[:, 1] = out.squeeze(1)
    return mask


# ============================================================================
# Shift / move / normalize (on decoded coordinates)
# ============================================================================

def shift_mask(mask: torch.Tensor, di: int, dj: int) -> torch.Tensor:
    """Shift a one-hot mask by (di, dj)."""
    H, W = mask.shape[2], mask.shape[3]
    out = torch.zeros_like(mask)
    # Source and dest slices
    src_r = max(0, -di)
    src_c = max(0, -dj)
    dst_r = max(0, di)
    dst_c = max(0, dj)
    h = min(H - abs(di), H - src_r, H - dst_r)
    w = min(W - abs(dj), W - src_c, W - dst_c)
    if h > 0 and w > 0:
        out[:, :, dst_r:dst_r+h, dst_c:dst_c+w] = mask[:, :, src_r:src_r+h, src_c:src_c+w]
    return out


# ============================================================================
# Dispatcher: name → function
# ============================================================================

PRIM_DISPATCH = {
    # Spatial
    "rot90": rot90, "rot180": rot180, "rot270": rot270,
    "hmirror": hmirror, "vmirror": vmirror, "dmirror": dmirror, "cmirror": cmirror,
    # Crop
    "crop": crop, "trim": trim,
    "tophalf": tophalf, "bottomhalf": bottomhalf,
    "lefthalf": lefthalf, "righthalf": righthalf,
    # Concat
    "hconcat": hconcat, "vconcat": vconcat,
    # Upscale
    "hupscale": hupscale, "vupscale": vupscale,
    "upscale": upscale, "downscale": downscale,
    # Value
    "add": add, "subtract": subtract, "multiply": multiply,
    "increment": increment, "decrement": decrement, "double": double_val,
    "minimum": minimum, "maximum": maximum,
    "crement": crement, "invert": invert, "sign": sign, "halve": halve,
    # Comparison
    "equality": equality, "greater": greater, "less": less,
    "even": even, "positive": positive,
    "flip": flip_bool, "both": both, "either": either,
    # Color
    "replace": replace_colors, "switch": switch_colors,
    "cellwise": cellwise, "merge": merge_grids,
    "ofcolor": ofcolor,
    "fill": fill_grid, "underfill": underfill_grid,
    "cover": cover_grid,
    "paint": paint_grid, "underpaint": underpaint_grid,
    "canvas": canvas_grid,
}

# Primitives needing extra int args (beyond tensor inputs)
PRIM_NEEDS_ARGS = {
    "crop": ["top", "left", "h", "w"],
    "hupscale": ["factor"], "vupscale": ["factor"],
    "upscale": ["factor"], "downscale": ["factor"],
    "replace": ["old_color", "new_color"],
    "switch": ["a", "b"],
    "increment": ["delta"], "decrement": ["delta"],
    "fill": ["color"], "underfill": ["color"],
    "ofcolor": ["color"],
    "canvas": ["color", "H", "W"],
    "cellwise": ["fallback"],
}

# Binary ops (two tensor inputs)
BINARY_OPS = {
    "hconcat", "vconcat", "cellwise", "add", "subtract", "multiply",
    "minimum", "maximum", "merge", "equality", "greater", "less",
    "both", "either",
}
