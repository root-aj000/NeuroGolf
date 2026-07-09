The architecture is:

## One shared framework + 400 lightweight "recipes"

**What's shared (write once):**

1. **Primitive library** — a set of small `nn.Module` classes implementing ONNX-safe, static-shape operations: `Rotate90`, `FlipH`, `FlipV`, `ColorMap`, `CropToBBox`, `TileNxM`, `MirrorComplete`, `Overlay`, etc.
2. **A generic "Compose" module** — chains primitives together (e.g., `FlipH → ColorMap → CropToBBox`) based on a parameter list.
3. **One export driver script** — loops over all 400 tasks, instantiates the right module + parameters for each, and calls `torch.onnx.export(...)` to produce `task001.onnx ... task400.onnx`.
4. **One validation/test harness** — loads each task JSON, builds the [1,10,30,30] tensors, runs the exported ONNX via onnxruntime, and checks exact match against train+test+arc-gen, plus checks op whitelist / static shapes / file size.

**What's per-task (small, not a full script):**

A short "recipe" — ideally just a config entry, not a whole program:

```python
TASK_RECIPES = {
    "task001": {"type": "rotate", "angle": 90},
    "task002": {"type": "colormap", "map": {1:2, 2:1}},
    "task003": {"type": "crop_bbox"},
    "task147": {"type": "custom", "module": Task147Module},  # bespoke case
    ...
}
```

For the majority of tasks (rotations, flips, recolors, crops, tiling, symmetry fills), the recipe is just a dict of parameters plugged into your shared primitives — no new code at all.

**For genuinely unique tasks** (weird object-selection logic, unusual compositional rules that don't fit any primitive), you write a **bespoke small module just for that task** — but it still plugs into the same shared export + validation pipeline. You're not writing separate boilerplate (data loading, export code, testing code) 400 times — only the actual novel transformation logic, when needed.

## Why this matters practically

- **Consistency**: every exported ONNX goes through the same shape-checking/op-whitelist/size-checking gate, so you don't accidentally violate constraints in task 217 because you copy-pasted an export snippet wrong.
- **Debuggability**: if task 88 fails, you know it's either (a) a recipe/parameter bug or (b) a primitive bug affecting possibly many other tasks too — much easier to isolate than debugging 400 independent files.
- **Speed**: most tasks will map to 1-3 existing primitives once you've built ~15-20 of them, since ARC-AGI transformations repeat heavily across tasks (many are literally the same rule with different colors/grids).
- **Golf score**: shared primitives can be individually optimized once (e.g., making `ColorMap` as small as possible), and that optimization benefits every task using it.

## Suggested repo layout

```
primitives.py       # shared nn.Module building blocks
recipes.py          # dict: task_id -> recipe/params (or custom module ref)
custom/             # bespoke modules for one-off tasks
  task147.py
export_all.py       # loop -> torch.onnx.export -> task{ID}.onnx
validate_all.py      # loop -> onnxruntime check vs train/test/arc-gen + constraint checks
solve.py            # (optional) semi-automated classifier to suggest a recipe per task
```

