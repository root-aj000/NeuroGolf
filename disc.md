

## Key insight: you're not starting from zero

These 400 tasks with `train`/`test` splits are almost certainly the **ARC-AGI-1 training set** (400 tasks is the exact size of that split), and `arc-gen` is from **ARC-GEN-100K**, a public synthetic-augmentation dataset that procedurally generates more examples *per task* using the same underlying rule with randomized colors/positions/sizes.

This matters a lot:

- **Public solvers already exist** for the original 400 ARC-AGI-1 training tasks — most notably Michael Hodel's `arc-dsl` repo, which contains a hand-written Python solver function for *every one* of the 400 training tasks, using a documented DSL of primitives (rotate, crop, recolor, object extraction, symmetry, flood fill, etc.).
- This turns your problem from **"discover the rule from scratch"** into **"look up the known rule, then translate/verify it."**
- The real engineering work becomes: (1) verify the known solution generalizes to the `arc-gen` examples too, and (2) re-implement the logic (which uses Python sets/loops/objects) as **static-shape ONNX-safe tensor ops**.

So your effective pipeline should be:

```
1. Match task JSON → corresponding known ARC-AGI-1 task ID (by comparing train pairs)
2. Pull the reference DSL solution for that task ID
3. Run it against train+test+arc-gen in Python to confirm it's fully correct
4. Translate that specific logic into a small torch module (your primitive library)
5. Export → validate → done
6. For any task where no reference solution fits / task doesn't match a known ID → fall back to manual/automated discovery
```

This alone probably gets you 300+ tasks "solved" at the logic level, leaving translation as the main work — not puzzle-solving.

---

## For the remainder (no clean match, or reference solution too "loopy" to translate easily)

You need a **recipe-discovery script**. Here's how I'd design it.

### Architecture

```
discover.py
├── load_task(path) -> train/test/arc-gen grid pairs (as numpy arrays)
├── primitive_registry: dict[name -> fit_fn]
│     each fit_fn(pairs) -> Optional[params]   # tries to infer params from train pairs
├── verify(program, all_pairs) -> bool          # exact match check
├── search(pairs) -> best matching program
└── main loop over all task files, log results
```

### Step 1 — Feature/family detectors (fast, cheap, catch easy tasks first)

For each task, compute simple signals across train pairs:
- input shape vs output shape ratio (same / scaled up / scaled down / cropped)
- color palette: same set, subset, remapped, added
- is output == some fixed rigid transform of input (rot90/180/270, flipH/V, transpose)?
- is output a tiled repetition of input?
- is output symmetric completion of input?
- bounding-box crop check

Each detector directly **fits parameters from the pairs** (not blind search):

```python
def try_rigid_transform(pairs):
    for name, fn in RIGID_TRANSFORMS.items():   # rot90, rot180, flipH, ...
        if all(np.array_equal(fn(inp), out) for inp, out in pairs):
            return {"type": name}
    return None

def try_colormap(pairs):
    mapping = {}
    for inp, out in pairs:
        if inp.shape != out.shape: return None
        for a, b in zip(inp.flatten(), out.flatten()):
            if a in mapping and mapping[a] != b: return None
            mapping[a] = b
    return {"type": "colormap", "map": mapping}

def try_crop_bbox(pairs):
    for inp, out in pairs:
        if not np.array_equal(crop_to_bbox(inp), out): return None
    return {"type": "crop_bbox"}
```

Run detectors in priority order (cheapest/most common first); first one that matches **all train pairs** is a candidate.

### Step 2 — Verify candidate against everything, not just train

This is the critical part given you have `arc-gen`:

```python
def verify(program, pairs):
    return all(np.array_equal(apply(program, inp), out) for inp, out in pairs)

candidate = try_rigid_transform(train_pairs) or try_colormap(train_pairs) or ...
if candidate and verify(candidate, train_pairs + test_pairs + arcgen_pairs):
    recipes[task_id] = candidate
else:
    queue_for_manual_review(task_id)
```

Because `arc-gen` gives you 100–300+ extra examples per task, a wrong/overfit hypothesis (e.g. "colormap 3→5" learned from a train set where that happened to be the only mapping) gets **caught immediately** when it fails on arc-gen variety. This is your best defense against false positives — always require full-pair verification, never just train.

### Step 3 — Compositional search for harder cases

If no single primitive matches, try short compositions (depth 2–3) of your primitive library, since many ARC tasks are literally "flip then recolor" or "crop then tile":

```python
from itertools import product

def compositional_search(pairs, primitives, max_depth=2):
    for depth in range(1, max_depth+1):
        for combo in product(primitives, repeat=depth):
            params = fit_composed(combo, pairs)   # try to fit each stage
            if params and verify_composed(combo, params, pairs):
                return combo, params
    return None
```

Keep this depth small (2–3) — combinatorics explode fast, and most ARC tasks aren't deep compositions of simple primitives anyway (that's why grid/object-reasoning tasks are hard — they need bespoke logic).

### Step 4 — Object-level primitives (bigger lift, still automatable)

A meaningful chunk of remaining tasks involve "objects" (4-connected/8-connected same-color regions): select largest, select unique-colored, count objects, move object to corner, etc. Build a small connected-components routine in numpy (fine here, since this is your *offline discovery* script — not the ONNX model itself), then add higher-level detectors:

```python
def try_select_largest_object(pairs): ...
def try_select_unique_color_object(pairs): ...
def try_object_count_to_color(pairs): ...
```

These get translated later using conv-based tricks (fixed kernels, iterative fixed-count convolution for connected components — no `Loop`/`Unique` needed since you unroll a fixed number of iterations).

### Step 5 — Logging & manual queue

```python
results = {"solved": {}, "unsolved": []}

for task_file in all_400_tasks:
    pairs = load_all_pairs(task_file)
    program = detectors_pipeline(pairs) or compositional_search(pairs)
    if program:
        results["solved"][task_id] = program
    else:
        results["unsolved"].append(task_id)

save_json(results, "recipes.json")
print(f"{len(results['solved'])}/400 solved automatically")
```

Whatever lands in `unsolved` is your manual-review list — likely 20–80 tasks depending on how rich your primitive library gets. For those, you look at the grids by eye (visualize input/output pairs), form a hypothesis, write a one-off Python function, and **run it through the same `verify()` against all pairs** before committing it as a custom module.

---

## Practical workflow suggestion

1. **First pass**: try to ID-match tasks against `arc-dsl`'s known 400 solutions (fastest win, near-guaranteed correct logic).
2. **Second pass**: run your detector pipeline on anything unmatched/unmatchable, to auto-classify easy families (geometry, colormap, crop, tile, symmetry).
3. **Third pass**: compositional search for anything still unsolved.
4. **Fourth pass**: manual coding + verification for stragglers, using visualization tools (plot grids side by side).
5. Every recipe — automatic or manual — must pass `verify()` against **train+test+arc-gen** before you spend time translating it to ONNX. This ordering saves you from wasting export/optimization effort on a wrong rule.

---

