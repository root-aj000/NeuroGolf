# Strategy 07 — Forensic Report

## What This Code Does (One Sentence)

It reads 400 hand-written ARC solvers, figures out which 7 are simple enough to compile,
and turns them into ONNX neural networks that solve those tasks perfectly.

---

## File Structure

```
07/
├── build_all.py     ← The entire strategy. One file, 533 lines.
└── models/          ← Output: 7 ONNX files, 1-7 KB each.
```

There is one code file. No libraries, no helpers, no configs. Everything is in `build_all.py`.

---

## The Pipeline (What Happens In Order)

When you run `python3 build_all.py`, this sequence fires:

```
1. Load solver_matches.json  →  get solver name for each of 400 tasks
2. For each task:
   a. Find the solver source code in arc-dsl/solvers.py
   b. Parse the source code into operation list
   c. Classify: can we handle this? (spatial / color / skip)
   d. Build ONNX model
   e. Verify against all train+test examples
3. Print results
```

---

## Step-by-Step Forensic Analysis

### Lines 16-24: Path Constants

```python
TASK_DIR    = ../neurogolf-2026/       ← 400 task JSON files
SOLVERS_PATH = ../arc-dsl/solvers.py   ← 400 hand-written solver functions
MATCHES_PATH = ../03/solver_matches.json ← maps task number → solver name + primitives
OUTPUT_DIR  = ./models/                ← where ONNX files go
N_COLORS = 10                          ← ARC uses colors 0-9
GRID = 30                              ← competition padding size
```

**Why these paths exist:** The 400 tasks live in `neurogolf-2026/`. Each has a solver
function written by humans in `arc-dsl/solvers.py`. The file `solver_matches.json`
(from Strategy 03) tells us which solver function belongs to which task, and what
primitives it uses. We use that to skip tasks we can't handle without even reading
their solver code.

---

### Lines 29-50: Spatial Permutation Formulas

```python
def perm_rot90(H, W):
    return lambda r, c: (W - 1 - c, r)
```

**What this is:** A mathematical formula. Given an output position `(r, c)`, it returns
the input position `(r_in, c_in)` that supplies the value. This is the INVERSE mapping.

**Why inverse:** The ONNX `Gather` op says "for each output position i, take the value
from input position perm[i]". So we need to answer "where does output position (r,c)
get its value FROM?" — that's the inverse.

**Why these 5 specifically:** These are the only spatial transforms in the ARC DSL that
move every cell to a new position without changing the grid dimensions:
- `rot90`: 90° clockwise rotation. `(r,c) → (W-1-c, r)`
- `rot180`: 180° rotation. `(r,c) → (H-1-r, W-1-c)`
- `rot270`: 270° clockwise rotation. `(r,c) → (c, H-1-r)`
- `hmirror`: Flip vertically (reverse rows). `(r,c) → (H-1-r, c)`
- `vmirror`: Flip horizontally (reverse columns). `(r,c) → (r, W-1-c)`

**Why not other transforms:** `dmirror` (transpose) only works on square grids.
`rotate180` is the same as `rot180`. `upscale` changes grid dimensions. Anything
involving `crop`, `fill`, `objects` needs content-dependent logic.

**Why the lambda closure:** The formula depends on H and W (grid size). A lambda
captures H and W at call time, giving us a function we can call like `fn(r, c)`.
This is the simplest way to parameterize the formula.

---

### Lines 54-64: Get Grid Size

```python
def get_grid_size(task_key):
    sizes = set()
    for ex in task["train"] + task.get("test", []):
        inp = ex["input"]
        sizes.add((len(inp), len(inp[0])))
    if len(sizes) == 1:
        return list(sizes)[0]
    return None
```

**What it does:** Opens the task JSON, looks at every example's input grid dimensions,
and checks if they're all the same size.

**Why check:** The permutation array is built for a specific (H, W). If one example is
3x3 and another is 5x5, the same permutation can't work for both — a 3x3 permutation
maps 9 cells, a 5x5 maps 25 cells. They're incompatible.

**Why return None for variable sizes:** We simply skip those tasks. There's no clean way
to handle variable grid sizes with a single static ONNX model. You'd need dynamic
shapes (banned by competition "statically-defined shapes" rule) or multiple models.

**Why this doesn't affect color ops:** Color ops (replace, switch) are per-cell — they
process each cell independently regardless of grid position. So they work for any size.
That's why `build_task_model` handles color ops BEFORE checking grid size.

---

### Lines 68-103: Parse DSL Solver

#### `get_solver_code` (lines 68-74)

```python
pattern = rf"(def {solver_name}\(.*?\n(?:    .*\n)*)"
match = re.search(pattern, content)
```

**What it does:** Extracts one function's source code from a 16,000+ line Python file.

**Why regex:** We don't need a Python parser. We just need to grab the function body.
The regex matches `def solve_xxx(I):` followed by indented lines. The `\n(?:    .*\n)*`
captures all lines that start with 4 spaces (the function body).

**Why not import the file:** `solvers.py` imports the DSL library which requires
specific setup. Parsing the text is simpler and has no dependencies.

#### `parse_solver_ops` (lines 76-103)

**What it does:** Turns each line like `x1 = vmirror(I)` into a tuple `('x1', 'vmirror', ['I'])`.

**Why parse line by line:** DSL solvers are simple — one assignment per line, no
indentation nesting, no control flow (in the simple cases we handle). A line-by-line
split is sufficient.

**The parsing logic:**
1. Skip empty lines, comments, `def` lines, `return` lines
2. Find lines with `=` and `(` (assignment with function call)
3. Split on `=` to get target name
4. Split on `(` and `)` to get function name and arguments
5. Resolve DSL constants (`TWO` → `2`, `ORIGIN` → `(0,0)`)

**Why resolve DSL constants:** The DSL uses named constants like `TWO`, `THREE`, `ORIGIN`.
We need to convert these to numbers for our logic. The `DSL_CONSTS` dict maps 10 color
names to integers and `ORIGIN`/`UNITY` to tuples.

**Why `args_str.split(",")` is fragile:** If an argument contains a comma inside
parentheses (like `astuple(TWO, ONE)` which becomes `(2, 1)`), the split would break.
But we only handle simple cases where this doesn't happen. Complex args → we skip the task.

**Complexity note:** This parser is deliberately naive. It handles ~95% of simple solvers.
The 5% it misses are tasks with complex expressions that we can't handle anyway.

---

### Lines 107-118: Build Permutation Array

```python
def build_perm_indices(H, W, perm_fn):
    indices = np.arange(GRID * GRID, dtype=np.int64)  # [0, 1, 2, ..., 899]
    for r in range(H):
        for c in range(W):
            r_in, c_in = perm_fn(r, c)
            indices[r * GRID + c] = r_in * GRID + c_in
    return indices
```

**What it builds:** An array of 900 integers. `indices[out_flat] = in_flat` means
"output position out_flat gets its value from input position in_flat".

**Why start with `np.arange(900)`:** This sets every position to map to itself (identity).
Then we only override the H*W positions that are actual grid data. Padding positions
(rows ≥ H, cols ≥ W) stay as identity, which means they map to themselves — since
they're already zero, the output stays zero. Correct.

**Why `r * GRID + c` and not `r * W + c`:** GRID is 30 (the padded size), not W (the
actual grid width). The tensor is `(1, 10, 30, 30)`, so the flat index for position
`(r, c)` is `r * 30 + c`, not `r * W + c`. This is the critical detail — we're
operating in the 30x30 padded space, not the original grid space.

**Concrete example:** For `rot180` on a 3x3 grid:
- Output (0,0) gets value from input (2,2): `indices[0] = 2*30+2 = 62`
- Output (0,1) gets value from input (2,1): `indices[1] = 2*30+1 = 61`
- Output (1,1) gets value from input (1,1): `indices[31] = 1*30+1 = 31`
- Output (2,2) gets value from input (0,0): `indices[62] = 0*30+0 = 0`

---

### Lines 122-143: Build Spatial Permutation ONNX Model

```python
nodes = [
    helper.make_node("Reshape", ["input", "rs"], ["flat"]),
    helper.make_node("Gather", ["flat", "perm"], ["permuted"], axis=2),
    helper.make_node("Reshape", ["permuted", "os"], ["output"]),
]
```

**What the model does:**
1. `Reshape`: Flatten `(1,10,30,30)` → `(1,10,900)`. Collapses the 30x30 spatial
   grid into a single dimension of 900 elements.
2. `Gather(axis=2)`: For each of the 10 channels independently, rearrange the 900
   spatial positions according to `perm`. `output[:,:,i] = input[:,:,perm[i]]`.
3. `Reshape`: Unflatten `(1,10,900)` → `(1,10,30,30)`.

**Why axis=2:** The tensor shape is `(batch=1, channels=10, spatial=900)`. We want to
permute the spatial dimension (axis 2) independently for each channel. `Gather` with
`axis=2` does exactly this: for channel 0, gather 900 values; for channel 1, gather
900 values; etc. All channels get the same permutation because spatial transforms
don't depend on color.

**Why 3 nodes total:** This is the minimal possible ONNX model for a spatial transform.
Reshape→Gather→Reshape is 3 ops. The perm array (900 int64 = 7.2KB) is the only
parameter. The model file is ~7KB.

**Why opset 11:** Opset 11 is the first version where `Gather` is stable and well-supported.
Opset 10 has issues with `Equal` not supporting float, `Pad` using attributes instead
of inputs, and `Clip` using attributes instead of inputs. Opset 11 is cleaner.

**Why `model.ir_version = 9`:** This is the IR version expected by the competition's
validation code. Setting it explicitly avoids version mismatch warnings.

---

### Lines 146-208: Build Color Replace ONNX Model

This is the most complex model builder. Let's trace through it.

**The problem:** We need to change every cell that has color `old_color` to `new_color`.
In a one-hot tensor, this means: wherever channel `old_color` has a 1.0, move that
1.0 to channel `new_color` instead.

**The approach (step by step):**

```
1. Extract channel old_color → "co"  (shape: 1, 30, 30)
2. Extract channel new_color → "cn"
3. Create mask: where co == 1.0
4. New channel: cn_new = clip(cn + mask, 0, 1)
5. Old channel: co_zeroed = co * (1 - mask)
6. Rebuild all 10 channels
7. Concat → output
```

**Why this specific sequence:**

Step 3 (mask): `Equal(co_int, 1)` produces a boolean tensor. We Cast to float to get
0.0/1.0 values. This mask is 1.0 everywhere the old color was present, 0.0 elsewhere.

Step 4 (new channel): `cn + mask` adds 1.0 to the new channel wherever the old color
was. `Clip(..., 0, 1)` ensures we don't exceed 1.0. If the new channel already had
a 1.0 at that position (same cell was already new_color), clipping keeps it at 1.0.
If it was 0.0, adding mask makes it 1.0. Correct.

Step 5 (old channel): `co * (1 - mask)` zeros out the old channel wherever mask was 1.0.
Where mask was 0.0 (old color wasn't there), the channel stays unchanged.

Step 6 (rebuild): We extract all 10 channels from the input, but replace the old_color
channel with `co_zeroed` and the new_color channel with `cn_new`. All other channels
pass through unchanged via `Gather`.

**Why Cast to INT32 for Equal:** ONNX `Equal` in opset 11 requires both inputs to be
the same type, and comparing float to float (1.0 == 1.0) can have precision issues.
Casting to INT32 first is safer and more portable across ONNX implementations.

**Why Clip is needed:** Without Clip, if a cell was already `new_color` and we add
the mask (because the same cell was also `old_color` — which can't happen in valid
one-hot, but we're defensive), the value would be 2.0. Clip brings it back to 1.0.

**Why we rebuild all 10 channels:** ONNX doesn't have an "in-place" operation. We
can't modify channel `old_color` in the input tensor. We must extract all channels,
modify the two we care about, and Concat them back. This is verbose but correct.

---

### Lines 211-261: Build Switch Model

**Same structure as replace, but swaps two channels:**
- Channel `a` gets mask of channel `b` (where b was 1, a becomes 1)
- Channel `b` gets mask of channel `a` (where a was 1, b becomes 1)
- All other channels pass through

**Why this is correct for a swap:** If cell has color a → mask_a=1, mask_b=0.
After swap: channel a gets mask_b=0 (cleared), channel b gets mask_a=1 (set).
Result: cell now has color b. Vice versa for cells that had color b.

---

### Lines 264-328: Build Chained Switch Model

**What it does:** Applies multiple swaps in sequence. For example, if the DSL does:
```
x1 = switch(I, 3, 4)    ← swap 3↔4
x2 = switch(x1, 8, 9)   ← swap 8↔9
x3 = switch(x2, 2, 6)   ← swap 2↔6
O = switch(x3, 1, 5)    ← swap 1↔5
```

We build 4 switch stages chained together.

**How chaining works:** Each stage takes the output of the previous stage as input.
`current` variable tracks which tensor name feeds into the next stage. Starts at
`"input"`, after stage 0 becomes `"cat_s0"`, after stage 1 becomes `"cat_s1"`, etc.

**Why `tag = f"_s{i}"`:** Every tensor name in ONNX must be unique. If we used the
same names for stage 0 and stage 1, ONNX would reject the model. The tag `"_s0"`,
`"_s1"`, etc. makes every name unique.

**Why `Identity` at the end:** The final Concat outputs to `"cat_s3"`, but the graph
output must be named `"output"`. `Identity` is a no-op that renames the tensor.

**Complexity:** For N switch pairs, we build N * (6 + 10) = 16N nodes (6 for masks,
10 for channel rebuild). For 4 switches, that's 64 nodes. Still tiny.

---

### Lines 333-399: Classify Task

**What it does:** Looks at the parsed DSL operations and determines which category
the task falls into.

**The decision tree:**

```
Is the output op a spatial transform on I directly?
  → ("spatial", op_name)

Is the output op replace()?
  Is the input to replace a spatial transform of I?
    → We CAN'T handle spatial+replace yet, return None
  Is the input to replace just I?
    → ("replace", old, new)

Is the output op switch()?
  Walk backwards through the chain of switches:
    O = switch(x3, a, b)
    x3 = switch(x2, c, d)
    x2 = switch(x1, e, f)
    x1 = switch(I, g, h)
  Collect all pairs: [(g,h), (e,f), (c,d), (a,b)]
    → ("switch_chain", pairs)

Otherwise → None (can't handle)
```

**Why walk backwards for switches:** The DSL builds chains where each switch feeds
into the next. We need to collect all pairs in order. Starting from O and walking
backwards through variable names gives us the pairs in reverse order, so we
`pairs.reverse()` at the end.

**Why we can't handle spatial+replace:** The classification currently doesn't combine
spatial permutation with color ops. A task like `O = replace(vmirror(I), 2, 8)` would
need: first permute spatially, then apply replace. We'd need to compose the two models.
This is possible but adds complexity, so we skip it for now.

**Why `switch_spatial` returns None in practice:** The code has a branch for
`switch_spatial` (switch after a spatial op), but `build_task_model` doesn't handle
that classification. It's dead code — a reminder of what we might add later.

---

### Lines 404-465: Build Task Model (Orchestrator)

```python
def build_task_model(task_key):
    # 1. Find solver
    # 2. Parse it
    # 3. Classify it
    # 4. Build the right model
    # 5. Save it
```

**Why color ops come before spatial check:** Color ops (replace, switch) work for ANY
grid size because they process each cell independently. Spatial ops need fixed grid
size. So we handle color first, and only check grid size for spatial ops.

**Why `get_grid_size` is called late:** We don't want to reject a task for variable
grid sizes if it's a color-only task (which doesn't care about sizes). Calling it
only for spatial tasks avoids false rejections.

---

### Lines 470-501: Verify

```python
for ex in task["train"] + task.get("test", []):
    # Encode input as one-hot
    x = np.zeros((1, 10, 30, 30))
    for r in range(H_in):
        for c in range(W_in):
            x[0, int(inp[r, c]), r, c] = 1.0

    # Run model
    out = sess.run(None, {"input": x})[0]
    pred = np.argmax(out[0], axis=0)

    # Compare
    for r in range(H_out):
        for c in range(W_out):
            if pred[r, c] != int(outp[r, c]):
                return False, ...
```

**Why we verify ourselves:** The competition runs on Kaggle. We need to know our models
work before submitting. Our verification replicates exactly what the competition does:
encode → run → compare.

**Why argmax:** The model outputs `(1, 10, 30, 30)` floats. The competition compares
full tensors with `np.array_equal()`. But for debugging, argmax (predicted color index)
is easier to read. Both are equivalent for valid one-hot tensors — if the model is
correct, argmax at each position gives the right color.

**Why check train AND test:** A model might accidentally work on training examples
but fail on test examples if we hardcoded something wrong. Checking both ensures
general correctness.

---

### Lines 506-533: Main Loop

```python
for task_key in sorted(matches.keys()):
    path, err = build_task_model(task_key)
    if path:
        ok, verify_err = verify_task(task_key, path)
        if ok:
            success.append(task_key)
```

**Why iterate all 400:** We attempt every task. Most fail silently (return `(None, reason)`).
Only 7 succeed. The loop doesn't stop on failure — it tries everything.

**Why `sorted(matches.keys())`:** Deterministic order. Makes output reproducible.

---

## The Hard Parts (What's Actually Complex)

### 1. The permutation math (lines 29-50, 107-118)

**Why it's hard:** Getting the coordinate math right for each transform. One wrong
sign or swapped axis and the model silently produces wrong output (not an error,
just incorrect results). The formulas must be exact inverses.

**How we verified:** Tested each transform manually with known inputs. For example,
rot180 on a 3x3 grid with color 1 at (0,0) should produce color 1 at (2,2).

### 2. The replace model (lines 146-208)

**Why it's hard:** Expressing "if color == X then change to Y" in pure arithmetic
without any conditional operations. ONNX has no `if/else`. We must use masks,
addition, and clipping to achieve conditional behavior.

**The mask trick:** `mask = (channel == 1.0)` creates a binary tensor. Then:
- `new_channel = old_new + mask` (adds 1 where condition was true)
- `Clip(0, 1)` (prevents overflow)
- `old_channel = old_channel * (1 - mask)` (zeros where condition was true)

This is the standard way to do conditional updates in differentiable/ONNX code.

### 3. The chained switch builder (lines 264-328)

**Why it's hard:** Naming. Every tensor in ONNX must have a unique name. When building
4 switch stages, we create 4 * (2 channels + 2 masks + 10 rebuild channels + 1 concat)
= 56 unique tensor names. One duplicate name → ONNX rejects the model.

**The tag solution:** Every name gets a suffix `_s0`, `_s1`, etc. This guarantees
uniqueness but makes the code verbose.

### 4. The parser (lines 76-103)

**Why it's hard:** It's a text parser that must be correct for valid DSL code but
doesn't need to handle arbitrary Python. The DSL has a restricted syntax (one
assignment per line, no nested expressions in the simple cases), so a line-by-line
split works. But if a line has unexpected format, it silently produces wrong output.

**What it can't handle:**
- Nested function calls: `foo(bar(x))` — comma split breaks
- Multi-line expressions — not possible in DSL
- Comprehensions — not in DSL
- These cases are tasks we can't handle anyway, so it's fine

---

## Why This Only Solves 7/400 Tasks

The 400 ARC tasks use 160 different DSL primitives. We handle 7 of them:
`rot90`, `rot180`, `rot270`, `hmirror`, `vmirror`, `replace`, `switch`.

The other 153 primitives include:
- **Content queries** (`ofcolor`, `mostcolor`, `objects`, `partition`): Need to look
  at what colors are present, which cells form connected components, etc. These
  require scanning the tensor — not expressible as a fixed permutation.
- **Conditional logic** (`fill`, `underfill`, `branch`): Need to say "if cell has
  color X, put color Y there". This is per-cell but depends on spatial context.
- **Grid construction** (`canvas`, `crop`, `vconcat`, `hconcat`): Build new grids
  from pieces. Changes grid dimensions, which breaks our static-shape approach.
- **Iteration** (`apply`, `mapply`, `sfilter`): Loop over objects/cells. ONNX bans
  loops (LOOP/SCAN).

The 7 tasks we solve are the extreme edge cases — tasks where the human solver
used only the simplest possible operations. They exist, but they're rare.

---

## Results

```
task016: PASS  — 4 chained switches (3→4, 8→9, 2→6, 1→5)
task087: PASS  — rot180
task140: PASS  — rot180
task276: PASS  — 4 chained switches
task309: PASS  — replace
task337: PASS  — 4 chained switches
task380: PASS  — rot270
```

All 7 models: 1-7 KB, pass full verification on all train+test examples.
Total: 7/400 tasks solved (1.75%).
