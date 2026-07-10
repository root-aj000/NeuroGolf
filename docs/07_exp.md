# Strategy 07: Compile DSL Solvers to ONNX

## What It Does

Takes the 400 hand-written DSL solvers from `arc-dsl/solvers.py` and compiles
the simplest ones into ONNX models that meet competition constraints.

## What It Handles

Three categories of tasks:

1. **Spatial only** — `rot90`, `rot180`, `rot270`, `hmirror`, `vmirror`
   Applied via permutation: `output[i] = input[perm[i]]`
   Requires fixed grid size across all examples.

2. **Color only** — `replace(a, b)`, `switch(a, b)`, chained switches
   Per-cell channel operations. Works for any grid size.

3. **Nothing else** — tasks using `fill`, `objects`, `crop`, `ofcolor`, etc. are skipped.

## How It Works

### Data flow

```
400 tasks from solver_matches.json
        │
        ▼
    Parse DSL solver → [(target, func, args), ...]
        │
        ▼
    classify_task() → spatial / replace / switch_chain / skip
        │
        ├── spatial → build_perm_model(perm_indices)
        ├── replace → build_replace_model(old, new)
        └── switch  → build_switch_chain_model(pairs)
        │
        ▼
    models/taskXXX.onnx  (1-7 KB each)
        │
        ▼
    verify against all train+test examples
```

### Spatial permutation (3 tasks)

A rotation or mirror moves every cell to a new position. We precompute
a 900-element index array where `perm[out] = in`, then ONNX just does:

```
Reshape(1,10,30,30 → 1,10,900)
Gather(perm, axis=2)
Reshape(1,10,900 → 1,10,30,30)
```

Works for fixed grid sizes only (same H,W across all examples).

### Color replace/switch (4 tasks)

Each cell's color is independent. To swap colors a↔b:
1. Extract channel a and b
2. Create binary masks (where each is 1.0)
3. Channel a gets mask_b, channel b gets mask_a
4. Rebuild all 10 channels, concat

Chained switches apply this repeatedly. Works for any grid size.

## Results

7 out of 400 tasks. All models are 1-7 KB, all pass full verification.

```
task016: PASS (4 chained switches)
task087: PASS (rot180)
task140: PASS (rot180)
task276: PASS (4 chained switches)
task309: PASS (replace)
task337: PASS (4 chained switches)
task380: PASS (rot270)
```

## Why Only 7?

Most ARC tasks use complex primitives (`fill`, `objects`, `ofcolor`, `crop`,
`paint`, etc.) that can't be expressed as simple permutations or channel swaps.
These would need loops, conditionals, or content-dependent logic — none of
which are allowed in competition ONNX (no LOOP/SCAN/NonZero/Unique).

The 7 tasks we handle are the simplest possible: move cells, or swap colors.
Everything else requires more powerful ops than ONNX provides under the
competition constraints.

## Files

```
07/
├── build_all.py    # The entire strategy (~450 lines)
└── models/         # 7 ONNX files, 1-7 KB each
```
