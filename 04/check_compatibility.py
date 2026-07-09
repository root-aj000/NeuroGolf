#!/usr/bin/env python3
"""
Filter neurogolf tasks to those solvable with ONNX-compatible DSL primitives.
Parses solver source code and checks each function call against a whitelist.
"""
import json
import sys
import os
import re
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'arc-dsl'))

import solvers as S
import inspect

# Primitives safe for ONNX (no loops, no ScatterND, no dynamic shapes)
ONNX_SAFE = {
    # Transform
    'rot90', 'rot180', 'rot270',
    'hmirror', 'vmirror', 'dmirror', 'cmirror',
    # Grid ops
    'crop', 'hconcat', 'vconcat',
    'hsplit', 'vsplit',
    'tophalf', 'bottomhalf', 'lefthalf', 'righthalf',
    # Color ops (simple)
    'replace', 'switch', 'fill', 'underfill', 'recolor', 'cover',
    # Spatial (index-based)
    'ofcolor', 'asindices', 'delta', 'box', 'border',
    'hfrontier', 'vfrontier',
    # Geometry
    'ulcorner', 'urcorner', 'llcorner', 'lrcorner',
    'center', 'height', 'width', 'shape', 'size',
    'lowermost', 'uppermost', 'leftmost', 'rightmost',
    # Scaling
    'upscale', 'downscale',
    # Creation
    'canvas',
    # Constants used
    'ORIGIN', 'DOWN', 'UP', 'LEFT', 'RIGHT',
    'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE',
    'TWO_BY_TWO', 'THREE_BY_THREE', 'ONE_BY_ONE',
    # Basic ops
    'identity', 'first', 'last',
    'add', 'subtract', 'multiply', 'divide',
    'astuple', 'totuple',
    'initset', 'insert',
}

# Primitives that are ONNX-problematic
ONNX_UNSAFE = {
    'objects', 'colorfilter', 'sizefilter', 'mfilter', 'sfilter',
    'partition', 'fgpartition',
    'apply', 'mapply', 'compose', 'fork', 'chain',
    'rbind', 'lbind', 'bind',
    'move', 'shift', 'gravitate', 'shoot',
    'paint', 'asobject', 'frontiers',
    'order', 'sorted',
    'neighbors', 'dneighbors', 'ineighbors',
    'product', 'prapply', 'mpapply',
    'merge', 'combine',
    'argmax', 'argmin',
    'extract', 'exist', 'any', 'all',
    'branch', 'condition',
    'power',
    'matcher', 'equality',
    'contains', 'greater', 'less', 'either',
    'interval', 'list',
    'double', 'halve', 'invert',
    'sign', 'negate',
    'decimate',
    'portrait', 'landscape',
    'trim',
}


def get_solver_source(solver_name):
    """Get source code of a solver function."""
    solver_fn = getattr(S, solver_name)
    return inspect.getsource(solver_fn)


def extract_calls(source):
    """Extract all function/constant calls from source."""
    # Find function calls: name(
    func_calls = re.findall(r'\b([a-z_][a-z0-9_]*)\s*\(', source)
    # Find constants: ALL_CAPS
    const_calls = re.findall(r'\b([A-Z][A-Z0-9_]+)\b', source)
    return set(func_calls), set(const_calls)


def is_onnx_compatible(solver_name):
    """Check if a solver uses only ONNX-compatible primitives."""
    source = get_solver_source(solver_name)
    func_calls, const_calls = extract_calls(source)

    # Skip common non-DSL words
    skip_funcs = {'solve', 'return', 'print', 'if', 'else', 'for', 'while',
                  'try', 'except', 'input', 'output', 'x1', 'x2', 'x3', 'x4'}
    func_calls -= skip_funcs

    # Check function calls
    unsafe_funcs = func_calls - ONNX_SAFE
    # Filter out unknown functions (not in ONNX_UNSAFE either)
    known_unsafe = unsafe_funcs & ONNX_UNSAFE

    if known_unsafe:
        return False, known_unsafe, func_calls

    # Check if there are unknown functions
    all_known = ONNX_SAFE | ONNX_UNSAFE
    unknown = func_calls - all_known - skip_funcs

    return True, set(), func_calls


def to_grid(list_2d):
    return tuple(tuple(row) for row in list_2d)


def verify_solver(solver_fn, task_data, max_examples=5):
    """Verify solver on training examples."""
    try:
        for ex in task_data['train'][:max_examples]:
            inp = to_grid(ex['input'])
            expected = to_grid(ex['output'])
            result = solver_fn(inp)
            if result != expected:
                return False
        return True
    except Exception:
        return False


# Load all tasks
tasks = {}
for i in range(1, 401):
    path = f"../neurogolf-2026/task{i:03d}.json"
    with open(path) as f:
        tasks[i] = json.load(f)

# Get all solvers
solver_names = sorted([name for name in dir(S) if name.startswith('solve_')])

# Find task->solver mapping first (reuse from 03/)
task_solver_map = {}
for task_num in range(1, 401):
    task_data = tasks[task_num]
    for solver_name in solver_names:
        solver_fn = getattr(S, solver_name)
        try:
            all_match = True
            for ex in task_data['train'][:3]:
                inp = to_grid(ex['input'])
                expected = to_grid(ex['output'])
                result = solver_fn(inp)
                if result != expected:
                    all_match = False
                    break
            if all_match:
                task_solver_map[task_num] = solver_name
                break
        except Exception:
            continue

print(f"Mapped {len(task_solver_map)}/400 tasks to solvers")

# Check ONNX compatibility for each matched task
compatible = []
incompatible = defaultdict(set)

for task_num, solver_name in sorted(task_solver_map.items()):
    is_safe, unsafe_funcs, all_funcs = is_onnx_compatible(solver_name)
    if is_safe:
        compatible.append(task_num)
    else:
        for f in unsafe_funcs:
            incompatible[f].add(task_num)

print(f"\nONNX-compatible tasks: {len(compatible)}/399")
print(f"Incompatible tasks: {399 - len(compatible)}")

print("\nTop blocking primitives:")
for func, tasks_set in sorted(incompatible.items(), key=lambda x: -len(x[1]))[:15]:
    print(f"  {func:20s} blocks {len(tasks_set):3d} tasks")

# Save results
result = {
    "compatible_tasks": compatible,
    "compatible_count": len(compatible),
    "total_matched": len(task_solver_map),
    "blocking_primitives": {k: sorted(v) for k, v in incompatible.items()},
    "task_solvers": {f"task{k:03d}": v for k, v in task_solver_map.items() if k in compatible}
}

with open("onnx_compatible_tasks.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nCompatible task numbers: {compatible[:20]}...")
print("Saved to onnx_compatible_tasks.json")
