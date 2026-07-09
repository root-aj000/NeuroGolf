#!/usr/bin/env python3
"""
Match arc-dsl solvers to neurogolf tasks.
For each match, extract the DSL primitives used.
"""
import json
import sys
import re
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'arc-dsl'))

import solvers as S
from dsl import *

# Get all solver functions
solver_names = sorted([name for name in dir(S) if name.startswith('solve_')])
print(f"Found {len(solver_names)} solvers")

# Load all neurogolf tasks
tasks = {}
for i in range(1, 401):
    path = f"../neurogolf-2026/task{i:03d}.json"
    with open(path) as f:
        tasks[i] = json.load(f)
print(f"Loaded {len(tasks)} tasks")


def to_grid(list_2d):
    """Convert list of lists to tuple of tuples (ARC grid format)."""
    return tuple(tuple(row) for row in list_2d)


def check_solver_on_task(solver_fn, task_data, max_examples=3):
    """Check if a solver works on a task's training examples."""
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


def extract_primitives(solver_name):
    """Extract DSL function names used in a solver's source code."""
    solver_fn = getattr(S, solver_name)
    source = __import__('inspect').getsource(solver_fn)
    # Find all function calls that are DSL primitives
    dsl_funcs = set()
    # Common DSL function patterns
    calls = re.findall(r'\b([a-z_][a-z0-9_]*)\s*\(', source)
    for call in calls:
        if call in ('solve', 'return', 'print', 'if', 'else', 'for', 'while', 'try', 'except'):
            continue
        # Check if it's a DSL function
        try:
            obj = eval(f"__import__('dsl').{call}")
            if callable(obj):
                dsl_funcs.add(call)
        except (AttributeError, NameError):
            pass
    # Also check constants used
    constants = re.findall(r'\b([A-Z][A-Z0-9_]+)\b', source)
    return sorted(dsl_funcs)


# Match solvers to tasks
results = {}
matched_count = 0

for task_num, task_data in tasks.items():
    task_key = f"task{task_num:03d}"
    results[task_key] = {
        "task_num": task_num,
        "solver": None,
        "primitives": [],
        "input_shape": None,
        "output_shape": None,
        "num_train": len(task_data.get('train', [])),
        "num_test": len(task_data.get('test', [])),
        "has_arc_gen": 'arc-gen' in task_data,
        "grid_sizes": [],
    }

    # Get grid info
    if task_data['train']:
        inp = task_data['train'][0]['input']
        out = task_data['train'][0]['output']
        results[task_key]["input_shape"] = f"{len(inp)}x{len(inp[0])}" if inp else None
        results[task_key]["output_shape"] = f"{len(out)}x{len(out[0])}" if out else None
        results[task_key]["grid_sizes"] = [
            (len(ex['input']), len(ex['input'][0]), len(ex['output']), len(ex['output'][0]))
            for ex in task_data['train']
        ]

    # Try each solver
    for solver_name in solver_names:
        solver_fn = getattr(S, solver_name)
        if check_solver_on_task(solver_fn, task_data):
            results[task_key]["solver"] = solver_name
            results[task_key]["primitives"] = extract_primitives(solver_name)
            matched_count += 1
            print(f"  MATCHED: {task_key} -> {solver_name} (primitives: {results[task_key]['primitives']})")
            break

print(f"\nMatched {matched_count}/{len(tasks)} tasks")

# Save results
with open("solver_matches.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to solver_matches.json")
