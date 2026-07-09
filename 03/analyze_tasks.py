#!/usr/bin/env python3
"""
Comprehensive analysis of neurogolf tasks using arc-dsl solvers.
Extracts primitives, verifies solutions, outputs debug JSON.
"""
import json
import sys
import os
import re
from collections import Counter, defaultdict
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'arc-dsl'))

import solvers as S
from dsl import *

# Get all solver functions
solver_names = sorted([name for name in dir(S) if name.startswith('solve_')])
print(f"Found {len(solver_names)} solvers")


def to_grid(list_2d):
    """Convert list of lists to tuple of tuples (ARC grid format)."""
    return tuple(tuple(row) for row in list_2d)


def verify_solver(solver_fn, task_data):
    """Verify a solver works on all training examples."""
    results = []
    try:
        for i, ex in enumerate(task_data['train']):
            inp = to_grid(ex['input'])
            expected = to_grid(ex['output'])
            result = solver_fn(inp)
            match = result == expected
            results.append({
                "example": i,
                "match": match,
                "input_shape": f"{len(inp)}x{len(inp[0])}",
                "output_shape": f"{len(result)}x{len(result[0])}" if result else None
            })
    except Exception as e:
        results.append({"error": str(e)})
    return results


def extract_primitives(solver_name):
    """Extract DSL function names used in a solver's source code."""
    solver_fn = getattr(S, solver_name)
    source = inspect.getsource(solver_fn)
    
    # Find all function calls
    calls = re.findall(r'\b([a-z_][a-z0-9_]*)\s*\(', source)
    
    dsl_funcs = set()
    skip_words = {'solve', 'return', 'print', 'if', 'else', 'for', 'while', 'try', 'except', 'input', 'output'}
    
    for call in calls:
        if call in skip_words:
            continue
        try:
            obj = eval(f"__import__('dsl').{call}")
            if callable(obj):
                dsl_funcs.add(call)
        except (AttributeError, NameError):
            pass
    
    # Extract constants
    constants = re.findall(r'\b([A-Z][A-Z0-9_]+)\b', source)
    constants = [c for c in constants if c not in ('I', 'O', 'T', 'F')]
    
    return {
        "functions": sorted(dsl_funcs),
        "constants": sorted(set(constants)),
        "source_preview": source.strip()[:200]
    }


# Load all neurogolf tasks
tasks = {}
for i in range(1, 401):
    path = f"../neurogolf-2026/task{i:03d}.json"
    with open(path) as f:
        tasks[i] = json.load(f)
print(f"Loaded {len(tasks)} tasks")

# Analyze each task
output = {
    "summary": {
        "total_tasks": 400,
        "matched_tasks": 0,
        "unmatched_tasks": 0,
        "verification_passed": 0,
        "verification_failed": 0,
    },
    "primitive_frequency": {},
    "tasks": {}
}

all_primitives = Counter()

for task_num in range(1, 401):
    task_key = f"task{task_num:03d}"
    task_data = tasks[task_num]
    
    # Get grid info
    train_info = []
    for ex in task_data['train']:
        inp = ex['input']
        out = ex['output']
        train_info.append({
            "input_shape": f"{len(inp)}x{len(inp[0])}",
            "output_shape": f"{len(out)}x{len(out[0])}",
            "same_size": len(inp) == len(out) and len(inp[0]) == len(out[0])
        })
    
    task_result = {
        "task_num": task_num,
        "solver": None,
        "primitives": None,
        "verification": None,
        "grid_info": {
            "num_train": len(task_data['train']),
            "num_test": len(task_data.get('test', [])),
            "has_arc_gen": 'arc-gen' in task_data,
            "train_examples": train_info
        }
    }
    
    # Try to find matching solver
    found = False
    for solver_name in solver_names:
        solver_fn = getattr(S, solver_name)
        try:
            all_match = True
            for ex in task_data['train'][:3]:  # Check first 3 examples
                inp = to_grid(ex['input'])
                expected = to_grid(ex['output'])
                result = solver_fn(inp)
                if result != expected:
                    all_match = False
                    break
            
            if all_match:
                task_result["solver"] = solver_name
                task_result["primitives"] = extract_primitives(solver_name)
                
                # Verify on all examples
                task_result["verification"] = verify_solver(solver_fn, task_data)
                
                # Count primitives
                for func in task_result["primitives"]["functions"]:
                    all_primitives[func] += 1
                
                output["summary"]["matched_tasks"] += 1
                if all(r.get("match", False) for r in task_result["verification"]):
                    output["summary"]["verification_passed"] += 1
                else:
                    output["summary"]["verification_failed"] += 1
                
                found = True
                break
        except Exception:
            continue
    
    if not found:
        output["summary"]["unmatched_tasks"] += 1
        task_result["solver"] = "NO_MATCH"
    
    output["tasks"][task_key] = task_result

# Add primitive frequency
output["primitive_frequency"] = dict(all_primitives.most_common())

# Save output
with open("task_analysis.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSummary:")
print(f"  Matched: {output['summary']['matched_tasks']}")
print(f"  Unmatched: {output['summary']['unmatched_tasks']}")
print(f"  Verification passed: {output['summary']['verification_passed']}")
print(f"  Verification failed: {output['summary']['verification_failed']}")
print(f"\nTop 20 primitives:")
for func, count in all_primitives.most_common(20):
    print(f"  {func}: {count}")
print(f"\nResults saved to task_analysis.json")
