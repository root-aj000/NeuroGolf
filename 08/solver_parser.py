"""
Solver Parser: Extract full solver source from solvers.py, parse into structured ops.

Usage:
    from solver_parser import extract_solver, parse_solver
    source = extract_solver("solve_007bbfb7")
    ops = parse_solver(source)
    # ops = [("hupscale", ["I", "THREE"]), ("vupscale", ["x1", "THREE"]), ...]
"""

import re
import ast
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

SOLVERS_PATH = Path(__file__).parent.parent / "arc-dsl" / "solvers.py"

# DSL constants from arc-dsl/constants.py
CONSTANTS = {
    "ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4,
    "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10,
    "NEG_ONE": -1, "NEG_TWO": -2,
    "T": True, "F": False,
    "DOWN": (1, 0), "RIGHT": (0, 1), "UP": (-1, 0), "LEFT": (0, -1),
    "ORIGIN": (0, 0), "UNITY": (1, 1), "NEG_UNITY": (-1, -1),
    "UP_RIGHT": (-1, 1), "DOWN_LEFT": (1, -1),
    "UP_LEFT": (-1, -1), "DOWN_RIGHT": (1, 1),
    "ZERO_BY_TWO": (0, 2), "TWO_BY_ZERO": (2, 0), "TWO_BY_TWO": (2, 2),
    "THREE_BY_THREE": (3, 3),
}

# All DSL primitives that exist
ALL_PRIMITIVES = {
    # Grid transforms
    "hmirror", "vmirror", "dmirror", "cmirror",
    "rot90", "rot180", "rot270",
    "crop", "trim", "tophalf", "bottomhalf", "lefthalf", "righthalf",
    "hconcat", "vconcat", "hsplit", "vsplit",
    "hupscale", "vupscale", "upscale", "downscale",
    "fill", "underfill", "replace", "switch", "cellwise",
    "paint", "underpaint", "cover", "move", "recolor",
    "compress",
    # Value ops
    "add", "subtract", "multiply", "divide", "double", "halve",
    "increment", "decrement", "crement", "invert", "sign",
    "even", "positive", "greater", "equality", "less",
    "flip", "both", "either",
    # Object/indices
    "objects", "partition", "fgpartition", "colorfilter",
    "ofcolor", "asindices", "asobject", "toobject", "toindices",
    "sfilter", "mfilter", "extract", "sizefilter",
    "merge", "combine", "difference", "intersection",
    "first", "last", "other", "remove", "dedupe",
    "size", "maximum", "minimum", "mostcolor", "leastcolor",
    "palette", "numcolors", "color", "colorcount",
    "mostcommon", "leastcommon",
    # Shape/metrics
    "height", "width", "shape", "portrait", "square",
    "vline", "hline",
    # Positional
    "ulcorner", "urcorner", "llcorner", "lrcorner",
    "uppermost", "lowermost", "leftmost", "rightmost",
    "center", "centerofmass", "corners",
    "dneighbors", "ineighbors", "neighbors",
    "hmatching", "vmatching", "manhattan", "adjacent", "bordering",
    "position", "gravitate",
    # Spatial
    "shift", "normalize",
    "connect", "shoot", "box", "inbox", "outbox", "backdrop", "delta",
    "vfrontier", "hfrontier",
    "subgrid",
    # Construction
    "canvas", "interval", "astuple", "toivec", "tojvec",
    "initset", "insert", "product", "repeat", "totuple",
    "contained",
    # Higher-order
    "compose", "chain", "matcher", "rbind", "lbind",
    "power", "fork", "branch",
    "apply", "rapply", "mapply", "papply", "mpapply", "prapply",
    "order", "valmax", "valmin", "argmax", "argmin",
    # Misc
    "frontiers", "occurrences", "hperiod", "vperiod",
    "identity",
}


def extract_solver(solver_name: str, solvers_path: Path = SOLVERS_PATH) -> str:
    """Extract complete solver function source from solvers.py.

    Args:
        solver_name: e.g. "solve_007bbfb7"
        solvers_path: path to solvers.py

    Returns:
        Full function source as string.
    """
    content = solvers_path.read_text()

    # Find the function definition
    pattern = rf'(def {re.escape(solver_name)}\(I\):)'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Solver {solver_name} not found in {solvers_path}")

    start = match.start()

    # Find the end: next "def " at column 0, or end of file
    next_def = re.search(r'\ndef ', content[match.end():])
    if next_def:
        end = match.end() + next_def.start()
    else:
        end = len(content)

    source = content[start:end].strip()
    return source


def extract_all_solvers(solvers_path: Path = SOLVERS_PATH) -> Dict[str, str]:
    """Extract all solver functions. Returns {name: source}."""
    content = solvers_path.read_text()
    solvers = {}

    for match in re.finditer(r'^def (solve_\w+)\(I\):', content, re.MULTILINE):
        name = match.group(1)
        start = match.start()

        # Find end
        next_def = re.search(r'\ndef ', content[match.end():])
        if next_def:
            end = match.end() + next_def.start()
        else:
            end = len(content)

        solvers[name] = content[start:end].strip()

    return solvers


def parse_solver(source: str) -> List[Dict[str, Any]]:
    """Parse solver source into structured operation list.

    Returns list of dicts:
        [{"var": "x1", "func": "hupscale", "args": ["I", "THREE"]},
         {"var": "x2", "func": "vupscale", "args": ["x1", "THREE"]},
         {"var": "O",  "func": "cellwise", "args": ["x2", "x6"]}]
    """
    lines = source.strip().split('\n')
    ops = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('def ') or line == 'return O':
            continue

        # Match: var = func(args)  or  O = func(args)
        m = re.match(r'(\w+)\s*=\s*(\w+)\((.*)\)\s*$', line)
        if not m:
            # Try: var = constant (e.g., x1 = I)
            m2 = re.match(r'(\w+)\s*=\s*(\w+)\s*$', line)
            if m2:
                ops.append({
                    "var": m2.group(1),
                    "func": "identity",
                    "args": [m2.group(2)],
                })
            continue

        var_name = m.group(1)
        func_name = m.group(2)
        args_str = m.group(3).strip()

        # Split args carefully (handle nested parens)
        args = _split_args(args_str) if args_str else []

        ops.append({
            "var": var_name,
            "func": func_name,
            "args": args,
        })

    return ops


def _split_args(s: str) -> List[str]:
    """Split argument string by commas, respecting nested parentheses."""
    parts = []
    depth = 0
    current = []

    for ch in s:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)

    if current:
        parts.append(''.join(current).strip())

    return parts


def resolve_constants(args: List[str]) -> List[Any]:
    """Resolve constant names to their values.

    ["I", "THREE", "x1"] → ["I", 3, "x1"]
    """
    result = []
    for a in args:
        if a in CONSTANTS:
            result.append(CONSTANTS[a])
        elif a.isdigit() or (a.startswith('-') and a[1:].isdigit()):
            result.append(int(a))
        else:
            result.append(a)  # variable reference
    return result


def identify_primitives(source: str) -> List[str]:
    """Get list of unique DSL primitives used in a solver."""
    ops = parse_solver(source)
    prims = set()
    for op in ops:
        if op["func"] not in ("identity",):
            prims.add(op["func"])
    return sorted(prims)


def get_solver_name_for_task(task_num: int, meta_path: str = None) -> str:
    """Get solver function name for a task number from tasks_meta.json."""
    if meta_path is None:
        meta_path = Path(__file__).parent.parent / "07" / "tasks_meta.json"

    import json
    with open(meta_path) as f:
        meta = json.load(f)

    task_key = f"task{task_num:03d}"
    if task_key in meta:
        return meta[task_key].get("solver", "")
    return ""


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    import json

    # Extract and parse a few solvers
    test_solvers = [
        "solve_007bbfb7",  # task001: hupscale + vupscale + cellwise
        "solve_3c9b0459",  # rot180
        "solve_67a3c6ac",  # vmirror
    ]

    for name in test_solvers:
        try:
            source = extract_solver(name)
            ops = parse_solver(source)
            prims = identify_primitives(source)
            print(f"\n{'='*50}")
            print(f"{name}:")
            print(f"  Source:\n    {source.replace(chr(10), chr(10)+'    ')}")
            print(f"  Parsed ops: {len(ops)}")
            for op in ops:
                print(f"    {op['var']} = {op['func']}({', '.join(str(a) for a in op['args'])})")
            print(f"  Primitives: {prims}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Test extract_all_solvers
    all_solvers = extract_all_solvers()
    print(f"\n{'='*50}")
    print(f"Total solvers extracted: {len(all_solvers)}")

    # Parse a complex one
    source = extract_solver("solve_00d62c1b")  # task002: objects + higher-order
    ops = parse_solver(source)
    print(f"\nsolve_00d62c1b (task002):")
    print(f"  Source:\n    {source.replace(chr(10), chr(10)+'    ')}")
    print(f"  Parsed ops:")
    for op in ops:
        print(f"    {op['var']} = {op['func']}({', '.join(str(a) for a in op['args'])})")
