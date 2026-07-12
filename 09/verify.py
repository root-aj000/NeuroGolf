import os
import onnx
from pathlib import Path
from collections import defaultdict

# Define disallowed operations
DISALLOWED_OPS = {
    'Loop',
    'Scan',
    'NonZero',
    'Unique',
    'Script',
    'Function'
}

def get_all_ops_from_model(model_path):
    """Extract all operation types from an ONNX model."""
    try:
        model = onnx.load(model_path)
        ops = set()
        
        # Get ops from main graph
        for node in model.graph.node:
            ops.add(node.op_type)
            
            # Check for subgraphs (e.g., inside If, Loop, Scan nodes)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    ops.update(get_ops_from_graph(attr.g))
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        ops.update(get_ops_from_graph(subgraph))
        
        # Check for functions
        if model.functions:
            ops.add('Function')
            for func in model.functions:
                for node in func.node:
                    ops.add(node.op_type)
        
        return ops, None
    except Exception as e:
        return None, str(e)

def get_ops_from_graph(graph):
    """Recursively get ops from a graph."""
    ops = set()
    for node in graph.node:
        ops.add(node.op_type)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                ops.update(get_ops_from_graph(attr.g))
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    ops.update(get_ops_from_graph(subgraph))
    return ops

def scan_onnx_directory(directory):
    """Scan all ONNX files in a directory for disallowed operations."""
    onnx_dir = Path(directory)
    onnx_files = list(onnx_dir.glob("*.onnx"))
    
    print(f"Found {len(onnx_files)} ONNX files in '{directory}'")
    print(f"Checking for disallowed ops: {', '.join(sorted(DISALLOWED_OPS))}")
    print("=" * 80)
    
    results = {
        'passed': [],
        'failed': [],
        'errors': []
    }
    
    disallowed_usage = defaultdict(list)  # op -> list of files
    
    for i, onnx_file in enumerate(onnx_files, 1):
        filename = onnx_file.name
        print(f"[{i}/{len(onnx_files)}] Scanning: {filename}...", end=" ")
        
        ops, error = get_all_ops_from_model(onnx_file)
        
        if error:
            print(f"ERROR: {error}")
            results['errors'].append((filename, error))
            continue
        
        found_disallowed = ops & DISALLOWED_OPS
        
        if found_disallowed:
            print(f"FAILED - Found: {', '.join(sorted(found_disallowed))}")
            results['failed'].append((filename, found_disallowed))
            for op in found_disallowed:
                disallowed_usage[op].append(filename)
        else:
            print("PASSED")
            results['passed'].append(filename)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files scanned: {len(onnx_files)}")
    print(f"  ✓ Passed: {len(results['passed'])}")
    print(f"  ✗ Failed: {len(results['failed'])}")
    print(f"  ! Errors: {len(results['errors'])}")
    
    if results['failed']:
        print("\n" + "-" * 40)
        print("FAILED FILES:")
        print("-" * 40)
        for filename, ops in results['failed']:
            print(f"  {filename}")
            print(f"    Disallowed ops: {', '.join(sorted(ops))}")
    
    if disallowed_usage:
        print("\n" + "-" * 40)
        print("DISALLOWED OPERATIONS USAGE:")
        print("-" * 40)
        for op in sorted(disallowed_usage.keys()):
            files = disallowed_usage[op]
            print(f"  {op}: used in {len(files)} file(s)")
            for f in files[:5]:  # Show first 5 files
                print(f"    - {f}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")
    
    if results['errors']:
        print("\n" + "-" * 40)
        print("FILES WITH ERRORS:")
        print("-" * 40)
        for filename, error in results['errors']:
            print(f"  {filename}: {error}")
    
    return results

def save_report(results, output_file="scan_report.txt"):
    """Save the scan results to a file."""
    with open(output_file, 'w') as f:
        f.write("ONNX DISALLOWED OPERATIONS SCAN REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Disallowed ops: {', '.join(sorted(DISALLOWED_OPS))}\n\n")
        
        f.write("PASSED FILES:\n")
        for filename in results['passed']:
            f.write(f"  {filename}\n")
        
        f.write("\nFAILED FILES:\n")
        for filename, ops in results['failed']:
            f.write(f"  {filename}: {', '.join(sorted(ops))}\n")
        
        f.write("\nERROR FILES:\n")
        for filename, error in results['errors']:
            f.write(f"  {filename}: {error}\n")
    
    print(f"\nReport saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scan ONNX files for disallowed operations")
    parser.add_argument("--dir", default="onnx", help="Directory containing ONNX files")
    parser.add_argument("--report", default="scan_report.txt", help="Output report file")
    parser.add_argument("--save", action="store_true", help="Save report to file")
    
    args = parser.parse_args()
    
    results = scan_onnx_directory(args.dir)
    
    if args.save:
        save_report(results, args.report)