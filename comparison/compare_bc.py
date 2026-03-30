#!/usr/bin/env python3
"""
BC Results Comparison Script
Compares Betweenness Centrality results between two implementations.

Usage:
  python3 compare_bc.py <file_A> <file_B> [--tolerance 1e-3] [--output comparison_report.txt]

Both files should contain lines in the format:
  node <id> BC=<value>
Lines not matching this pattern are skipped (headers, logs, etc.)
"""

import sys
import re
import argparse
import os
from collections import OrderedDict

def parse_bc_file(filepath):
    """Parse a BC output file and return dict {node_id: bc_value}."""
    bc_values = OrderedDict()
    pattern = re.compile(r'node\s+(\d+)\s+BC=([\d.eE+\-]+|nan|inf|-inf)')
    with open(filepath, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                node_id = int(m.group(1))
                bc_val = float(m.group(2))
                bc_values[node_id] = bc_val
    return bc_values

def compare_bc(bc_a, bc_b, tolerance_abs=1e-3, tolerance_rel=1e-4):
    """Compare two BC dictionaries and return comparison stats."""
    all_nodes = sorted(set(bc_a.keys()) | set(bc_b.keys()))
    
    total = len(all_nodes)
    exact_match = 0
    within_tol = 0
    mismatches = []
    only_in_a = []
    only_in_b = []
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    max_abs_node = -1
    max_rel_node = -1
    
    for node in all_nodes:
        if node not in bc_a:
            only_in_b.append(node)
            continue
        if node not in bc_b:
            only_in_a.append(node)
            continue
        
        va, vb = bc_a[node], bc_b[node]
        abs_diff = abs(va - vb)
        
        # Relative diff based on max value
        denom = max(abs(va), abs(vb), 1e-12)
        rel_diff = abs_diff / denom
        
        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
            max_abs_node = node
        if rel_diff > max_rel_diff:
            max_rel_diff = rel_diff
            max_rel_node = node
        
        if abs_diff == 0.0:
            exact_match += 1
            within_tol += 1
        elif abs_diff <= tolerance_abs or rel_diff <= tolerance_rel:
            within_tol += 1
        else:
            mismatches.append((node, va, vb, abs_diff, rel_diff))
    
    return {
        'total_nodes': total,
        'common_nodes': total - len(only_in_a) - len(only_in_b),
        'exact_match': exact_match,
        'within_tol': within_tol,
        'mismatches': mismatches,
        'only_in_a': only_in_a,
        'only_in_b': only_in_b,
        'max_abs_diff': max_abs_diff,
        'max_abs_node': max_abs_node,
        'max_rel_diff': max_rel_diff,
        'max_rel_node': max_rel_node,
    }

def format_report(file_a, file_b, bc_a, bc_b, stats, tol_abs, tol_rel):
    """Format a human-readable comparison report."""
    lines = []
    lines.append("=" * 70)
    lines.append("BETWEENNESS CENTRALITY COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append(f"File A: {file_a}  ({len(bc_a)} nodes)")
    lines.append(f"File B: {file_b}  ({len(bc_b)} nodes)")
    lines.append(f"Absolute Tolerance: {tol_abs}")
    lines.append(f"Relative Tolerance: {tol_rel}")
    lines.append("")
    
    common = stats['common_nodes']
    pct_exact = (stats['exact_match'] / common * 100) if common else 0
    pct_tol = (stats['within_tol'] / common * 100) if common else 0
    num_mismatch = len(stats['mismatches'])
    
    lines.append("--- SUMMARY ---")
    lines.append(f"Total unique nodes:       {stats['total_nodes']}")
    lines.append(f"Common nodes:             {common}")
    lines.append(f"Only in File A:           {len(stats['only_in_a'])}")
    lines.append(f"Only in File B:           {len(stats['only_in_b'])}")
    lines.append(f"Exact matches:            {stats['exact_match']} / {common}  ({pct_exact:.2f}%)")
    lines.append(f"Within tolerance:         {stats['within_tol']} / {common}  ({pct_tol:.2f}%)")
    lines.append(f"Mismatches (out of tol):  {num_mismatch} / {common}  ({(num_mismatch/common*100) if common else 0:.2f}%)")
    lines.append(f"Max absolute difference:  {stats['max_abs_diff']:.6f}  (node {stats['max_abs_node']})")
    lines.append(f"Max relative difference:  {stats['max_rel_diff']:.8f}  (node {stats['max_rel_node']})")
    lines.append("")
    
    if num_mismatch == 0:
        lines.append("✅ RESULT: ALL VALUES MATCH within tolerance!")
    else:
        lines.append(f"❌ RESULT: {num_mismatch} MISMATCHES found!")
    lines.append("")
    
    # Show top mismatches
    if stats['mismatches']:
        lines.append("--- TOP MISMATCHES (sorted by abs diff, max 20) ---")
        lines.append(f"{'Node':>8}  {'File A':>16}  {'File B':>16}  {'Abs Diff':>14}  {'Rel Diff':>12}")
        lines.append("-" * 70)
        sorted_mm = sorted(stats['mismatches'], key=lambda x: -x[3])[:20]
        for node, va, vb, ad, rd in sorted_mm:
            lines.append(f"{node:>8}  {va:>16.3f}  {vb:>16.3f}  {ad:>14.6f}  {rd:>12.8f}")
        lines.append("")
    
    lines.append("=" * 70)
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Compare two BC result files")
    parser.add_argument("file_a", help="Path to first BC result file")
    parser.add_argument("file_b", help="Path to second BC result file")
    parser.add_argument("--tolerance-abs", type=float, default=1e-3, help="Absolute tolerance (default: 1e-3)")
    parser.add_argument("--tolerance-rel", type=float, default=1e-4, help="Relative tolerance (default: 1e-4)")
    parser.add_argument("--output", "-o", help="Output file for the report (optional)")
    args = parser.parse_args()
    
    print(f"Parsing {args.file_a} ...")
    bc_a = parse_bc_file(args.file_a)
    print(f"  -> Found {len(bc_a)} nodes")
    
    print(f"Parsing {args.file_b} ...")
    bc_b = parse_bc_file(args.file_b)
    print(f"  -> Found {len(bc_b)} nodes")
    
    print("Comparing ...")
    stats = compare_bc(bc_a, bc_b, args.tolerance_abs, args.tolerance_rel)
    
    report = format_report(args.file_a, args.file_b, bc_a, bc_b, stats, args.tolerance_abs, args.tolerance_rel)
    print(report)
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(report + "\n")
        print(f"\nReport saved to: {args.output}")

if __name__ == "__main__":
    main()
