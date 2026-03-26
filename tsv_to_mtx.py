#!/usr/bin/env python3
"""
Convert a TSV edge list into a Matrix Market (.mtx) file.

Assumptions:
- Input has three whitespace-separated columns per non-comment line:
    src  dst  weight
- Vertex IDs are 1-based already (common for graph datasets).
- Lines starting with '#' are ignored.

Usage:
  python tsv_to_mtx.py input.tsv [output.mtx]

If output.mtx is not provided, the script writes input.tsv + ".mtx".
"""

import sys
from pathlib import Path


def tsv_to_mtx(in_path: str, out_path: str | None = None) -> None:
  in_file = Path(in_path)
  if out_path is None:
    out_file = in_file.with_suffix(in_file.suffix + ".mtx")
  else:
    out_file = Path(out_path)

  edges: list[tuple[int, int, float]] = []
  max_vertex = 0

  with in_file.open("r") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      parts = line.split()
      if len(parts) < 2:
        continue

      try:
        src = int(parts[0])
        dst = int(parts[1])
        w = float(parts[2]) if len(parts) >= 3 else 1.0
      except ValueError:
        # Skip malformed lines
        continue

      max_vertex = max(max_vertex, src, dst)
      edges.append((src, dst, w))

  n = max_vertex
  m = max_vertex
  nnz = len(edges)

  with out_file.open("w") as f:
    f.write("%%MatrixMarket matrix coordinate real general\n")
    f.write(f"{n} {m} {nnz}\n")
    for src, dst, w in edges:
      f.write(f"{src} {dst} {w}\n")

  print(f"Wrote Matrix Market file to: {out_file}")


def main(argv: list[str]) -> None:
  if not (2 <= len(argv) <= 3):
    print("Usage: python tsv_to_mtx.py input.tsv [output.mtx]", file=sys.stderr)
    sys.exit(1)

  in_path = argv[1]
  out_path = argv[2] if len(argv) == 3 else None
  tsv_to_mtx(in_path, out_path)


if __name__ == "__main__":
  main(sys.argv)

