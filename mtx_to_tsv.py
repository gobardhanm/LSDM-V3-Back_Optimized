#!/usr/bin/env python3
"""
Convert a Matrix Market (.mtx) edge list into a TSV file.

Assumptions:
- Input is Matrix Market coordinate format:
    %%MatrixMarket matrix coordinate <field> <symmetry>
- Data lines contain at least:
    src  dst [weight]
- Vertex IDs are kept as-is.
- Lines starting with '%' are ignored.

Usage:
  python mtx_to_tsv.py input.mtx [output.tsv]

If output.tsv is not provided, the script writes input.mtx + ".tsv".
"""

import sys
from pathlib import Path


def mtx_to_tsv(in_path: str, out_path: str | None = None) -> None:
  in_file = Path(in_path)
  if out_path is None:
    out_file = in_file.with_suffix(in_file.suffix + ".tsv")
  else:
    out_file = Path(out_path)

  edges: list[tuple[int, int, float]] = []
  seen_size_line = False

  with in_file.open("r") as f:
    for raw_line in f:
      line = raw_line.strip()
      if not line or line.startswith("%"):
        continue

      parts = line.split()

      # First non-comment non-empty line is the size line: rows cols nnz
      if not seen_size_line:
        if len(parts) < 3:
          raise ValueError("Invalid Matrix Market size line.")
        seen_size_line = True
        continue

      if len(parts) < 2:
        continue

      try:
        src = int(parts[0])
        dst = int(parts[1])
        w = float(parts[2]) if len(parts) >= 3 else 1.0
      except ValueError:
        # Skip malformed data lines
        continue

      edges.append((src, dst, w))

  if not seen_size_line:
    raise ValueError("Matrix Market header/size line not found.")

  with out_file.open("w") as f:
    for src, dst, w in edges:
      # Preserve integer-looking weights as integers for cleaner TSV output.
      w_str = str(int(w)) if w.is_integer() else str(w)
      f.write(f"{src}\t{dst}\t{w_str}\n")

  print(f"Wrote TSV file to: {out_file}")


def main(argv: list[str]) -> None:
  if not (2 <= len(argv) <= 3):
    print("Usage: python mtx_to_tsv.py input.mtx [output.tsv]", file=sys.stderr)
    sys.exit(1)

  in_path = argv[1]
  out_path = argv[2] if len(argv) == 3 else None
  mtx_to_tsv(in_path, out_path)


if __name__ == "__main__":
  main(sys.argv)
