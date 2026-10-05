#!/usr/bin/env python3
"""Recover complete JSON objects separated by literal backslash-n into valid JSONL.

This does not modify source captures. Output is a normalized derivative; record both
input and output hashes when using it.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    raw = args.input.read_bytes()
    # The flawed writer emitted literal '\\n' byte pairs, not line-feed bytes.
    pieces = [part for part in raw.split(b"\\n") if part]
    rows = []
    for i, piece in enumerate(pieces):
        try:
            row = json.loads(piece)
        except Exception as exc:
            raise SystemExit(f"invalid JSON object {i} in {args.input}: {exc}")
        if not isinstance(row, dict):
            raise SystemExit(f"object {i} is not a JSON object")
        rows.append(row)
    if not rows:
        raise SystemExit("no JSON objects found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as out:
        for row in rows:
            out.write(json.dumps(row, separators=(",", ":")).encode() + b"\n")
    # Reparse every emitted line and require final newline.
    emitted = args.output.read_bytes()
    lines = emitted.splitlines()
    if len(lines) != len(rows) or not emitted.endswith(b"\n"):
        raise SystemExit("normalized output line-count/final-newline validation failed")
    for line in lines:
        if not isinstance(json.loads(line), dict):
            raise SystemExit("normalized output contains non-object")
    print(json.dumps({"source": str(args.input), "source_sha256": sha(args.input),
                      "normalized": str(args.output), "normalized_sha256": sha(args.output),
                      "rows": len(rows), "validated": True}, separators=(",", ":")))


if __name__ == "__main__":
    main()
