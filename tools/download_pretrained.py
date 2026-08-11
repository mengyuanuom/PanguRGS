#!/usr/bin/env python3
"""Download official pretrained backbones used by ToolRGSNPU/CROG-NPU."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pretrained import ARTIFACTS, ensure_pretrained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("pretrain"))
    parser.add_argument(
        "--output",
        type=Path,
        help="Exact destination; valid only when downloading one artifact.",
    )
    parser.add_argument(
        "--ca-bundle",
        type=Path,
        help="PEM CA bundle for an HTTPS-inspecting proxy.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=None,
        help="Disable TLS verification (trusted networks only).",
    )
    args = parser.parse_args()

    unknown = sorted(set(args.artifacts) - set(ARTIFACTS))
    if unknown:
        parser.error("unknown artifact(s): " + ", ".join(unknown))

    if args.all and args.artifacts:
        parser.error("use either --all or explicit artifact names")
    names = list(ARTIFACTS) if args.all else list(args.artifacts)
    if args.output is not None and len(names) != 1:
        parser.error("--output requires exactly one artifact")
    if not names:
        print("Available official pretrained weights:")
        for key, artifact in ARTIFACTS.items():
            print(f"  {key:22} -> {artifact.filename}\n    {artifact.url}")
        return 0

    for key in names:
        artifact = ARTIFACTS[key]
        target = args.output or (args.output_dir / artifact.filename)
        ensure_pretrained(
            target,
            key,
            ca_bundle=args.ca_bundle,
            insecure=args.insecure,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
