#!/usr/bin/env python3
"""Download and verify the official OpenAI CLIP RN50 checkpoint."""

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request


URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/"
    "RN50.pt"
)
SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(output: Path) -> None:
    if output.is_file():
        if sha256(output) == SHA256:
            print(f"[skip] verified CLIP RN50: {output}")
            return
        raise RuntimeError(
            f"Existing checkpoint failed SHA-256 validation: {output}. "
            "Remove it before downloading the official file."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    request = urllib.request.Request(
        URL, headers={"User-Agent": "CROG-NPU-weight-downloader/1.0"}
    )
    print(f"[download] OpenAI CLIP RN50\n  from: {URL}\n  to:   {output}")
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as stream:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
                received += len(block)
                if total:
                    print(
                        f"\r  {received / 1024**2:.1f}/{total / 1024**2:.1f} MiB",
                        end="",
                        flush=True,
                    )
            if total:
                print()
        if sha256(temporary) != SHA256:
            raise RuntimeError(f"SHA-256 validation failed: {temporary}")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[ok] verified CLIP RN50: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pretrain/RN50.pt"),
        help="Checkpoint destination",
    )
    args = parser.parse_args()
    try:
        download(args.output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
