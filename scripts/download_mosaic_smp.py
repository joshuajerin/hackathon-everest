#!/usr/bin/env python3
"""Download a small, deterministic MOSAiC SnowMicroPen calibration subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

INDEX_URL = "https://doi.pangaea.de/10.1594/PANGAEA.935554?format=textfile"
FILE_URL = "https://download.pangaea.de/dataset/935554/files/{filename}"
DOI = "https://doi.org/10.1594/PANGAEA.935554"
LICENSE = "CC BY 4.0"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "hackathon-everest-calibration/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def parse_index(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8", errors="replace")
    marker = "*/\n"
    if marker not in text:
        raise ValueError("PANGAEA index did not contain the expected metadata terminator")
    table = text.split(marker, 1)[1]
    return list(csv.DictReader(io.StringIO(table), delimiter="\t"))


def select_profiles(rows: list[dict[str, str]], per_month: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        filename = row.get("Binary", "")
        comment = row.get("Comment", "").lower()
        timestamp = row.get("Date/Time", "")
        profile_id = row.get("ID", "")
        if not filename.endswith(".pnt") or len(timestamp) < 7:
            continue
        if any(flag in comment for flag in ("corrupt", "failed", "test", "empty")):
            continue
        if not profile_id.startswith(("M", "T")):
            continue
        grouped[timestamp[:7]].append(row)
    selected = []
    for month in sorted(grouped):
        candidates = sorted(grouped[month], key=lambda row: (row["Event"], row["Binary"]))
        count = min(per_month, len(candidates))
        indices = [round(i * (len(candidates) - 1) / max(1, count - 1)) for i in range(count)]
        selected.extend(candidates[index] for index in indices)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/external/mosaic_smp"))
    parser.add_argument("--per-month", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    index_payload = fetch(INDEX_URL)
    (args.out / "mosaic-smp-index.tab").write_bytes(index_payload)
    selected = select_profiles(parse_index(index_payload), args.per_month)
    files = []
    failures = []
    for row in selected:
        filename = row["Binary"]
        try:
            payload = fetch(FILE_URL.format(filename=filename))
            if len(payload) < 1000:
                raise ValueError(f"unexpectedly short payload ({len(payload)} bytes)")
            (args.out / filename).write_bytes(payload)
            files.append(
                {
                    "filename": filename,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "event": row["Event"],
                    "timestamp": row["Date/Time"],
                    "location": row["Location"],
                    "id": row["ID"],
                    "url": FILE_URL.format(filename=filename),
                }
            )
        except (OSError, TimeoutError, urllib.error.URLError, ValueError) as error:
            failures.append({"filename": filename, "error": str(error)})
    manifest = {
        "dataset": "MOSAiC Snowpit SnowMicroPen force profiles",
        "doi": DOI,
        "license": LICENSE,
        "citation": (
            "Macfarlane, A. R. et al. (2021), Snowpit SnowMicroPen (SMP) force profiles "
            "collected during the MOSAiC expedition, PANGAEA, doi:10.1594/PANGAEA.935554."
        ),
        "selection": f"Up to {args.per_month} deterministic M/T profiles per calendar month",
        "files": files,
        "failures": failures,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"downloaded": len(files), "failed": len(failures), "out": str(args.out)}, indent=2))
    return 0 if files else 1


if __name__ == "__main__":
    raise SystemExit(main())
