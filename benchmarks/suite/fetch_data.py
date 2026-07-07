#!/usr/bin/env python3
"""Fetch benchmark corpora into ~/.cache/qwen3_asr_causal/bench/.

LibriSpeech splits download from OpenSLR with pinned sha256 verification and
extract to the standard layout (LibriSpeech/<split>/<spk>/<chap>/*.flac +
*.trans.txt). Everything is idempotent: verified tarballs are not
re-downloaded, extracted splits are not re-extracted.

MCIF (IWSLT long-form talks) cannot be auto-downloaded (the source requires
manual license acceptance); ``--mcif`` prints the manual steps and the local
path conventions the suite expects.

Usage:
    python benchmarks/suite/fetch_data.py --librispeech test-clean
    python benchmarks/suite/fetch_data.py --librispeech test-clean --librispeech test-other
    python benchmarks/suite/fetch_data.py --mcif
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

BENCH_ROOT = Path.home() / ".cache" / "qwen3_asr_causal" / "bench"

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{name}.tar.gz"
LIBRISPEECH_SHA256 = {
    "test-clean": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
    "test-other": "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",
}

MCIF_INSTRUCTIONS = """\
MCIF / IWSLT long-form talks: manual download required (license acceptance).

1. Get the MCIF long-form audio from the IWSLT 2026 release (requires
   accepting the source's terms; see https://iwslt.org).
2. Place the wavs under the path convention the suite expects:
       ~/Downloads/IWSLT2026/mcif-long-trans/audio/*.wav
3. Reference transcripts live in this repo at:
       experiments/qwen3-causal/data/mcif_refs/manifest.human.jsonl
   (rows: {"audio_id","wav","language","human_text"}).
4. Run, rebasing the manifest's relative wav names onto the audio dir:
       python benchmarks/suite/run_wer.py \\
           --manifest-jsonl experiments/qwen3-causal/data/mcif_refs/manifest.human.jsonl \\
           --audio-dir ~/Downloads/IWSLT2026/mcif-long-trans/audio
"""


def sha256_of(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_mark = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if done >= next_mark:
                pct = f" ({done * 100 // total}%)" if total else ""
                print(f"  {done / 1e6:.0f} MB{pct}", flush=True)
                next_mark += 50 * (1 << 20)
    tmp.rename(dest)


def fetch_librispeech(name: str) -> Path:
    expected = LIBRISPEECH_SHA256[name]
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    split_dir = BENCH_ROOT / "LibriSpeech" / name
    if split_dir.is_dir() and any(split_dir.rglob("*.trans.txt")):
        print(f"{name}: already extracted at {split_dir}", flush=True)
        return split_dir

    tarball = BENCH_ROOT / f"{name}.tar.gz"
    if tarball.exists() and sha256_of(tarball) == expected:
        print(f"{name}: verified tarball already at {tarball}", flush=True)
    else:
        if tarball.exists():
            print(f"{name}: existing tarball failed sha256; re-downloading", flush=True)
            tarball.unlink()
        download(LIBRISPEECH_URL.format(name=name), tarball)
        actual = sha256_of(tarball)
        if actual != expected:
            tarball.unlink()
            raise SystemExit(
                f"{name}: sha256 mismatch (got {actual}, expected {expected}); "
                "deleted the download, try again"
            )
        print(f"{name}: sha256 verified", flush=True)

    print(f"{name}: extracting to {BENCH_ROOT}", flush=True)
    with tarfile.open(tarball, "r:gz") as tar:
        try:
            tar.extractall(BENCH_ROOT, filter="data")
        except TypeError:  # Python < 3.11.4 without the filter parameter
            tar.extractall(BENCH_ROOT)  # noqa: S202 - hash-pinned archive
    if not any(split_dir.rglob("*.trans.txt")):
        raise SystemExit(f"{name}: extraction produced no *.trans.txt under {split_dir}")
    print(f"{name}: ready at {split_dir}", flush=True)
    return split_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--librispeech",
        action="append",
        choices=sorted(LIBRISPEECH_SHA256),
        default=None,
        help="LibriSpeech split to fetch (repeatable)",
    )
    parser.add_argument(
        "--mcif", action="store_true", help="print MCIF manual-download instructions"
    )
    args = parser.parse_args()
    if not args.librispeech and not args.mcif:
        parser.print_help()
        sys.exit(2)

    for name in args.librispeech or []:
        fetch_librispeech(name)
    if args.mcif:
        print(MCIF_INSTRUCTIONS, flush=True)


if __name__ == "__main__":
    main()
