"""check_public.py — refuse to commit recordings or transcripts to the public repository.

The GitHub repository is public (molab syncs the runner notebook from it). Voice recordings identify the people who
made them, so audio, manifests and run outputs stay in the private Hugging Face datasets. This blocks audio and data
files and anything under data/, experiments/runs/ or submissions/.

    python scripts/check_public.py            # the files staged for commit (the pre-commit hook runs this)
    python scripts/check_public.py --all      # every tracked file
Install the hook once with `make hooks`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_SUFFIX = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".csv", ".tsv", ".parquet", ".npz", ".npy",
                 ".zip", ".tar", ".gz", ".pt", ".pkl", ".safetensors"}
BANNED_DIRS = ("data/", "experiments/runs/", "submissions/", "runner_work/")


def git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [x for x in out.split("\0") if x]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    files = git("ls-files", "-z") if a.all else git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    bad = 0
    for f in files:
        why = ("lives in a data folder" if f.startswith(BANNED_DIRS)
               else f"{Path(f).suffix} files may hold recordings or transcripts" if Path(f).suffix.lower() in BANNED_SUFFIX
               else None)
        if why:
            print(f"BLOCKED {f}: {why}", file=sys.stderr)
            bad += 1
    if bad:
        print(f"check_public: {bad} problem(s). The repository is public; move data to the private HF dataset.", file=sys.stderr)
        return 1
    print(f"check_public: {len(files)} file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
