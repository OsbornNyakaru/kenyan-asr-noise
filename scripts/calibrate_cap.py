"""calibrate_cap.py — the output-length cap for EXP-006, set from TRAINING transcripts only.

EXP-005 showed Paza Whisper transcribing a short clip and then writing on, inventing text, until its length limit. EXP-006
caps how many tokens it may write for a clip: ceil(rate * clip seconds) + MARGIN, where `rate` is a high percentile of
tokens per second (Paza tokenizer) in speech the test set does not contain:

    kik   ANV Kikuyu scripted, train split (other speakers: 0/23 of our test speakers are in ANV train)
    swh   FLEURS sw_ke train.tsv   (our clips come from FLEURS test and validation)
    eng   FLEURS en_us train.tsv

No test reference is read. Output: experiments/CAP_CALIBRATION.json (rates for the spec) and .md (the table).

    python scripts/calibrate_cap.py [--pct 99] [--anv-files 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_data import ANV_REPO  # noqa: E402

MODEL = "microsoft/paza-whisper-large-v3-turbo"
MARGIN = 4      # tokens on top of the rate, so a 1-2 s clip can still be transcribed


def fleurs(config: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("google/fleurs", f"data/{config}/train.tsv", repo_type="dataset")
    t = pd.read_csv(p, sep="\t", header=None, quoting=3, dtype=str,
                    names=["id", "file", "raw", "text", "phon", "samples", "gender"])
    return pd.DataFrame({"text": t["raw"], "dur": t["samples"].astype(float) / 16_000})


def anv(n_files: int) -> pd.DataFrame:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    fs = HfFileSystem()
    files = sorted(f for f in HfApi().list_repo_files(ANV_REPO, repo_type="dataset")
                   if f.startswith("data/train-") and f.endswith(".parquet"))
    step = max(len(files) // n_files, 1)
    frames = []
    for f in files[::step][:n_files]:     # spread over the shards; metadata columns only, no audio
        with fs.open(f"datasets/{ANV_REPO}/{f}", "rb") as fh:
            t = pq.read_table(fh, columns=["transcription", "audio_duration"]).to_pandas()
        frames.append(t.rename(columns={"transcription": "text", "audio_duration": "dur"}))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, default=99.0)
    ap.add_argument("--anv-files", type=int, default=6)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    sources = {"kik": (lambda: anv(a.anv_files), f"{ANV_REPO} train, {a.anv_files} shards"),
               "swh": (lambda: fleurs("sw_ke"), "google/fleurs sw_ke train.tsv"),
               "eng": (lambda: fleurs("en_us"), "google/fleurs en_us train.tsv")}
    out = {"model": MODEL, "percentile": a.pct, "margin_tokens": MARGIN, "langs": {}}
    lines = ["# Output-length cap for EXP-006 (calibrated on training transcripts only)", "",
             f"cap(clip) = ceil(rate × clip seconds) + {MARGIN} tokens, rate = {a.pct:g}th percentile of "
             f"tokens per second ({MODEL} tokenizer).", "",
             "| lang | source | utterances | median tok/s | rate (cap) | cap for 4 s | cap for 10 s |",
             "|---|---|--:|--:|--:|--:|--:|"]
    for lang, (load, desc) in sources.items():
        d = load().dropna()
        d = d[(d["dur"] >= 1.0) & d["text"].str.strip().astype(bool)]
        n_tok = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in d["text"]])
        r = n_tok / d["dur"].to_numpy()
        rate = float(np.percentile(r, a.pct))
        out["langs"][lang] = {"rate": round(rate, 3), "source": desc, "n": int(len(d)),
                              "median": round(float(np.median(r)), 3)}
        lines.append(f"| {lang} | {desc} | {len(d)} | {np.median(r):.2f} | {rate:.2f} | "
                     f"{int(np.ceil(rate * 4)) + MARGIN} | {int(np.ceil(rate * 10)) + MARGIN} |")
        print(lines[-1], flush=True)
    (ROOT / "experiments" / "CAP_CALIBRATION.json").write_text(json.dumps(out, indent=2) + "\n")
    (ROOT / "experiments" / "CAP_CALIBRATION.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
