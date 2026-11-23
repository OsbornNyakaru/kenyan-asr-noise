"""anv_overlap.py — do the Kikuyu test clips overlap with African Next Voices training or validation data?

Any model or language model trained on ANV (for example microsoft/paza-whisper-large-v3-turbo, trained on ANV Kenya)
could have heard our test speakers or read our test sentences. This reads only the metadata columns (audio_id,
recorder_uuid, transcription, split) of the public ANV Kikuyu copy and writes:

    experiments/ANV_OVERLAP.md                    the counts
    experiments/anv_overlap_clips.json            clip_ids whose SENTENCE appears in ANV train (report them apart)

    python scripts/anv_overlap.py [--cache path/to/anv_meta.parquet]
Takes ~8 minutes without a cache (40 parquet files, metadata columns only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
sys.path.insert(0, str(ROOT / "scripts"))
import prepare as P  # noqa: E402
from build_data import ANV_REPO  # noqa: E402


def read_meta() -> pd.DataFrame:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    fs, rows = HfFileSystem(), []
    for f in sorted(f for f in HfApi().list_repo_files(ANV_REPO, repo_type="dataset") if f.endswith(".parquet")):
        with fs.open(f"datasets/{ANV_REPO}/{f}", "rb", block_size=2**20) as fh:
            df = pq.ParquetFile(fh).read(columns=["audio_id", "recorder_uuid", "transcription", "split"]).to_pandas()
        df["file"] = f.split("/")[1].split("-")[0]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache")
    a = ap.parse_args()
    meta = pd.read_parquet(a.cache) if a.cache and Path(a.cache).exists() else read_meta()
    meta["norm"] = meta["transcription"].fillna("").map(P.normalise)
    m = P.load_manifest()
    k = m[m["lang"] == "kik"].copy()
    k["norm"] = k["text"].map(P.normalise)
    lines = ["# Kikuyu test clips vs African Next Voices (ANV) train and validation", "",
             f"Source: public copy `{ANV_REPO}` (metadata columns only). Test set: {len(k)} clips, "
             f"{k['speaker'].nunique()} speakers.", "",
             "| ANV part | rows | speakers | our speakers found | our clips by those speakers | our sentences found |",
             "|---|--:|--:|--:|--:|--:|"]
    for part in ("train", "validation"):
        b = meta[meta["file"] == part]
        sp = set(k["speaker"]) & set(b["recorder_uuid"].astype(str))
        tx = set(k["norm"]) & set(b["norm"])
        lines.append(f"| {part} | {len(b)} | {b['recorder_uuid'].nunique()} | {len(sp)}/{k['speaker'].nunique()} | "
                     f"{k['speaker'].isin(sp).sum()}/{len(k)} | {len(tx)}/{len(k)} |")
    splits = meta.groupby(["file", "split"]).size()
    lines += ["", "Split labels in the copy: " + ", ".join(f"{f}/{s} {n}" for (f, s), n in splits.items()) + ".",
              "The copy's validation files contain the dev_test rows again, so validation must never be used for "
              "training or model selection here.", ""]
    seen = k[k["norm"].isin(set(meta.loc[meta["file"] == "train", "norm"]))]["clip_id"].tolist()
    lines.append(f"{len(seen)} test clips have a sentence that also appears in ANV train (read by other speakers). "
                 "Report Kikuyu results on all clips and on the remaining unseen-sentence clips, and leave these "
                 "sentences out of any language-model or fine-tuning text.")
    (ROOT / "experiments" / "ANV_OVERLAP.md").write_text("\n".join(lines) + "\n")
    (ROOT / "experiments" / "anv_overlap_clips.json").write_text(json.dumps(sorted(seen), indent=1) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
