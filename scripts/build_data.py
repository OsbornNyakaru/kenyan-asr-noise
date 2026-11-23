"""build_data.py — choose the test clips, resample everything to 16 kHz, write data/raw/, and (with --push) upload it
to the PRIVATE Hugging Face dataset the molab runner reads.

Sources per language, as lang=SPEC (repeat --source):
    hf:<dataset>:<config>:<split>      a Hugging Face dataset with audio + text columns, e.g. hf:google/fleurs:sw_ke:test
    cv:<folder>                        an extracted Mozilla Common Voice language folder (test.tsv + clips/)

    python scripts/build_data.py --source eng=hf:google/fleurs:en_us:test --source swh=hf:google/fleurs:sw_ke:test \\
        --source kik=cv:~/Downloads/cv-corpus-ki --noise ~/Downloads/mill.wav --n 300 [--push]

DECIDE (Osborn): the sources. FLEURS for English and Kiswahili with Common Voice for Kikuyu mixes read-sentence
corpora with different recording set-ups; if a single source covers all three languages, prefer it and say so.

Selection is deterministic: clips of 2 to 15 s, at most --per-speaker clips per speaker (Common Voice has a speaker
id; FLEURS does not), shuffled with seed 42, first N kept. Voice recordings identify people, so the data repo must stay
private even when the source licence (CC0, CC-BY) would allow sharing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
import prepare as P  # noqa: E402

RAW = P.RAW
SEED = 42


def resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == P.SR:
        return x.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, P.SR)
    return resample_poly(x, P.SR // g, sr // g).astype(np.float32)


def _candidates_hf(dataset: str, config: str, split: str):
    from datasets import Audio, load_dataset

    ds = load_dataset(dataset, config, split=split).cast_column("audio", Audio(sampling_rate=P.SR))
    text_col = next(c for c in ("transcription", "raw_transcription", "sentence", "text") if c in ds.column_names)
    for i, r in enumerate(ds):
        yield {"key": f"{config}-{r.get('id', i)}-{i}", "audio": np.asarray(r["audio"]["array"], dtype=np.float32),
               "sr": P.SR, "text": r[text_col], "speaker": str(r.get("speaker_id", "")), "gender": str(r.get("gender", "")),
               "source": f"hf:{dataset}:{config}:{split}"}


def _candidates_fleurs(config: str, splits: str):
    """google/fleurs from its parquet files (the loading script no longer runs on current `datasets`).
    `splits` is test, validation or test+validation. FLEURS has no speaker id; gender is FLEURS's integer code."""
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    for split in splits.split("+"):
        p = hf_hub_download("google/fleurs", f"parquet-data/{config}/{split}-00000-of-00001.parquet", repo_type="dataset")
        t = pq.read_table(p)
        for r in t.to_pylist():
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
            yield {"key": f"fleurs-{config}-{split}-{r['id']}-{r['audio']['path']}", "audio": x.mean(1), "sr": sr,
                   "text": r["raw_transcription"], "speaker": "", "gender": str(r["gender"]),
                   "source": f"fleurs:{config}:{split}"}


ANV_REPO = "badrex/anv_data_ke_kikuyu_scripted"   # public copy of African Next Voices Kenya, Kikuyu scripted


def _candidates_anv(split: str = "test"):
    """African Next Voices Kenya (KenCorpus Consortium; Kikuyu by DeKUT and LDRI), Kikuyu scripted speech,
    licence CC BY 4.0. Read from a public copy, by Osborn's decision (2026-11-23), while access to the gated original
    (Anv-ke/kikuyu) is pending. The original's ethical-use terms (no profiling, respect contributors' consent) are
    followed. recorder_uuid is the speaker. When official access arrives, rebuild and compare audio_id (kept in key)."""
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(ANV_REPO, repo_type="dataset")
                   if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    for f in files:
        t = pq.read_table(hf_hub_download(ANV_REPO, f, repo_type="dataset"),
                          columns=["audio", "audio_id", "recorder_uuid", "transcription", "domain"])
        for r in t.to_pylist():
            if not r["audio"] or not r["audio"].get("bytes"):   # a few rows in the copy carry no audio
                continue
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
            yield {"key": f"anv-kik-{r['audio_id']}", "audio": x.mean(1), "sr": sr, "text": r["transcription"],
                   "speaker": str(r["recorder_uuid"]), "gender": "", "source": f"anv:{ANV_REPO}:{split}:{r['domain']}"}


def _candidates_cv(folder: str):
    import soundfile as sf

    root = Path(folder).expanduser()
    tsv = pd.read_csv(root / "test.tsv", sep="\t", quoting=3, dtype=str)
    for r in tsv.itertuples():
        x, sr = sf.read(root / "clips" / r.path, dtype="float32", always_2d=True)
        yield {"key": r.path, "audio": x.mean(1), "sr": sr, "text": r.sentence, "speaker": r.client_id[:12],
               "gender": getattr(r, "gender", "") or "", "source": f"cv:{root.name}"}


def build_lang(lang: str, src: str, n: int, per_speaker: int, max_dur: float = 15.0) -> pd.DataFrame:
    kind, _, rest = src.partition(":")
    gen = {"hf": lambda: _candidates_hf(*rest.split(":")), "fleurs": lambda: _candidates_fleurs(*rest.split(":")),
           "anv": lambda: _candidates_anv(rest or "test"), "cv": lambda: _candidates_cv(rest)}[kind]()
    rows = []
    for c in gen:
        x = resample(c["audio"], c["sr"])
        dur = len(x) / P.SR
        if not (2.0 <= dur <= max_dur) or not str(c["text"]).strip() or P.power(x) == 0:
            continue
        c.update(audio=x, duration=round(dur, 2))
        rows.append(c)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    kept, per, seen = [], {}, set()
    for i in order:
        spk, txt = rows[i]["speaker"], P.normalise(rows[i]["text"])
        if txt in seen:          # one recording per sentence: FLEURS repeats sentences across speakers
            continue
        if spk and per.get(spk, 0) >= per_speaker:
            continue
        seen.add(txt)
        per[spk] = per.get(spk, 0) + 1
        kept.append(rows[i])
        if len(kept) == n:
            break
    if len(kept) < n:
        raise SystemExit(f"{lang}: only {len(kept)} usable clips from {src}, need {n}")
    import soundfile as sf

    out = []
    for c in kept:
        cid = f"{lang}_{hashlib.sha256(c['key'].encode()).hexdigest()[:10]}"
        rel = f"audio/{lang}/{cid}.flac"
        (RAW / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(RAW / rel, c["audio"], P.SR)
        out.append({"clip_id": cid, "lang": lang, "path": rel, "text": c["text"], "speaker": c["speaker"],
                    "gender": c["gender"], "duration": c["duration"], "source": c["source"]})
    return pd.DataFrame(out)


def main() -> int:
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="lang=SPEC, see the module docstring")
    ap.add_argument("--noise", action="append", default=[], help="noise recording(s); the first is saved as machine.wav")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--per-speaker", type=int, default=5)
    ap.add_argument("--max-dur", type=float, default=15.0, help="longest clip kept, seconds (FLEURS needs ~30)")
    ap.add_argument("--keep", action="store_true", help="keep other languages already in the manifest (build one at a time)")
    ap.add_argument("--push", action="store_true", help="upload data/raw to the private HF data repo")
    ap.add_argument("--repo", default=None)
    a = ap.parse_args()
    frames = []
    langs = [s.partition("=")[0] for s in a.source]
    if a.keep and (RAW / "manifest.csv").exists():
        old = pd.read_csv(RAW / "manifest.csv", dtype=str)
        frames.append(old[~old["lang"].isin(langs)])
    for s in a.source:
        lang, _, src = s.partition("=")
        frames.append(build_lang(lang, src, a.n, a.per_speaker, a.max_dur))
        print(f"{lang}: {a.n} clips from {src}")
    pd.concat(frames, ignore_index=True).to_csv(RAW / "manifest.csv", index=False)
    (RAW / "noise").mkdir(parents=True, exist_ok=True)
    if not a.noise and not (RAW / "noise" / "machine.wav").exists():
        sys.exit("no --noise given and no data/raw/noise/machine.wav yet")
    for i, f in enumerate(a.noise):
        x, sr = sf.read(Path(f).expanduser(), dtype="float32", always_2d=True)
        name = "machine.wav" if i == 0 else Path(f).stem + ".wav"
        sf.write(RAW / "noise" / name, resample(x.mean(1), sr), P.SR)
        print(f"noise/{name}: {len(x) / sr:.0f}s from {f}")
    P.load_manifest()   # schema check
    if a.push:
        return push(a.repo)
    return 0


def push(repo: str | None) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    repo = repo or f"{api.whoami()['name']}/kenyan-asr-noise-data"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        print(f"refusing to upload: {repo} is PUBLIC", file=sys.stderr)
        return 1
    files = sorted(p.relative_to(RAW).as_posix() for p in RAW.rglob("*") if p.is_file() and p.name != "manifest.json")
    (RAW / "manifest.json").write_text(json.dumps({f: hashlib.sha256((RAW / f).read_bytes()).hexdigest() for f in files}, indent=2))
    api.upload_folder(folder_path=str(RAW), repo_id=repo, repo_type="dataset", commit_message="upload clips, noise, manifest")
    print(f"{repo} (private): {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
