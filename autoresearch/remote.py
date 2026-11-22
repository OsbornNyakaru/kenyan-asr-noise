"""remote.py — runs one queued spec, language by language and noise level by noise level. Called by
notebooks/molab_runner.py from the repository as it was at the spec's git commit, so a run always uses the code the
spec was written against.

    store   where results go: exists(name), get(name) -> local path, put({name: bytes | str | DataFrame})
            (names are relative to runs/<EXP>/ in the private Hugging Face runs repo)
    log     callable(str)

Each finished condition is uploaded at once as <lang>/<snr>.csv (the transcripts), so a run that dies resumes at the
next condition. After a language: <lang>.json (its metrics). At the end: summary.csv (all languages).

Locally: `python autoresearch/remote.py EXP-001 --clips 5 --set 'systems.kik.family="oracle"'` runs the same code
into experiments/runs/<EXP>-local/ (a smoke test; not recorded).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import experiment as E  # noqa: E402
import prepare as P  # noqa: E402


def run_spec(spec: dict, store, log=print, max_minutes: float | None = None, **_) -> dict:
    t0 = time.time()
    deadline = t0 + 60 * float(max_minutes or spec.get("max_minutes") or 10_000)
    cfg = spec["config"]
    manifest = P.load_manifest()
    noise = P.load_noise(cfg["noise"])
    device = E._device(cfg)
    log(f"{len(cfg['langs'])} languages x {len(cfg['snrs'])} noise levels x {cfg['n_clips']} clips on {device}")

    summary = {}
    for lang in E.phases(cfg):
        if store.exists(f"{lang}.json"):
            summary[lang] = json.loads(Path(store.get(f"{lang}.json")).read_text())
            log(f"{lang}: already done, skipping")
            continue
        clips = E.clips_for(manifest, lang, int(cfg["n_clips"]))
        todo = [s for s in cfg["snrs"] if not store.exists(f"{lang}/{P.snr_tag(s)}.csv")]
        transcribe = None
        if todo:
            if time.time() > deadline:
                log(f"{lang}: time limit reached")
                break
            tl = time.time()
            transcribe = E.load_system(cfg["systems"][lang], device)
            log(f"{lang}: {cfg['systems'][lang]['family']} {cfg['systems'][lang].get('model', '')} loaded "
                f"in {time.time() - tl:.0f}s")
        for snr in cfg["snrs"]:
            name = f"{lang}/{P.snr_tag(snr)}.csv"
            if store.exists(name):
                continue
            tc = time.time()
            df = E.run_condition(cfg, lang, snr, clips, noise, transcribe)
            store.put({name: df})
            m = P.metrics(pd.DataFrame([P.clip_counts(a, b) for a, b in zip(df["ref"], df["hyp"])]))
            log(f"{lang} {P.snr_tag(snr)}: WER {m['wer']:.3f}, CER {m['cer']:.3f}, short-word recall "
                f"{m['short_recall']:.3f} ({time.time() - tc:.0f}s)")
        del transcribe
        parts = [pd.read_csv(store.get(f"{lang}/{P.snr_tag(s)}.csv"), dtype={"clip_id": str}, keep_default_na=False)
                 for s in cfg["snrs"]]
        tab = P.summarise(pd.concat(parts, ignore_index=True))
        summary[lang] = json.loads(tab.to_json(orient="records"))
        store.put({f"{lang}.json": json.dumps(summary[lang], indent=2)})
    done = [lang for lang in E.phases(cfg) if lang in summary]
    if len(done) == len(E.phases(cfg)):
        store.put({"summary.csv": pd.DataFrame([r for lang in done for r in summary[lang]])})
    return {"phases": summary, "status": "ok" if len(done) == len(E.phases(cfg)) else "incomplete",
            "harness": P.harness_hash(), "seconds": round(time.time() - t0, 1)}


class LocalStore:
    """Writes into a local folder (dry runs)."""

    def __init__(self, folder: Path):
        self.d = folder
        self.d.mkdir(parents=True, exist_ok=True)

    def exists(self, name):
        return (self.d / name).exists()

    def get(self, name):
        return str(self.d / name)

    def put(self, files: dict):
        for name, v in files.items():
            (self.d / name).parent.mkdir(parents=True, exist_ok=True)
            if isinstance(v, pd.DataFrame):
                v.to_csv(self.d / name, index=False)
            elif isinstance(v, bytes):
                (self.d / name).write_bytes(v)
            else:
                (self.d / name).write_text(v)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--clips", type=int, help="first N clips per language (smoke test)")
    ap.add_argument("--set", action="append", default=[], help="config override key=value (JSON value), "
                    "e.g. 'systems.kik.family=\"oracle\"'")
    a = ap.parse_args()
    root = HERE.parent
    s = json.loads((root / "experiments" / "specs" / f"{a.exp}.json").read_text())
    for kv in a.set:
        k, _, v = kv.partition("=")
        *path, last = k.split(".")
        node = s["config"]
        for p in path:
            node = node[p]
        node[last] = json.loads(v)
    if a.clips:
        s["config"]["n_clips"] = a.clips
    out = root / "experiments" / "runs" / f"{a.exp}-local"
    print(json.dumps(run_spec(s, LocalStore(out)), indent=2, default=str))
