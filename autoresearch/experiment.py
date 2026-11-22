"""experiment.py — the one file an experiment edits. DEFAULT_CONFIG is the baseline for Project 1:
the same model family (MMS-1B-all, one language adapter per language) for Kikuyu, Kiswahili and English, so that a
difference between the noise curves is about the language, not about three different systems.

Work is split into phases (one per language) and, inside a phase, conditions (one per noise level), so
autoresearch/remote.py can upload each finished condition and resume after a molab disconnect.
A config key that is not used here is a bug: add it or drop it.

Day-1 check (2026-11-22, notebooks/analysis.py): facebook/mms-1b-all lists 1,198 language vocabularies and ships
adapters for kik, swh and eng, so one model family covers all three languages. It has no Ekegusii (guz) adapter, so
Ekegusii cannot be added to this comparison without changing systems.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import prepare as P

DEFAULT_CONFIG = {
    "systems": {
        "kik": {"family": "mms", "model": "facebook/mms-1b-all", "lang": "kik"},
        "swh": {"family": "mms", "model": "facebook/mms-1b-all", "lang": "swh"},
        "eng": {"family": "mms", "model": "facebook/mms-1b-all", "lang": "eng"},
    },
    "langs": ["kik", "swh", "eng"],
    "snrs": [None, 20, 10, 5, 0],          # None = clean
    "noise": "machine.wav",                # file in noise/ of the data repo
    "n_clips": 300,                        # first N clips per language, in manifest order
    "batch_size": 8,
    "device": "auto",                      # auto | cpu | cuda
}


def phases(cfg: dict) -> list[str]:
    return list(cfg["langs"])


def clips_for(manifest: pd.DataFrame, lang: str, n: int) -> pd.DataFrame:
    g = manifest[manifest["lang"] == lang].head(n)
    if len(g) < n:
        raise RuntimeError(f"{lang}: only {len(g)} clips in manifest, config asks for {n}")
    return g


# ── systems ──────────────────────────────────────────────────────────────────


def _device(cfg: dict) -> str:
    if cfg.get("device", "auto") != "auto":
        return cfg["device"]
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_system(sys_cfg: dict, device: str):
    """A callable: list of 16 kHz float32 arrays -> list of transcripts."""
    fam = sys_cfg["family"]
    if fam == "oracle":          # pipeline smoke test only: returns the reference text it is handed
        return None
    if fam == "mms":
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        proc = AutoProcessor.from_pretrained(sys_cfg["model"], target_lang=sys_cfg["lang"])
        model = Wav2Vec2ForCTC.from_pretrained(sys_cfg["model"], target_lang=sys_cfg["lang"],
                                               ignore_mismatched_sizes=True).to(device).eval()

        def run(batch):
            inp = proc(batch, sampling_rate=P.SR, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                ids = model(**inp).logits.argmax(-1)
            return proc.batch_decode(ids)

        return run
    if fam == "whisper":
        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", model=sys_cfg["model"], device=0 if device == "cuda" else -1)

        def run(batch):
            out = asr([{"raw": x, "sampling_rate": P.SR} for x in batch],
                      generate_kwargs={"language": sys_cfg["lang"], "task": "transcribe"}, batch_size=len(batch))
            return [o["text"] for o in out]

        return run
    raise ValueError(f"unknown family {fam}")


def mms_languages(model: str = "facebook/mms-1b-all") -> list[str]:
    """Language codes with a tokenizer vocabulary in an MMS checkpoint (the day-1 coverage check)."""
    import json

    from huggingface_hub import hf_hub_download

    return sorted(json.loads(open(hf_hub_download(model, "vocab.json"), encoding="utf-8").read()))


# ── one condition ────────────────────────────────────────────────────────────


def run_condition(cfg: dict, lang: str, snr, clips: pd.DataFrame, noise: np.ndarray, transcribe) -> pd.DataFrame:
    """Transcribe every clip of one language at one noise level."""
    refs, hyps = clips["text"].tolist(), []
    bs = int(cfg["batch_size"])
    for i in range(0, len(clips), bs):
        part = clips.iloc[i:i + bs]
        audio = [P.mix(P.load_audio(p), noise, snr, cid) for p, cid in zip(part["path"], part["clip_id"])]
        hyps += part["text"].tolist() if transcribe is None else transcribe(audio)
    return pd.DataFrame({"clip_id": clips["clip_id"].to_numpy(), "lang": lang, "snr_tag": P.snr_tag(snr),
                         "ref": refs, "hyp": hyps})
