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
    "n_clips": 300,                        # first N clips per language, in manifest order (int, or {lang: N})
    "clip_set": "main",                    # main = the original test clips; confirm = fresh clips (EXP-007 onwards)
    "batch_size": 8,
    "device": "auto",                      # auto | cpu | cuda
}


def phases(cfg: dict) -> list[str]:
    return list(cfg["langs"])


def n_clips(cfg: dict, lang: str) -> int:
    n = cfg["n_clips"]
    return int(n[lang] if isinstance(n, dict) else n)


def clips_for(manifest: pd.DataFrame, lang: str, n: int, clip_set: str = "main") -> pd.DataFrame:
    """First n clips of one language in one clip set. Manifests built before clip sets existed are all "main"."""
    sets = manifest["clip_set"].fillna("main") if "clip_set" in manifest.columns else pd.Series("main", index=manifest.index)
    g = manifest[(manifest["lang"] == lang) & (sets == clip_set)].head(n)
    if len(g) < n:
        raise RuntimeError(f"{lang}: only {len(g)} clips in clip set {clip_set!r}, config asks for {n}")
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


def cap_tokens(seconds: float, rate: float, margin: int = 4) -> int:
    """Most new tokens a system may write for a clip of `seconds` (EXP-006 output-length cap)."""
    return int(np.ceil(rate * seconds)) + margin


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
        import torch
        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", model=sys_cfg["model"], device=0 if device == "cuda" else -1,
                       torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        lang = sys_cfg["lang"]
        gc = asr.model.generation_config
        if lang.startswith("<|") and lang not in (gc.lang_to_id or {}):
            # Fine-tunes that add languages (Paza adds <|kik|>, <|luo|>, ...) put the token in the tokenizer but not in
            # generation_config.lang_to_id, and transformers refuses a language it cannot map. Register it.
            tid = asr.tokenizer.convert_tokens_to_ids(lang)
            if tid is None or tid == asr.tokenizer.unk_token_id:
                raise ValueError(f"{sys_cfg['model']} has no language token {lang}")
            gc.lang_to_id = {**(gc.lang_to_id or {}), lang: int(tid)}

        gen = {"language": sys_cfg["lang"], "task": "transcribe"}
        rate = sys_cfg.get("tokens_per_sec")

        def run(batch):
            if rate is None:
                out = asr([{"raw": x, "sampling_rate": P.SR} for x in batch], generate_kwargs=gen,
                          batch_size=len(batch))
                return [o["text"] for o in out]
            # Output-length cap (EXP-006): at most ceil(rate * seconds) + margin new tokens per clip, with the rate set
            # from training transcripts (scripts/calibrate_cap.py), never from the test references. Per clip, because
            # the cap differs by clip length.
            res = []
            for x in batch:
                cap = cap_tokens(len(x) / P.SR, rate, int(sys_cfg.get("cap_margin", 4)))
                res.append(asr({"raw": x, "sampling_rate": P.SR}, generate_kwargs={**gen, "max_new_tokens": cap})["text"])
            return res

        return run
    if fam == "gemma":
        # Gemma 4 (E2B / E4B / 12B take audio, up to 30 s). Prompt from the model card, plus the language name so
        # that, like MMS (adapter) and Whisper (language token), the model is told which language it hears.
        # Needs transformers >= 5.5 (AutoModelForMultimodalLM); the molab runner upgrades it when missing.
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        proc = AutoProcessor.from_pretrained(sys_cfg["model"])
        model = AutoModelForMultimodalLM.from_pretrained(sys_cfg["model"], dtype="auto", device_map="auto").eval()
        prompt = (f"Transcribe the following speech segment in its original language ({sys_cfg['lang_name']}). "
                  "Follow these specific instructions for formatting the answer:\n"
                  "* Only output the transcription, with no newlines.\n"
                  "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, "
                  "and write 3 instead of three.")

        def run(batch):
            out = []
            for x in batch:   # one clip per call: generation with padded audio batches is not worth the risk here
                msgs = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                     {"type": "audio", "audio": np.asarray(x, dtype=np.float32)}]}]
                inp = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                               add_generation_prompt=True).to(model.device)
                n = inp["input_ids"].shape[-1]
                with torch.inference_mode():
                    ids = model.generate(**inp, max_new_tokens=int(sys_cfg.get("max_new_tokens", 256)), do_sample=False)
                out.append(proc.decode(ids[0][n:], skip_special_tokens=True).strip().replace("\n", " "))
            return out

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
