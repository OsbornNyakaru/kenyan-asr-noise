"""prepare.py — FROZEN harness for Project 1: do command words survive machine noise in Kikuyu?

Do not edit: the molab runner refuses a run whose prepare.py does not hash to PREPARE_SHA256 in
notebooks/molab_runner.py, and exp.py refuses to compare results made under different hashes. Everything that decides
what a number means lives here: the clips, how noise is mixed in, how text is normalised, and how errors are counted.

Data (private Hugging Face dataset, mirrored to data/raw/ by the runner):
    manifest.csv   clip_id, lang, path, text, speaker, gender, duration, source   (built by scripts/build_data.py)
    audio/<lang>/<clip_id>.flac   16 kHz mono
    noise/<name>.wav              16 kHz mono machine noise

Metrics:
    WER   word error rate, corpus level: total word edits / total reference words
    CER   character error rate with spaces removed ("no-space CER", as in Gatere 2026), corpus level
    short-word recall   share of reference words of 1-2 syllables that appear in the hypothesis (multiset match).
          Syllables are counted as vowel groups, which suits Bantu CV syllables; it is a proxy, not a phonology.
    relative degradation   WER at a noise level / WER on clean audio, per language, with a paired bootstrap CI
          (the same clips are resampled for both conditions).

Harness v2 (2026-11-23): normalise() maps ī → ĩ and ū → ũ (corrupted Kikuyu vowels found in 19 ANV references; the
AfriVoices hackathon reported the same corruption) and the modifier apostrophe ʼ (U+02BC, emitted by MMS) to '.
Only scoring changed; v1 transcripts are re-scored with scripts/rescore.py.
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SR = 16_000
LANGS = ("kik", "swh", "eng")            # Kikuyu, Kiswahili, English (ISO 639-3)
MANIFEST_COLS = ["clip_id", "lang", "path", "text", "speaker", "gender", "duration", "source"]
SHORT_MAX_SYLLABLES = 2                  # "short word" = 1 or 2 syllables, what spoken commands look like
BOOTSTRAP_B, BOOTSTRAP_SEED = 2000, 42
VOWELS = set("aeiou")
HARNESS_VERSION = 2
# Applied after lower-casing: corrupted Kikuyu macron vowels to the tilde vowels, and the modifier apostrophe to '.
CHAR_FIXES = str.maketrans({"ī": "ĩ", "ū": "ũ", "ʼ": "'"})
VALIDATION = ("Fixed clip set from manifest.csv; noise mixed deterministically per clip; corpus-level WER and "
              "no-space CER; 95% bootstrap CIs over clips (B=2000, seed 42), paired across noise levels.")


# ── data ─────────────────────────────────────────────────────────────────────


def load_manifest() -> pd.DataFrame:
    m = pd.read_csv(RAW / "manifest.csv", dtype={"clip_id": str, "speaker": str})
    missing = [c for c in MANIFEST_COLS if c not in m.columns]
    if missing:
        raise RuntimeError(f"manifest.csv is missing columns {missing}")
    if m["clip_id"].duplicated().any():
        raise RuntimeError("duplicate clip_id in manifest.csv")
    bad = sorted(set(m["lang"]) - set(LANGS))
    if bad:
        raise RuntimeError(f"unknown languages {bad}; expected {LANGS}")
    return m.reset_index(drop=True)


def load_audio(rel_path: str) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(RAW / rel_path, dtype="float32", always_2d=True)
    if sr != SR:
        raise RuntimeError(f"{rel_path}: {sr} Hz, expected {SR} (build_data.py resamples)")
    return x.mean(axis=1)


def load_noise(name: str) -> np.ndarray:
    return load_audio(f"noise/{name}")


# ── noise mixing ─────────────────────────────────────────────────────────────


def _seed_for(clip_id: str) -> int:
    return int(hashlib.sha256(clip_id.encode()).hexdigest()[:8], 16)


def power(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def mix(clean: np.ndarray, noise: np.ndarray, snr_db: float | None, clip_id: str) -> np.ndarray:
    """Clean speech plus noise at snr_db (signal power / noise power, in dB). snr_db None = the clean clip.

    The noise segment starts at an offset fixed by clip_id, so every run mixes the same noise into the same clip.
    If the mixture would clip, speech and noise are scaled down together, which keeps the SNR exact.
    """
    clean = np.asarray(clean, dtype=np.float32)
    if snr_db is None:
        return clean.copy()
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))) + 1)
    start = _seed_for(clip_id) % (len(noise) - len(clean) + 1)
    seg = noise[start:start + len(clean)].astype(np.float64)
    ps, pn = power(clean), power(seg)
    if ps == 0 or pn == 0:
        raise ValueError(f"{clip_id}: silent speech or silent noise segment")
    seg *= np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    out = clean.astype(np.float64) + seg
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out *= 0.99 / peak
    return out.astype(np.float32)


def snr_tag(snr_db: float | None) -> str:
    return "clean" if snr_db is None else f"{snr_db:g}dB"


# ── text ─────────────────────────────────────────────────────────────────────


def normalise(text: str) -> str:
    """Lower case, Unicode NFC, punctuation to spaces, single spaces. Diacritics are KEPT: Kikuyu ĩ and ũ are
    different vowels from i and u, so stripping them would hide real errors."""
    text = unicodedata.normalize("NFC", str(text)).lower().translate(CHAR_FIXES)
    out = []
    for i, ch in enumerate(text):
        if unicodedata.category(ch).startswith("P"):
            inside_word = (ch in "'’" and 0 < i < len(text) - 1 and text[i - 1].isalpha() and text[i + 1].isalpha())
            out.append("'" if inside_word else " ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def syllables(word: str) -> int:
    base = "".join(c for c in unicodedata.normalize("NFD", word) if not unicodedata.combining(c))
    count, prev = 0, False
    for c in base:
        v = c in VOWELS
        count += v and not prev
        prev = v
    return max(count, 1)


# ── errors ───────────────────────────────────────────────────────────────────


def edit_distance(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def clip_counts(ref: str, hyp: str) -> dict:
    """Per-clip error counts. Corpus metrics are ratios of sums, so they are computed from these."""
    r, h = normalise(ref).split(), normalise(hyp).split()
    rc, hc = "".join(r), "".join(h)
    short = [w for w in r if syllables(w) <= SHORT_MAX_SYLLABLES]
    pool = {}
    for w in h:
        pool[w] = pool.get(w, 0) + 1
    hits = 0
    for w in short:
        if pool.get(w, 0) > 0:
            pool[w] -= 1
            hits += 1
    return {"w_err": edit_distance(r, h), "w_ref": len(r), "c_err": edit_distance(list(rc), list(hc)),
            "c_ref": len(rc), "short_hit": hits, "short_ref": len(short)}


def _ratio(num: np.ndarray, den: np.ndarray) -> float:
    d = den.sum()
    return float(num.sum() / d) if d else float("nan")


def metrics(counts: pd.DataFrame) -> dict:
    return {"wer": _ratio(counts["w_err"].to_numpy(), counts["w_ref"].to_numpy()),
            "cer": _ratio(counts["c_err"].to_numpy(), counts["c_ref"].to_numpy()),
            "short_recall": _ratio(counts["short_hit"].to_numpy(), counts["short_ref"].to_numpy()),
            "n_clips": int(len(counts))}


def _boot_idx(n: int) -> np.ndarray:
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(BOOTSTRAP_B, n))


def _ci(samples: np.ndarray) -> list[float]:
    s = samples[np.isfinite(samples)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))] if len(s) else [float("nan")] * 2


def summarise(transcripts: pd.DataFrame) -> pd.DataFrame:
    """One row per (lang, snr): WER, CER, short-word recall with 95% CIs, and WER relative to that language's clean
    WER with a paired CI. `transcripts` has clip_id, lang, snr_tag, ref, hyp."""
    t = transcripts.copy()
    c = pd.DataFrame([clip_counts(a, b) for a, b in zip(t["ref"], t["hyp"])], index=t.index)
    t = pd.concat([t, c], axis=1)
    rows = []
    for lang, g in t.groupby("lang", sort=False):
        piv = {tag: gg.set_index("clip_id").sort_index() for tag, gg in g.groupby("snr_tag", sort=False)}
        clean = piv.get("clean")
        for tag, gg in piv.items():
            n = len(gg)
            idx = _boot_idx(n)
            w = gg["w_err"].to_numpy()[idx].sum(1) / np.maximum(gg["w_ref"].to_numpy()[idx].sum(1), 1)
            ch = gg["c_err"].to_numpy()[idx].sum(1) / np.maximum(gg["c_ref"].to_numpy()[idx].sum(1), 1)
            sh_den = gg["short_ref"].to_numpy()[idx].sum(1)
            sh = np.where(sh_den > 0, gg["short_hit"].to_numpy()[idx].sum(1) / np.maximum(sh_den, 1), np.nan)
            row = {"lang": lang, "snr": tag, **metrics(gg), "wer_ci": _ci(w), "cer_ci": _ci(ch), "short_recall_ci": _ci(sh)}
            if clean is not None:
                common = gg.index.intersection(clean.index)
                a, b = gg.loc[common], clean.loc[common]
                pi = _boot_idx(len(common))
                num = a["w_err"].to_numpy()[pi].sum(1) / np.maximum(a["w_ref"].to_numpy()[pi].sum(1), 1)
                den = b["w_err"].to_numpy()[pi].sum(1) / np.maximum(b["w_ref"].to_numpy()[pi].sum(1), 1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = num / den
                base = metrics(b)["wer"]
                row["rel_wer"] = metrics(a)["wer"] / base if base else float("nan")
                row["rel_wer_ci"] = _ci(rel)
            rows.append(row)
    return pd.DataFrame(rows)


def harness_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
