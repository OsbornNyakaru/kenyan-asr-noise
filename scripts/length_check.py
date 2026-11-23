"""length_check.py — is the "same damage from noise" finding an artefact of clip length?

Kikuyu clips (ANV) are short (median 4.2 s), English (FLEURS) 9.2 s and Kiswahili (FLEURS) 13.4 s. This post-hoc check,
added after EXP-001, asks whether the absolute rise in error rate from clean audio to a noise level still looks alike
across languages when length is controlled, two ways:

  1. common band: only clips of BAND_LO-BAND_HI seconds, in every language;
  2. within language: each language split at its own median length into a shorter and a longer half.

Measures: corpus WER, no-space CER and short-word recall from autoresearch/prepare.py (the frozen harness), and the rise
(noisy minus clean) with a paired 95% bootstrap CI over clips (same clips resampled for both conditions).

    python scripts/length_check.py [--exp EXP-001] [--snr 0dB]   -> experiments/LENGTH_CHECK_<EXP>_<SNR>.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
import prepare as P  # noqa: E402

BAND_LO, BAND_HI = 5.0, 12.0


def counts(exp: str, lang: str, tag: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "experiments" / "runs" / exp / lang / f"{tag}.csv", dtype={"clip_id": str},
                    keep_default_na=False)
    c = pd.DataFrame([P.clip_counts(a, b) for a, b in zip(d["ref"], d["hyp"])])
    c["clip_id"] = d["clip_id"].to_numpy()
    return c.set_index("clip_id")


def rates(c: pd.DataFrame, idx: np.ndarray | None = None) -> dict:
    s = c.iloc[idx] if idx is not None else c
    return {"wer": s["w_err"].sum() / max(s["w_ref"].sum(), 1), "cer": s["c_err"].sum() / max(s["c_ref"].sum(), 1),
            "short": s["short_hit"].sum() / max(s["short_ref"].sum(), 1)}


def rise(clean: pd.DataFrame, noisy: pd.DataFrame, ids) -> dict:
    a, b = clean.loc[ids], noisy.loc[ids]
    base, top = rates(a), rates(b)
    rng = np.random.default_rng(P.BOOTSTRAP_SEED)
    boot = {k: [] for k in base}
    for _ in range(P.BOOTSTRAP_B):
        i = rng.integers(0, len(ids), len(ids))
        x, y = rates(a, i), rates(b, i)
        for k in boot:
            boot[k].append(y[k] - x[k])
    out = {"n": len(ids)}
    for k in base:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        out[k] = (base[k], top[k], top[k] - base[k], lo, hi)
    return out


def fmt(r: dict, k: str) -> str:
    b, t, d, lo, hi = r[k]
    return f"{b:.3f} → {t:.3f} | **{d:+.3f}** [{lo:+.3f}, {hi:+.3f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="EXP-001")
    ap.add_argument("--snr", default="0dB")
    a = ap.parse_args()
    man = pd.read_csv(P.RAW / "manifest.csv", dtype={"clip_id": str}).set_index("clip_id")
    langs = sorted(p.name for p in (ROOT / "experiments" / "runs" / a.exp).iterdir() if p.is_dir())
    lines = [f"# Clip-length check: {a.exp}, clean → {a.snr}", "",
             "Post-hoc check (added after EXP-001). Each cell: clean → noisy | **rise** [paired 95% bootstrap CI].", ""]
    sections = {"all clips": lambda g: g.index,
                f"common band {BAND_LO:g}-{BAND_HI:g} s": lambda g: g.index[g["duration"].between(BAND_LO, BAND_HI)]}
    for title, pick in sections.items():
        lines += [f"## {title}", "", "| lang | clips | median s | WER | CER | short-word recall |", "|---|--:|--:|---|---|---|"]
        for lang in langs:
            clean, noisy = counts(a.exp, lang, "clean"), counts(a.exp, lang, a.snr)
            g = man.loc[clean.index]
            ids = list(pick(g))
            r = rise(clean, noisy, ids)
            lines.append(f"| {lang} | {r['n']} | {g.loc[ids, 'duration'].median():.1f} | {fmt(r, 'wer')} | "
                         f"{fmt(r, 'cer')} | {fmt(r, 'short')} |")
        lines.append("")
    lines += ["## within each language: shorter vs longer half (split at that language's median)", "",
              "| lang | half | clips | median s | WER | CER | short-word recall |", "|---|---|--:|--:|---|---|---|"]
    for lang in langs:
        clean, noisy = counts(a.exp, lang, "clean"), counts(a.exp, lang, a.snr)
        g = man.loc[clean.index]
        med = g["duration"].median()
        for half, ids in (("shorter", g.index[g["duration"] <= med]), ("longer", g.index[g["duration"] > med])):
            r = rise(clean, noisy, list(ids))
            lines.append(f"| {lang} | {half} | {r['n']} | {g.loc[ids, 'duration'].median():.1f} | {fmt(r, 'wer')} | "
                         f"{fmt(r, 'cer')} | {fmt(r, 'short')} |")
    out = ROOT / "experiments" / f"LENGTH_CHECK_{a.exp}_{a.snr}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
