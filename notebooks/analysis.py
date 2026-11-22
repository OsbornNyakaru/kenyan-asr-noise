# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.16",
#     "pandas",
#     "matplotlib",
#     "huggingface_hub>=0.35",
# ]
# ///
"""analysis.py — Project 1 analysis notebook. Two jobs:

  1. Day 1: does MMS cover Kikuyu? Lists the language adapters in facebook/mms-1b-all (no token needed).
  2. After a run: WER and short-word recall against noise level per language, and relative degradation, read from
     experiments/runs/<EXP>/summary.csv after `exp.py pull` (or straight from the private runs repo with a token).

Interpretation is yours: this notebook draws the curves, it does not decide what they mean.
"""

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", app_title="Kenyan ASR in noise — analysis")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    ROOT = Path(mo.notebook_dir() or Path.cwd()).resolve().parent
    return Path, ROOT, json, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # Kenyan ASR in noise — analysis

    ## Day 1: does MMS cover Kikuyu?

    `experiment.py` assumes the MMS-1B-all checkpoint has a Kikuyu adapter (`kik`). Press the button to list its
    languages. If `kik` is missing, choose another open Kikuyu checkpoint and note that the comparison is now across
    systems, not only languages.
    """)
    return


@app.cell
def _(mo):
    check_btn = mo.ui.run_button(label="Check MMS language list")
    check_btn
    return (check_btn,)


@app.cell
def _(check_btn, json, mo):
    mo.stop(not check_btn.value, mo.md("_Press the button to check._"))
    from huggingface_hub import hf_hub_download

    _langs = sorted(json.loads(open(hf_hub_download("facebook/mms-1b-all", "vocab.json"), encoding="utf-8").read()))
    _want = {"kik": "Kikuyu", "guz": "Ekegusii", "swh": "Kiswahili", "eng": "English", "luo": "Dholuo"}
    mo.md(f"facebook/mms-1b-all has **{len(_langs)}** language vocabularies.\n\n"
          + "\n".join(f"- `{c}` {n}: **{'yes' if c in _langs else 'no'}**" for c, n in _want.items()))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Results

    Pick a pulled run. Curves come from `experiments/runs/<EXP>/summary.csv`.
    """)
    return


@app.cell
def _(ROOT, mo):
    _runs = sorted(p.parent.name for p in (ROOT / "experiments" / "runs").glob("*/summary.csv"))
    run_pick = mo.ui.dropdown(options=_runs, value=_runs[-1] if _runs else None, label="run")
    run_pick if _runs else mo.md("_No pulled runs yet: `python scripts/exp.py pull`, or `make smoke` for a toy run._")
    return (run_pick,)


@app.cell
def _(ROOT, json, mo, pd, run_pick):
    mo.stop(not run_pick.value)
    summary = pd.read_csv(ROOT / "experiments" / "runs" / run_pick.value / "summary.csv")
    def _ci(s):   # CSV holds Python list text such as "[0.1, 0.2]", "[nan, nan]" or "[None, None]"
        if not isinstance(s, str):
            return [float("nan"), float("nan")]
        v = json.loads(s.replace("nan", "NaN").replace("None", "NaN"))
        return [float(x) for x in v]

    for _c in [c for c in summary.columns if c.endswith("_ci")]:
        summary[_c] = summary[_c].apply(_ci)
    _order = {"clean": 99}
    summary["snr_db"] = summary["snr"].map(lambda s: _order.get(s, None) or float(s.replace("dB", "")))
    summary = summary.sort_values(["lang", "snr_db"], ascending=[True, False])
    mo.ui.table(summary[["lang", "snr", "wer", "rel_wer", "cer", "short_recall", "n_clips"]].round(3))
    return (summary,)


@app.cell
def _(summary):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for lang, g in summary.groupby("lang"):
        x = g["snr_db"].where(g["snr"] != "clean", 30)   # draw clean at 30 dB, labelled below
        for ax, col in zip(axes, ["wer", "rel_wer", "short_recall"]):
            lo = [c[0] for c in g[f"{col}_ci"]] if f"{col}_ci" in g else None
            hi = [c[1] for c in g[f"{col}_ci"]] if f"{col}_ci" in g else None
            ax.plot(x, g[col], marker="o", label=lang)
            if lo is not None:
                ax.fill_between(x, lo, hi, alpha=0.15)
    for ax, title in zip(axes, ["WER", "WER relative to clean", "short-word recall"]):
        ax.set_title(title)
        ax.set_xlabel("SNR (dB); 30 = clean")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    **Reading the curves (questions for you, not answers):** do the relative-WER intervals for Kikuyu and English
    overlap at 0 dB? Is short-word recall falling faster than WER rises? Would the answer change if Kikuyu used a
    different model from the other two?
    """)
    return


if __name__ == "__main__":
    app.run()
