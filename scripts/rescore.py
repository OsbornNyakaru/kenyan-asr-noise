"""rescore.py — recompute every pulled run's metrics from its saved transcripts under the CURRENT harness.

Only scoring lives in autoresearch/prepare.py, and every run keeps its transcripts (experiments/runs/<EXP>/<lang>/
<snr>.csv), so a scoring change (harness v2: corrupted Kikuyu vowels ī/ū -> ĩ/ũ) does not need any model to run
again. Rewrites each run's <lang>.json and summary.csv, updates its line in experiments/results.jsonl (phases,
harness, rescored_from) and rebuilds RESULTS.md. Transcripts are never changed.

    python scripts/rescore.py            # all pulled runs
    python scripts/rescore.py EXP-001    # one run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
sys.path.insert(0, str(ROOT / "scripts"))
import exp as X  # noqa: E402
import prepare as P  # noqa: E402

ORDER = ["clean", "20dB", "10dB", "5dB", "0dB"]


def rescore(exp: str) -> dict:
    d = X.RUNS / exp
    out = {}
    for lang_dir in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("_")):
        parts = [pd.read_csv(f, dtype={"clip_id": str}, keep_default_na=False)
                 for f in sorted(lang_dir.glob("*.csv"), key=lambda f: ORDER.index(f.stem) if f.stem in ORDER else 99)]
        tab = P.summarise(pd.concat(parts, ignore_index=True))
        out[lang_dir.name] = json.loads(tab.to_json(orient="records"))
        (d / f"{lang_dir.name}.json").write_text(json.dumps(out[lang_dir.name], indent=2))
    pd.DataFrame([r for rows in out.values() for r in rows]).to_csv(d / "summary.csv", index=False)
    return out


def main() -> int:
    recs = X.jsonl(X.RESULTS)
    want = set(sys.argv[1:]) or {r["exp"] for r in recs}
    for r in recs:
        if r["exp"] not in want:
            continue
        old = {lang: {x["snr"]: x["wer"] for x in rows} for lang, rows in r["phases"].items()}
        r["phases"] = rescore(r["exp"])
        r["rescored_from"] = r.get("rescored_from") or r.get("harness")
        r["harness"] = P.harness_hash()
        for lang, rows in r["phases"].items():
            ch = ", ".join(f"{x['snr']} {old.get(lang, {}).get(x['snr'], float('nan')):.3f}->{x['wer']:.3f}" for x in rows)
            print(f"{r['exp']} {lang}: WER {ch}")
    X.RESULTS.write_text("".join(json.dumps(r) + "\n" for r in recs))
    X.write_results_md()
    return 0


if __name__ == "__main__":
    sys.exit(main())
