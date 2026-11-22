"""make_toy_data.py — synthetic clips and noise in data/raw/ so `make smoke` can run the whole pipeline with no
downloads and no real voices. Tones stand in for speech; the "oracle" system returns the reference text, so a smoke run
checks the plumbing (mixing, uploads, resuming, metrics), never a model. Refuses to overwrite a real manifest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
import prepare as P  # noqa: E402

# Placeholder strings, not real commands: they only exercise the text pipeline (ĩ and ũ included on purpose).
TEXT = {"kik": ["toy ĩ one", "toy ũ two words", "toy three"], "swh": ["toy one", "toy two words", "toy three"],
        "eng": ["toy one", "toy two words", "toy three"]}


def main() -> int:
    import soundfile as sf

    m = P.RAW / "manifest.csv"
    if m.exists() and not (P.RAW / ".toy").exists():
        sys.exit("data/raw/manifest.csv is real data: refusing to overwrite it")
    rng = np.random.default_rng(0)
    rows = []
    for lang, phrases in TEXT.items():
        (P.RAW / "audio" / lang).mkdir(parents=True, exist_ok=True)
        for i in range(10):
            t = np.arange(int(P.SR * (2 + rng.random()))) / P.SR
            x = (0.3 * np.sin(2 * np.pi * (200 + 40 * i) * t)).astype(np.float32)
            cid = f"{lang}_toy{i:02d}"
            sf.write(P.RAW / "audio" / lang / f"{cid}.flac", x, P.SR)
            rows.append({"clip_id": cid, "lang": lang, "path": f"audio/{lang}/{cid}.flac", "text": phrases[i % 3],
                         "speaker": f"toy{i % 4}", "gender": "", "duration": round(len(x) / P.SR, 2), "source": "toy"})
    pd.DataFrame(rows).to_csv(m, index=False)
    (P.RAW / "noise").mkdir(parents=True, exist_ok=True)
    sf.write(P.RAW / "noise" / "machine.wav", (0.1 * rng.standard_normal(P.SR * 30)).astype(np.float32), P.SR)
    (P.RAW / ".toy").write_text("synthetic data from scripts/make_toy_data.py\n")
    print(f"toy data: {len(rows)} clips, 30 s of noise in {P.RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
