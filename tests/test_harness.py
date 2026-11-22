"""Harness tests. They need no real data: mixing, text and metrics are checked on synthetic inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
import experiment as E  # noqa: E402
import prepare as P  # noqa: E402


def _tone(seconds=2.0, f=220.0, amp=0.3):
    t = np.arange(int(P.SR * seconds)) / P.SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


@pytest.mark.parametrize("snr", [20, 10, 5, 0, -5])
def test_mix_hits_the_requested_snr(snr):
    clean = _tone()
    noise = np.random.default_rng(0).standard_normal(P.SR * 10).astype(np.float32) * 0.05
    out = P.mix(clean, noise, snr, "clip_a")
    added = out.astype(np.float64) - clean * (out @ clean) / (clean @ clean)   # remove the (possibly rescaled) speech
    scale = (out @ clean) / (clean @ clean)
    measured = 10 * np.log10(P.power(clean * scale) / P.power(added))
    assert measured == pytest.approx(snr, abs=0.2)


def test_mix_is_deterministic_and_clean_is_untouched():
    clean, noise = _tone(), np.random.default_rng(1).standard_normal(P.SR * 5).astype(np.float32)
    assert np.array_equal(P.mix(clean, noise, 5, "c1"), P.mix(clean, noise, 5, "c1"))
    assert not np.array_equal(P.mix(clean, noise, 5, "c1"), P.mix(clean, noise, 5, "c2"))   # different offset
    assert np.array_equal(P.mix(clean, noise, None, "c1"), clean)


def test_mix_never_clips_and_handles_short_noise():
    out = P.mix(_tone(amp=0.95), np.random.default_rng(2).standard_normal(1000).astype(np.float32), 0, "c")
    assert np.max(np.abs(out)) <= 0.99 + 1e-6 and len(out) == P.SR * 2


def test_normalise_keeps_kikuyu_vowels_and_word_internal_apostrophes():
    assert P.normalise("Ĩ  ũ, NĨ!") == "ĩ ũ nĩ"
    assert P.normalise("don't  'quote'") == "don't quote"
    assert P.normalise("i") != P.normalise("ĩ")   # ĩ is a different vowel; stripping it would hide errors


def test_syllables_count_vowel_groups():
    assert P.syllables("go") == 1 and P.syllables("stop") == 1 and P.syllables("haha") == 2
    assert P.syllables("mbele") == 2 and P.syllables("simama") == 3 and P.syllables("ũũ") == 1


def test_counts_and_metrics():
    c = P.clip_counts("go stop here now", "go top here")
    assert c["w_err"] == 2 and c["w_ref"] == 4          # one substitution, one deletion
    assert c["short_ref"] == 4 and c["short_hit"] == 2   # 'go' and 'here' found; 'stop' and 'now' missed
    m = P.metrics(pd.DataFrame([c, P.clip_counts("a b", "a b")]))
    assert m["wer"] == pytest.approx(2 / 6)


def test_summarise_relative_wer_is_one_for_clean_and_paired():
    rows = []
    for i in range(20):
        ref = "go stop here now"
        rows += [{"clip_id": f"c{i}", "lang": "eng", "snr_tag": "clean", "ref": ref, "hyp": "go stop here"},
                 {"clip_id": f"c{i}", "lang": "eng", "snr_tag": "0dB", "ref": ref, "hyp": "go"}]
    s = P.summarise(pd.DataFrame(rows)).set_index("snr")
    assert s.loc["clean", "rel_wer"] == pytest.approx(1.0)
    assert s.loc["0dB", "rel_wer"] == pytest.approx(3.0)
    lo, hi = s.loc["0dB", "wer_ci"]
    assert lo <= s.loc["0dB", "wer"] <= hi


def test_default_config_phases_and_keys_are_used():
    cfg = E.DEFAULT_CONFIG
    assert E.phases(cfg) == ["kik", "swh", "eng"]
    assert set(cfg["systems"]) == set(cfg["langs"]) <= set(P.LANGS)
    assert cfg["snrs"][0] is None   # clean first: relative WER needs it


def test_end_to_end_with_toy_data_and_oracle(tmp_path, monkeypatch):
    """The whole remote path on synthetic data: an oracle system must give WER 0 at every noise level."""
    import remote as R

    raw = tmp_path / "raw"
    monkeypatch.setattr(P, "RAW", raw)
    import soundfile as sf

    rows = []
    for lang in E.DEFAULT_CONFIG["langs"]:
        (raw / "audio" / lang).mkdir(parents=True)
        for i in range(3):
            cid = f"{lang}_{i}"
            sf.write(raw / "audio" / lang / f"{cid}.flac", _tone(f=200 + 50 * i), P.SR)
            rows.append({"clip_id": cid, "lang": lang, "path": f"audio/{lang}/{cid}.flac", "text": "toy words",
                         "speaker": "s", "gender": "", "duration": 2.0, "source": "test"})
    pd.DataFrame(rows).to_csv(raw / "manifest.csv", index=False)
    (raw / "noise").mkdir()
    sf.write(raw / "noise" / "machine.wav", np.random.default_rng(0).standard_normal(P.SR * 5).astype(np.float32) * 0.1, P.SR)
    cfg = {**E.DEFAULT_CONFIG, "n_clips": 3, "device": "cpu",
           "systems": {k: {"family": "oracle"} for k in E.DEFAULT_CONFIG["langs"]}}
    store = R.LocalStore(tmp_path / "run")
    res = R.run_spec({"config": cfg}, store, log=lambda *_: None)
    assert res["status"] == "ok"
    assert all(r["wer"] == 0 for rows_ in res["phases"].values() for r in rows_)
    assert store.exists("summary.csv") and store.exists("kik/0dB.csv")
    again = R.run_spec({"config": cfg}, store, log=lambda *_: None)    # resumes: everything already done
    assert again["phases"] == res["phases"]
