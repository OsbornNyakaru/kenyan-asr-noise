# CLAUDE.md — agent entry point

ML Lagos Research Onramp capstone, Project 1: **how much faster does an open Kikuyu speech recogniser lose accuracy as
machine noise rises from 20 dB to 0 dB SNR, compared with Kiswahili and English?** Supporting question: do short words
(1-2 syllables, what spoken commands look like) degrade faster than longer words? Same structure as
`OsbornNyakaru/geoai-aquaculture`: frozen harness, one experiment file, specs, a molab runner, private HF datasets.

## Read in this order

1. `README.md`: the question, the loop and the day-1 checklist.
2. `experiments/RESULTS.md`: the ledger (appears after the first `exp.py pull`).
3. `autoresearch/experiment.py`: systems and config.

## Files

- `autoresearch/prepare.py`: FROZEN. Manifest, noise mixing at a given SNR, text normalisation, WER/CER/short-word
  recall, bootstrap CIs. Its SHA-256 is pinned in `notebooks/molab_runner.py`; the runner refuses a mismatch.
- `autoresearch/experiment.py`: the one file an experiment edits. `DEFAULT_CONFIG` = MMS-1B-all for all three languages.
- `autoresearch/remote.py`: runs one spec language by language and noise level by noise level, resumable.
- `notebooks/molab_runner.py`: the molab notebook (synced from this public repo). It runs queued specs.
- `notebooks/analysis.py`: day-1 MMS coverage check, then the curves.
- `scripts/exp.py`: `new`, `submit`, `pull`, `status`, `report`. `scripts/build_data.py`: clips + noise -> private HF.

## The loop

1. `exp.py new --from EXP-001 --set <key>=<value> --desc "..."`: one change per experiment.
2. Commit and `git push` (molab runs the code at the spec's commit, never local edits).
3. `exp.py submit`: queues the spec in the private HF dataset `nyakaruosborn/kenyan-asr-noise-runs`.
4. On molab: open `notebooks/molab_runner.py`, paste an HF token, **Run queue**.
5. `exp.py pull`: transcripts and metrics land in `experiments/runs/EXP-NNN/`, `RESULTS.md` is rebuilt.

## Principles (from The Engineer's Operating Principles)

1. **Osborn owns the decisions that shape the answer**: data sources, which model per language, what a "short word"
   is, and what the curves mean. Code that changes those must say so in the spec description.
2. **One change per experiment.** `exp.py new` refuses more unless `--allow-multi`.
3. **A difference counts only when the 95% intervals do not overlap.** Report the intervals, not just the point.
4. **Do not edit `prepare.py`** once a real run exists. If it must change, bump the runner's PREPARE_SHA256 and
   treat earlier results as a different harness (exp.py warns).
5. **Recordings never enter git.** `make hooks` installs `scripts/check_public.py` as a pre-commit check.

## Environment

- Python: `.venv/bin/python` (`make setup`). Heavy runs go to molab; `make test` and `make smoke` (toy data, oracle
  system, no downloads) are the only local runs.
- The GitHub repository is public (so molab can sync). Clips and run outputs live in private Hugging Face datasets
  (`kenyan-asr-noise-data`, `-runs`).
- Report times in EAT (UTC+3). Commits carry no AI attribution lines. No tokens or keys in code.
