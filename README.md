# Kenyan ASR in machine noise

Project 1 of the ML Lagos Research Onramp capstone (Osborn Nyakaru, Group 4).

**Question.** How much faster does an openly available Kikuyu speech recogniser lose accuracy as factory-like noise
rises from 20 dB to 0 dB signal-to-noise ratio, compared with Kiswahili and English recognisers tested on the same
number of clips?

**Supporting question.** Do short words of one or two syllables, which is what spoken commands look like, degrade
faster than longer words?

A stop command is short and is usually shouted over noise. Published Kikuyu results are measured on clean recordings;
this measures what happens when they are not.

## What is measured

| | |
|---|---|
| Languages | Kikuyu (`kik`), Kiswahili (`swh`), English (`eng`), 300 clips each |
| Noise | one machine-noise recording mixed in at 20, 10, 5 and 0 dB, plus clean |
| Systems | MMS-1B-all with one adapter per language (default), so curves compare languages, not systems |
| Metrics | WER, no-space CER, short-word recall; WER relative to clean; 95% bootstrap CIs, paired across noise levels |

The definitions live in `autoresearch/prepare.py` and are frozen.

## How a run works

```
laptop                                GitHub (public)            Hugging Face (private)         molab
exp.py new  -> spec JSON
git push  --------------------------> code at commit SHA
exp.py submit ---------------------------------------------------> queue/EXP-NNN.json
                                                                                       runner: Run queue
                                      code @ SHA  <------------------------------------ downloads code + clips
                                                                   runs/EXP-NNN/...  <-- uploads each noise level
exp.py pull  <---------------------------------------------------- transcripts + metrics -> RESULTS.md
```

## Quick start

```bash
make setup          # .venv
make hooks          # pre-commit check: no recordings in this public repo
make test           # 13 tests, no data needed
make smoke          # whole pipeline on toy data with an oracle system, no downloads
make analysis       # day-1 check: does MMS cover Kikuyu?
```

## Day-1 checklist (decisions are Osborn's)

1. **Model coverage.** Open `notebooks/analysis.py` and check whether `facebook/mms-1b-all` lists `kik`. If not, pick
   an open Kikuyu checkpoint and change `systems.kik` in a new spec.
2. **Data sources.** Choose where the 300 clips per language come from (`scripts/build_data.py` takes Hugging Face
   datasets such as FLEURS, or an extracted Common Voice folder). Check each source's licence and whether it is
   still available where you expect it.
3. **Noise.** A public corpus (for example MUSAN) or ten minutes recorded in a workshop or posho mill.
4. **Build and push the data** to the private dataset:
   `python scripts/build_data.py --source kik=... --source swh=... --source eng=... --noise mill.wav --push`
5. **First spec:** `python scripts/exp.py new --desc "baseline: MMS-1B-all, 3 languages, 5 noise levels"`, commit,
   push, `exp.py submit`, then Run queue on molab.

## Limitations

Read speech is calmer than shouting, and noise added afterwards is not speech produced in noise (the Lombard effect).
If the languages come from different corpora, recording conditions differ too. Results describe these clips and this
noise, not every speaker of each language.
