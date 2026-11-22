"""exp.py — the experiment loop's ledger: spec -> molab run -> pull -> results table.

    python scripts/exp.py new --from EXP-001 --set noise='"mill.wav"' --desc "..." [--hyp H-001]
    python scripts/exp.py submit [EXP-002]    # stamp the pushed git commit, queue on Hugging Face for molab
    python scripts/exp.py pull [--wait]       # fetch molab results into experiments/runs/
    python scripts/exp.py status

A spec is experiments/specs/EXP-NNN.json: a full config, its parent and the keys it changed. A pulled run is one line
of experiments/results.jsonl, and experiments/RESULTS.md is rebuilt from it. Unlike the GeoAI loop there is no
leaderboard: the result is the measurement itself (WER by language and noise level, with confidence intervals).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
import experiment as E  # noqa: E402
import prepare as P  # noqa: E402

EXP = ROOT / "experiments"
SPECS, RUNS, RESULTS = EXP / "specs", EXP / "runs", EXP / "results.jsonl"
RUNS_REPO = "nyakaruosborn/kenyan-asr-noise-runs"   # private Hugging Face dataset: queue/ and runs/


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def spec(exp: str) -> dict:
    return json.loads((SPECS / f"{exp}.json").read_text())


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def set_key(cfg: dict, key: str, value) -> None:
    *path, last = key.split(".")
    for p in path:
        cfg = cfg.setdefault(p, {})
    cfg[last] = value


def flat(cfg: dict, pre: str = "") -> dict:
    out = {}
    for k, v in cfg.items():
        out.update(flat(v, f"{pre}{k}.") if isinstance(v, dict) else {f"{pre}{k}": v})
    return out


def cmd_new(a) -> None:
    parent = a.from_ or None
    cfg = json.loads(json.dumps(spec(parent)["config"] if parent else E.DEFAULT_CONFIG))
    before = flat(cfg)
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        set_key(cfg, k, v)
    after = flat(cfg)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if parent and len(changed) != 1 and not a.allow_multi:
        sys.exit(f"one change per experiment: this changes {changed or 'nothing'} (use --allow-multi to override)")
    n = 1 + max([int(p.stem.split("-")[1]) for p in SPECS.glob("EXP-*.json")], default=0)
    exp = f"EXP-{n:03d}"
    SPECS.mkdir(parents=True, exist_ok=True)
    (SPECS / f"{exp}.json").write_text(json.dumps({"exp": exp, "hypothesis": a.hyp, "description": a.desc, "parent": parent,
                                                   "changed": changed, "config": cfg, "created": now()}, indent=2) + "\n")
    print(f"wrote experiments/specs/{exp}.json (changed vs {parent or 'defaults'}: {changed})\n"
          f"next: commit, git push, `exp.py submit`")


# ── molab: submit and pull ───────────────────────────────────────────────────


def _hf():
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(RUNS_REPO, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(RUNS_REPO, repo_type="dataset").private:
        sys.exit(f"{RUNS_REPO} is PUBLIC: refusing to use it (transcripts come from volunteers' recordings)")
    return api


def cmd_submit(a) -> None:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    if subprocess.run(["git", "status", "--porcelain", "--", "autoresearch", "notebooks"], cwd=ROOT,
                      capture_output=True, text=True).stdout.strip():
        sys.exit("autoresearch/ or notebooks/ has uncommitted changes: commit and push first, so molab runs this code")
    sha = git_sha()
    if not subprocess.run(["git", "branch", "-r", "--contains", sha], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        sys.exit(f"commit {sha[:10]} is not on GitHub yet: git push first")
    api = _hf()
    files = set(api.list_repo_files(RUNS_REPO, repo_type="dataset"))
    done = {r["exp"] for r in jsonl(RESULTS)}
    todo = a.exps or sorted(p.stem for p in SPECS.glob("EXP-*.json") if p.stem not in done and f"queue/{p.stem}.json" not in files)
    if a.resubmit and not a.exps:
        sys.exit("--resubmit needs the EXP ids to run again")
    if not todo:
        return print("nothing to submit")
    ops = []
    for exp in todo:
        s = spec(exp)
        s["git_sha"] = sha if a.resubmit else (s.get("git_sha") or sha)
        s["submitted"] = now()
        (SPECS / f"{exp}.json").write_text(json.dumps(s, indent=2) + "\n")
        ops.append(CommitOperationAdd(path_in_repo=f"queue/{exp}.json", path_or_fileobj=json.dumps(s, indent=2).encode()))
        if a.resubmit:   # a crashed run: clear its end marker so the runner takes it again (finished conditions stay)
            ops += [CommitOperationDelete(path_in_repo=f) for f in (f"runs/{exp}/result.json", f"runs/{exp}/claim.json") if f in files]
    api.create_commit(RUNS_REPO, repo_type="dataset", operations=ops, commit_message=f"queue {', '.join(todo)}")
    print(f"queued {', '.join(todo)} at {sha[:10]}. On molab: open notebooks/molab_runner.py and press Run queue.\n"
          f"Then `exp.py pull`, and commit the stamped specs.")


def cmd_pull(a) -> None:
    from huggingface_hub import hf_hub_download

    api = _hf()
    while True:
        files = api.list_repo_files(RUNS_REPO, repo_type="dataset")
        queued = sorted(f.split("/")[1][:-5] for f in files if f.startswith("queue/") and f.endswith(".json"))
        have = {r["exp"] for r in jsonl(RESULTS)}
        waiting = []
        for exp in queued:
            if exp in have or not (SPECS / f"{exp}.json").exists():
                continue
            if f"runs/{exp}/result.json" not in files:
                n_done = sum(1 for f in files if f.startswith(f"runs/{exp}/") and f.count("/") == 3 and f.endswith(".csv"))
                waiting.append(f"{exp} ({n_done} conditions uploaded)")
                continue
            d = RUNS / exp
            d.mkdir(parents=True, exist_ok=True)
            for f in files:
                if f.startswith(f"runs/{exp}/") and not f.endswith("claim.json"):
                    src = hf_hub_download(RUNS_REPO, f, repo_type="dataset", local_dir=str(RUNS / "_hf"), force_download=True)
                    dst = d / f[len(f"runs/{exp}/"):]
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    Path(src).replace(dst)
            record(exp, spec(exp), json.loads((d / "result.json").read_text()))
        write_results_md()
        if not (a.wait and waiting):
            if waiting:
                print(f"still running on molab: {', '.join(waiting)}")
            return
        time.sleep(120)


def record(exp: str, s: dict, res: dict) -> None:
    rec = {"exp": exp, "hypothesis": s.get("hypothesis"), "description": s["description"], "parent": s.get("parent"),
           "changed": s.get("changed"), "status": res.get("status"), "phases": res.get("phases", {}),
           "git_sha": s.get("git_sha"), "harness": res.get("harness"), "minutes": round(res.get("seconds", 0) / 60, 1),
           "runner": res.get("runner"), "versions": res.get("versions"), "finished": now()}
    if res.get("status") == "crash":
        print(f"{exp}: CRASHED on molab\n{(res.get('error') or '')[-1500:]}")
    if rec["harness"] and rec["harness"] != P.harness_hash():
        print(f"{exp}: made under harness {rec['harness']}, local is {P.harness_hash()}: not comparable")
    kept = [r for r in jsonl(RESULTS) if r["exp"] != exp]
    RESULTS.write_text("".join(json.dumps(r) + "\n" for r in kept + [rec]))
    for lang, rows in rec["phases"].items():
        print(f"{exp} {lang}: " + ", ".join(f"{r['snr']} WER {r['wer']:.3f}" for r in rows))


def _fmt(x, ci=None) -> str:
    if x is None or x != x:
        return "—"
    return f"{x:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci and ci[0] == ci[0] else "")


def write_results_md() -> None:
    rs = jsonl(RESULTS)
    lines = ["# Results", "", f"Validation: {P.VALIDATION}", "",
             "WER and short-word recall with 95% bootstrap CIs. Relative WER = WER at that noise level / clean WER "
             "for the same language and clips (paired CI).", ""]
    for r in rs:
        lines += [f"## {r['exp']}: {r['description']}", "",
                  f"status {r['status']} · {r['minutes']} min · harness {r['harness']} · commit {(r['git_sha'] or '')[:10]}", "",
                  "| lang | noise | WER | relative WER | CER | short-word recall | clips |", "|---|---|--:|--:|--:|--:|--:|"]
        for lang, rows in r["phases"].items():
            for x in rows:
                lines.append(f"| {lang} | {x['snr']} | {_fmt(x['wer'], x.get('wer_ci'))} | "
                             f"{_fmt(x.get('rel_wer'), x.get('rel_wer_ci'))} | {_fmt(x['cer'])} | "
                             f"{_fmt(x['short_recall'], x.get('short_recall_ci'))} | {x['n_clips']} |")
        lines.append("")
    (EXP / "RESULTS.md").write_text("\n".join(lines) + "\n")


def _remote_state() -> dict[str, str]:
    try:
        api = _hf()
        files = set(api.list_repo_files(RUNS_REPO, repo_type="dataset"))
    except Exception:  # noqa: BLE001 - status still works offline
        return {}
    out = {}
    for exp in {f.split("/")[1] for f in files if f.startswith("runs/") and f.count("/") >= 2}:
        if f"runs/{exp}/result.json" in files:
            out[exp] = "finished "
        elif f"runs/{exp}/claim.json" in files:
            n = sum(f.startswith(f"runs/{exp}/") and f.count("/") == 3 and f.endswith(".csv") for f in files)
            out[exp] = f"running   ({n} conditions uploaded)"
    return out


def cmd_status(a) -> None:
    done = {r["exp"]: r for r in jsonl(RESULTS)}
    remote = _remote_state()
    for p in sorted(SPECS.glob("EXP-*.json")):
        s = spec(p.stem)
        state = "pulled   " if p.stem in done else (remote.get(p.stem, "queued   ") if s.get("submitted") else "not run  ")
        print(f"{p.stem}  {state} {s['description']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--from", dest="from_", default="", help="an EXP id, or '' for the defaults")
    n.add_argument("--set", action="append", help="config override key=value (JSON values; dotted keys)")
    n.add_argument("--desc", required=True)
    n.add_argument("--hyp")
    n.add_argument("--allow-multi", action="store_true")
    sub.add_parser("status")
    q = sub.add_parser("submit")
    q.add_argument("exps", nargs="*")
    q.add_argument("--resubmit", action="store_true", help="re-queue these EXPs at the current commit (after a crash)")
    pl = sub.add_parser("pull")
    pl.add_argument("--wait", action="store_true", help="poll every 2 minutes until every queued run is finished")
    sub.add_parser("report", help="rebuild experiments/RESULTS.md from results.jsonl")
    a = ap.parse_args()
    {"new": cmd_new, "status": cmd_status, "submit": cmd_submit, "pull": cmd_pull,
     "report": lambda _: write_results_md()}[a.cmd](a)


if __name__ == "__main__":
    main()
