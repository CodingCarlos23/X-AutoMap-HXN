#!/usr/bin/env python3
"""
Model sweep — the same VLM method run with several models; only the model changes.

Each model is one normal run_comparison.py run (its own results.json, images, raw replies), all inside one sweep
folder, run in parallel processes:

  benchmarks/runs/<timestamp>_<sweep name>/
      <model>/              a full run for that model (results.json, index.html, …)
      configs/<model>.json  the exact config each run used
      logs/<model>.log
      sweep.json            models, folders, exit codes
      model_sweep.html      combined report (model_sweep_report.py, free to rebuild)

Usage (from the repo root):
  pixi run python benchmarks/model_sweep.py benchmarks/configs/car_model_sweep.json [--workers 7] [--no-vlm]
Add / replace models in an existing sweep (same config, other models are kept):
  pixi run python benchmarks/model_sweep.py benchmarks/configs/car_model_sweep.json \
      --into benchmarks/runs/<sweep> --models gpt-6-sol --exclude gpt-6-astra="material call timed out (HTTP 504)"
"""
import argparse
import concurrent.futures
import copy
import datetime
import json
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent


def safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=pathlib.Path)
    ap.add_argument("--workers", type=int, default=7, help="models run at the same time")
    ap.add_argument("--no-vlm", action="store_true", help="pass --no-vlm to every run (free pipeline check)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None, help="runs folder (default benchmarks/runs/)")
    ap.add_argument("--into", type=pathlib.Path, default=None, help="add the models to this existing sweep folder")
    ap.add_argument("--models", nargs="+", default=None, help="run only these models (default: the config's list)")
    ap.add_argument("--exclude", nargs="*", default=[], metavar='MODEL="reason"',
                    help="leave these models out of the report (their files are kept)")
    args = ap.parse_args()

    base = json.loads(args.config.read_text())
    models = args.models or base.pop("models")
    base.pop("models", None)
    name = base.get("run_name") or "model_sweep"
    if args.into:
        sweep = args.into.resolve()
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep = (args.out_dir.resolve() if args.out_dir else HERE / "runs") / f"{stamp}_{name}"
    (sweep / "configs").mkdir(parents=True, exist_ok=True)
    (sweep / "logs").mkdir(exist_ok=True)
    tmp = sweep / ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()

    def run_one(model):
        cfg = copy.deepcopy(base)
        cfg["vlm"]["model"] = model
        cfg["run_name"] = safe(model)
        cpath = sweep / "configs" / f"{safe(model)}.json"
        cpath.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        out = tmp / safe(model)
        out.mkdir()
        cmd = [sys.executable, "-u", str(HERE / "run_comparison.py"), str(cpath), "--out-dir", str(out)]
        if args.no_vlm:
            cmd.append("--no-vlm")
        t0 = time.time()
        with open(sweep / "logs" / f"{safe(model)}.log", "w") as log:
            code = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        runs = [d for d in out.iterdir() if d.is_dir() and (d / "results.json").exists()]
        folder = None
        if runs:
            if (sweep / safe(model)).exists():  # re-running a model replaces its old folder
                shutil.rmtree(sweep / safe(model))
            shutil.move(str(runs[0]), str(sweep / safe(model)))
            folder = safe(model)
        print(f"  {model:22} exit {code} · {time.time() - t0:6.0f}s · {'ok' if folder else 'NO RESULTS'}", flush=True)
        return {"model": model, "folder": folder, "exit_code": code, "time_s": round(time.time() - t0, 1)}

    print(f"Model sweep → {sweep}\n{len(models)} models: {', '.join(models)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        res = list(ex.map(run_one, models))
    shutil.rmtree(tmp, ignore_errors=True)
    sj = sweep / "sweep.json"
    old = json.loads(sj.read_text()) if sj.exists() else {
        "name": name, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config), "baseline": base.get("baseline_model"), "models": []}
    new = {r["model"]: r for r in res}
    old["models"] = [new.pop(m["model"], m) for m in old["models"]] + list(new.values())
    for ex in args.exclude:
        mname, _, reason = ex.partition("=")
        for m in old["models"]:
            if m["model"] == mname:
                m["excluded"] = reason or "excluded"
    sj.write_text(json.dumps(old, indent=2))
    from report import write_runs_index
    write_runs_index(sweep.parent)  # runs list shows the sweep
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO / "src"))
    import model_sweep_report
    print("Report →", model_sweep_report.build(sweep))


if __name__ == "__main__":
    main()
