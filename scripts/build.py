#!/usr/bin/env python3
"""Validate every run under results/ and build results.json.

    python scripts/build.py                          # validate, print a summary table
    python scripts/build.py --out _site/results.json # also write the combined JSON

Exit status 1 if any run fails. CI runs the same command on every pull request.
"""
import argparse
import csv
import datetime
import json
import pathlib
import re
import sys

import yaml
from jsonschema import Draft202012Validator

ROOT = pathlib.Path(__file__).resolve().parent.parent
COLUMNS = ["label", "scenario", "in_len", "out_len", "prompts", "conc",
           "req_s", "out_tok_s", "total_tok_s", "ttft_ms", "tpot_ms", "p99_tpot_ms"]
INT_COLS = ["in_len", "out_len", "prompts", "conc"]
METRIC_COLS = ["req_s", "out_tok_s", "total_tok_s", "ttft_ms", "tpot_ms", "p99_tpot_ms"]
CONTEXT_MARGIN = 512  # a point is skipped when in_len + out_len + 512 > max_context

# (scenario, in_len, out_len, prompts, concurrency) of every point in a suite
SUITES = {
    "olivia-v1": (
        [("conc-scaling", 1024, 128, 4 * c, c) for c in (1, 4, 16, 32, 64, 128)]
        + [("prefill", i, 32, 16, 8) for i in (1024, 4096, 16384)]
        + [("decode", 256, o, 16, 8) for o in (128, 512, 1024)]
    ),
}


def plain(value):
    """YAML turns 2026-09-27 into a date; the schema wants strings."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    return value


def read_points(path, errors):
    with path.open(newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if not rows or rows[0] != COLUMNS:
        errors.append(f"result.tsv header must be: {' '.join(COLUMNS)}")
        return []
    points = []
    for n, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(COLUMNS):
            errors.append(f"result.tsv line {n}: {len(row)} columns, expected {len(COLUMNS)}")
            continue
        p = dict(zip(COLUMNS, row))
        try:
            for c in INT_COLS:
                p[c] = int(p[c])
            for c in METRIC_COLS:
                p[c] = float(p[c])
        except ValueError:
            errors.append(f"result.tsv line {n}: empty or non-numeric value (a failed point?)")
            continue
        if min(p[c] for c in METRIC_COLS) <= 0:
            errors.append(f"result.tsv line {n}: metrics must be positive")
            continue
        points.append(p)
    return points


def summarise(points, gpus):
    by_key = {(p["scenario"], p["in_len"], p["out_len"], p["conc"]): p for p in points}
    conc = {p["conc"]: p for p in points if p["scenario"] == "conc-scaling"}
    peak = max(conc.values(), key=lambda p: p["out_tok_s"]) if conc else None
    pick = lambda key, col: by_key[key][col] if key in by_key else None
    return {
        "out_tok_s_by_conc": {str(c): conc[c]["out_tok_s"] for c in sorted(conc)},
        "peak_out_tok_s": peak["out_tok_s"] if peak else None,
        "peak_conc": peak["conc"] if peak else None,
        "peak_out_tok_s_per_gpu": round(peak["out_tok_s"] / gpus, 1) if peak else None,
        "tpot_ms_c1": pick(("conc-scaling", 1024, 128, 1), "tpot_ms"),
        "ttft_ms_c1": pick(("conc-scaling", 1024, 128, 1), "ttft_ms"),
        "prefill_ttft_ms": {str(i): pick(("prefill", i, 32, 8), "ttft_ms") for i in (1024, 4096, 16384)},
        "decode_out_tok_s": {str(o): pick(("decode", 256, o, 8), "out_tok_s") for o in (128, 512, 1024)},
    }


def check_run(folder, validator):
    errors = []
    run_file, tsv_file = folder / "run.yaml", folder / "result.tsv"
    for f in (run_file, tsv_file):
        if not f.is_file():
            errors.append(f"missing {f.name}")
    extra = sorted(p.name for p in folder.iterdir() if p.name not in ("run.yaml", "result.tsv", "README.md"))
    if extra:
        errors.append(f"unexpected files: {', '.join(extra)} (only run.yaml, result.tsv, optional README.md)")
    if errors:
        return None, errors

    try:
        run = plain(yaml.safe_load(run_file.read_text()))
    except yaml.YAMLError as e:
        return None, [f"run.yaml does not parse: {e}"]
    for e in sorted(validator.iter_errors(run), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in e.path) or "(top level)"
        errors.append(f"run.yaml {where}: {e.message}")
    if errors:
        return None, errors

    if run["id"] != folder.name:
        errors.append(f"id {run['id']!r} must equal the folder name {folder.name!r}")
    if not folder.name.startswith(run["date"] + "_"):
        errors.append(f"folder name must start with the run date {run['date']}")
    if run["gpus"] > run["nodes"] * run["system"]["gpus_per_node"]:
        errors.append("gpus is larger than nodes x gpus_per_node")
    par = run["parallelism"]
    if par["tp"] * par["pp"] > run["gpus"]:
        errors.append("tp x pp is larger than gpus")

    points = read_points(tsv_file, errors)
    suite = set(SUITES[run["benchmark"]["suite"]])
    want = {pt for pt in suite if pt[1] + pt[2] + CONTEXT_MARGIN <= run["max_context"]}
    got = [(p["scenario"], p["in_len"], p["out_len"], p["prompts"], p["conc"]) for p in points]
    if len(got) != len(set(got)):
        errors.append("result.tsv has duplicate points")
    for pt in sorted(want - set(got)):
        errors.append(f"result.tsv is missing point {pt[0]} in={pt[1]} out={pt[2]} prompts={pt[3]} c={pt[4]}")
    for pt in sorted(set(got) - want):
        why = f"does not fit max_context {run['max_context']}" if pt in suite else f"is not in suite {run['benchmark']['suite']}"
        errors.append(f"result.tsv point {pt[0]} in={pt[1]} out={pt[2]} prompts={pt[3]} c={pt[4]} {why}")

    # The server's own log is the check that the client measured this server and nothing else did.
    expected = sum(p["prompts"] for p in points)
    logged = run["validation"]["server_chat_requests"]
    slack = len(points) + 1  # one warm-up request per point and one at server start, at most
    if not expected <= logged <= expected + slack:
        errors.append(f"server logged {logged} chat requests; the suite sent {expected} "
                      f"(allowed {expected}..{expected + slack}); another job may have shared the server")
    if errors:
        return None, errors

    run["benchmark_requests"] = expected
    run["points"] = [{k: v for k, v in p.items() if k != "label"} for p in points]
    run["summary"] = summarise(points, run["gpus"])
    run["path"] = f"results/{folder.name}"
    return run, []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, help="write the combined results.json here")
    args = ap.parse_args()

    validator = Draft202012Validator(json.loads((ROOT / "schema" / "run.schema.json").read_text()))
    folders = sorted(p for p in (ROOT / "results").iterdir() if p.is_dir())
    runs, failed = [], 0
    for folder in folders:
        run, errors = check_run(folder, validator)
        if errors:
            failed += 1
            print(f"FAIL {folder.name}")
            for e in errors:
                print(f"     {e}")
        else:
            runs.append(run)

    if runs:
        print(f"\n{'run':<48} {'gpus':>4} {'c=1':>7} {'c=16':>7} {'c=64':>7} {'c=128':>7} {'tpot c=1':>9}")
        for r in runs:
            by = r["summary"]["out_tok_s_by_conc"]
            cell = lambda c: f"{by[c]:>7.0f}" if c in by else f"{'-':>7}"
            print(f"{r['id']:<48} {r['gpus']:>4} {cell('1')} {cell('16')} {cell('64')} {cell('128')} "
                  f"{r['summary']['tpot_ms_c1']:>7.2f}ms")
    print(f"\n{len(runs)} valid, {failed} failed")

    if args.out and not failed:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema_version": 1,
            "suites": {name: [dict(zip(("scenario", "in_len", "out_len", "prompts", "conc"), pt)) for pt in pts]
                       for name, pts in SUITES.items()},
            "columns": {"out_tok_s": "output tokens/s, all requests", "total_tok_s": "input+output tokens/s",
                        "ttft_ms": "mean time to first token", "tpot_ms": "mean time per output token",
                        "p99_tpot_ms": "99th percentile time per output token"},
            "runs": runs,
        }
        args.out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.out} ({len(runs)} runs)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
