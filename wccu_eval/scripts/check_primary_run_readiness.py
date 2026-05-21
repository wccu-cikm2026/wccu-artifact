from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any


def _read_text(root: Path | None, zf: zipfile.ZipFile | None, name: str) -> str:
    if zf is not None:
        try:
            return zf.read(name).decode("utf-8", errors="replace")
        except KeyError:
            return ""
    assert root is not None
    p = root / name
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def _exists(root: Path | None, zf: zipfile.ZipFile | None, name: str) -> bool:
    if zf is not None:
        return name in set(zf.namelist())
    assert root is not None
    return (root / name).exists()


def _csv_rows(root: Path | None, zf: zipfile.ZipFile | None, name: str) -> list[dict[str, str]]:
    text = _read_text(root, zf, name)
    if not text.strip():
        return []
    return list(csv.DictReader(text.splitlines()))


def _json(root: Path | None, zf: zipfile.ZipFile | None, name: str) -> Any:
    text = _read_text(root, zf, name)
    if not text.strip():
        return None
    return json.loads(text)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        val = row.get(key, "")
        if val in ("", None):
            return default
        return float(val)
    except Exception:
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        val = row.get(key, "")
        if val in ("", None):
            return default
        return int(float(val))
    except Exception:
        return default


def _condition(rows: list[dict[str, str]], cond: str) -> dict[str, str] | None:
    for r in rows:
        if r.get("condition") == cond:
            return r
    return None


def _add(checks: list[dict[str, Any]], name: str, ok: bool, severity: str, detail: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "severity": severity, "detail": detail})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check whether a CooperBench-derived live-LLM pilot/main run is ready to scale or cite."
    )
    ap.add_argument("--tag", required=True, help="Experiment tag, e.g. wccu_pilot_seed7")
    ap.add_argument("--root", default=".", help="Repository/output root when reading unpacked results.")
    ap.add_argument("--zip", default="", help="Optional packaged output zip to inspect directly.")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--min-converted", type=int, default=50)
    ap.add_argument("--min-subset", type=int, default=30)
    ap.add_argument("--min-commitment", type=int, default=30)
    ap.add_argument("--min-repos", type=int, default=2)
    args = ap.parse_args(argv)

    zf = zipfile.ZipFile(args.zip) if args.zip else None
    root = None if zf is not None else Path(args.root)
    tag = args.tag

    checks: list[dict[str, Any]] = []

    primary_note = _read_text(root, zf, f"results/{tag}/PRIMARY_RUN_NOTE.txt")
    _add(
        checks,
        "primary run note exists",
        bool(primary_note.strip()),
        "error",
        f"results/{tag}/PRIMARY_RUN_NOTE.txt",
    )
    _add(
        checks,
        "mock LLM disabled",
        "mock_llm=0" in primary_note,
        "error",
        "primary evidence must use real LLM generations, not mock rows.",
    )
    _add(
        checks,
        "CooperBench-derived primary workload",
        "CooperBench-derived" in primary_note,
        "error",
        "primary note should identify the CooperBench-derived live-LLM workload.",
    )

    provider_errors = [
        f"analysis/{tag}/cooperbench_workspace_wccu/provider_errors.jsonl",
        f"analysis/{tag}/cooperbench_commitment_stale_wccu/provider_errors.jsonl",
        f"results/{tag}/provider_errors.jsonl",
    ]
    present_provider_logs = [p for p in provider_errors if _exists(root, zf, p)]
    provider_error_text = "".join(_read_text(root, zf, p) for p in present_provider_logs)
    _add(
        checks,
        "provider errors empty",
        provider_error_text.strip() == "",
        "error",
        "non-empty provider error logs indicate rows may be missing or model calls failed.",
    )

    target_failures = [
        f"analysis/{tag}/cooperbench_workspace_wccu/target_grounding_failures.jsonl",
        f"analysis/{tag}/cooperbench_commitment_stale_wccu/target_grounding_failures.jsonl",
    ]
    target_failure_text = "".join(_read_text(root, zf, p) for p in target_failures if _exists(root, zf, p))
    _add(
        checks,
        "target grounding failures empty",
        target_failure_text.strip() == "",
        "warning",
        "target grounding failures are not always fatal, but main runs should inspect them.",
    )

    report = _json(root, zf, f"analysis/{tag}/cooperbench_dataset_report.json")
    if isinstance(report, dict) and isinstance(report.get("stages"), list):
        sections = {
            str(stage.get("stage") or ""): stage
            for stage in report.get("stages", [])
            if isinstance(stage, dict)
        }
    elif isinstance(report, dict):
        sections = report.get("sections") or report.get("datasets") or report
    else:
        sections = None
    # The report schema may be either a dict-of-sections, a {"datasets": ...},
    # or the current {"stages": [{"stage": ...}]} form.
    def get_section(name: str) -> dict[str, Any]:
        if isinstance(sections, dict):
            val = sections.get(name) or sections.get(name.replace("_", "-"))
            if isinstance(val, dict):
                return val
        if isinstance(report, dict):
            val = report.get(name)
            if isinstance(val, dict):
                return val
        return {}

    converted = get_section("converted_cooperbench")
    subset = get_section("conflict_preferred_subset")
    commitment = get_section("commitment_staleness_diagnostic")

    def records(sec: dict[str, Any]) -> int:
        for k in ("records", "record_count", "n"):
            if k in sec:
                try:
                    return int(sec[k])
                except Exception:
                    pass
        return 0

    def repos(sec: dict[str, Any]) -> int:
        val = sec.get("repos") or sec.get("repo_counts")
        if isinstance(val, dict):
            return len(val)
        if isinstance(val, str):
            try:
                return len(json.loads(val))
            except Exception:
                return 1 if val else 0
        return 0

    converted_n, subset_n, commitment_n = records(converted), records(subset), records(commitment)
    repo_n = repos(converted)

    _add(
        checks,
        "dataset report exists",
        isinstance(report, dict),
        "error",
        f"analysis/{tag}/cooperbench_dataset_report.json",
    )
    _add(
        checks,
        f"converted records >= {args.min_converted}",
        converted_n >= args.min_converted,
        "warning",
        f"converted records={converted_n}; pilot runs may be smaller, but main runs should use a larger slice.",
    )
    _add(
        checks,
        f"subset records >= {args.min_subset}",
        subset_n >= args.min_subset,
        "warning",
        f"subset records={subset_n}; main tables are more stable with at least {args.min_subset}.",
    )
    _add(
        checks,
        f"commitment diagnostic records >= {args.min_commitment}",
        commitment_n >= args.min_commitment,
        "warning",
        f"commitment diagnostic records={commitment_n}; main tables are more stable with at least {args.min_commitment}.",
    )
    _add(
        checks,
        f"repo diversity >= {args.min_repos}",
        repo_n >= args.min_repos,
        "warning",
        f"repo_count={repo_n}; one-repo pilots are fine, but main claims should use broader CooperBench coverage when available.",
    )

    workspace = _csv_rows(root, zf, f"analysis/{tag}/cooperbench_workspace_wccu/cooperbench_workspace_table.csv")
    commitment_rows = _csv_rows(root, zf, f"analysis/{tag}/cooperbench_commitment_stale_wccu/cooperbench_commitment_table.csv")
    ablation = _csv_rows(root, zf, f"analysis/{tag}/live_llm_wccu_ablation.csv")

    ws_wccu = _condition(workspace, "adaptive_wccu_execution_trace")
    ws_append = _condition(workspace, "uniform_append_only")
    if ws_wccu:
        _add(
            checks,
            "workspace WCCU execution safety pass",
            _int(ws_wccu, "safety_pass") == _int(ws_wccu, "runs") and _int(ws_wccu, "runs") > 0,
            "error",
            f"WCCU execution workspace safety_pass={ws_wccu.get('safety_pass')} runs={ws_wccu.get('runs')}.",
        )
        _add(
            checks,
            "workspace WCCU execution selected lock lane",
            _int(ws_wccu, "lock_lane_selected") == _int(ws_wccu, "runs") and _int(ws_wccu, "runs") > 0,
            "warning",
            f"lock_lane_selected={ws_wccu.get('lock_lane_selected')} runs={ws_wccu.get('runs')}.",
        )
    else:
        _add(checks, "workspace WCCU row exists", False, "error", "Missing adaptive_wccu_execution_trace row.")

    if ws_append:
        _add(
            checks,
            "workspace append-only exposes unsafe baseline",
            _int(ws_append, "unsafe_auto_commit_count") > 0,
            "warning",
            f"append-only unsafe_auto_commit_count={ws_append.get('unsafe_auto_commit_count')}.",
        )

    cm_wccu = _condition(commitment_rows, "adaptive_wccu_execution_trace")
    cm_no_wccu = _condition(commitment_rows, "adaptive_policy")
    cm_no_read = _condition(commitment_rows, "adaptive_wccu_no_read_validation")
    if cm_wccu:
        _add(
            checks,
            "commitment WCCU execution rejects stale dependency",
            _int(cm_wccu, "stale_dependency_accepted_count") == 0 and _int(cm_wccu, "runs") > 0,
            "error",
            f"WCCU stale_dependency_accepted_count={cm_wccu.get('stale_dependency_accepted_count')}.",
        )
        _add(
            checks,
            "commitment WCCU produces interventions",
            _int(cm_wccu, "wccu_intervention_count") > 0,
            "warning",
            f"WCCU intervention_count={cm_wccu.get('wccu_intervention_count')}.",
        )
    else:
        _add(checks, "commitment WCCU row exists", False, "error", "Missing adaptive_wccu_execution_trace row.")

    if cm_no_wccu:
        _add(
            checks,
            "commitment adaptive-no-WCCU exposes stale baseline",
            _int(cm_no_wccu, "stale_dependency_accepted_count") > 0,
            "warning",
            f"adaptive no-WCCU stale_dependency_accepted_count={cm_no_wccu.get('stale_dependency_accepted_count')}.",
        )
    if cm_no_read:
        _add(
            checks,
            "freshness ablation fails as expected",
            _int(cm_no_read, "stale_dependency_accepted_count") > 0,
            "warning",
            f"no-read-validation stale_dependency_accepted_count={cm_no_read.get('stale_dependency_accepted_count')}.",
        )

    _add(
        checks,
        "live LLM ablation table exists",
        len(ablation) > 0,
        "warning",
        f"rows={len(ablation)}.",
    )

    errors = [c for c in checks if c["severity"] == "error" and not c["ok"]]
    warnings = [c for c in checks if c["severity"] == "warning" and not c["ok"]]
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    result = {
        "tag": tag,
        "status": status,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": checks,
        "recommendation": (
            "Do not scale or cite until error checks pass."
            if errors
            else (
                "Safe to scale to main, but address warnings before paper claims."
                if warnings
                else "Ready for main run or citation."
            )
        ),
    }

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"# WCCU primary run readiness: `{tag}`",
        "",
        f"Status: **{status}**",
        "",
        result["recommendation"],
        "",
        "| Check | Severity | Result | Detail |",
        "|---|---:|---:|---|",
    ]
    for c in checks:
        mark = "PASS" if c["ok"] else "FAIL"
        lines.append(f"| {c['name']} | {c['severity']} | {mark} | {c['detail']} |")
    md = "\n".join(lines) + "\n"
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md, encoding="utf-8")
    else:
        print(md)

    if zf is not None:
        zf.close()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
