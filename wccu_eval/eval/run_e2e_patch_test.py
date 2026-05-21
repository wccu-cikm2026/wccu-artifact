from __future__ import annotations

import argparse
import json

from wccu_eval.e2e.patch_test_runner import run_e2e_patch_tests

DEFAULT_CONDITIONS = 'adaptive_wccu_execution_trace,adaptive_readset_occ,adaptive_policy,uniform_review_gated'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run end-to-end patch/apply/test diagnostics gated by substrate commit decisions.')
    parser.add_argument('--input', required=True, help='JSON/JSONL task file with repo/base_files, patches, and test commands.')
    parser.add_argument('--conditions', default=DEFAULT_CONDITIONS)
    parser.add_argument('--out', default='results/e2e_patch_test.json')
    parser.add_argument('--work-dir', default='runs/e2e_patch_test')
    parser.add_argument('--timeout-s', type=int, default=120)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--apply-reviewed', action='store_true', help='Apply review-routed patches too; by default only auto-committed patches are applied.')
    parser.add_argument('--isolate-rows', action='store_true', help='Run each task/condition cell in a fresh Python process with a row-level timeout.')
    parser.add_argument('--row-timeout-s', type=int, default=60, help='Process-level timeout for each task/condition cell when row isolation is enabled.')
    args = parser.parse_args(argv)
    payload = run_e2e_patch_tests(input_path=args.input, conditions=args.conditions, out=args.out, work_dir=args.work_dir, apply_reviewed=args.apply_reviewed, timeout_s=args.timeout_s, limit=args.limit, isolate_rows=args.isolate_rows, row_timeout_s=args.row_timeout_s)
    print(json.dumps({'ok': True, 'out': payload['args']['out'], 'rows': len(payload['results'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
