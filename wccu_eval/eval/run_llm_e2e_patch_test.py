from __future__ import annotations

import argparse
import json
from pathlib import Path

from wccu_eval.e2e.llm_patch_generator import generate_llm_patch_tasks
from wccu_eval.e2e.patch_test_runner import run_e2e_patch_tests
from wccu_eval.utils import clean, ensure_dir, write_json

DEFAULT_CONDITIONS = 'adaptive_wccu_execution_trace,adaptive_readset_occ,adaptive_policy,uniform_review_gated'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Generate LLM patch proposals, then run end-to-end patch/apply/test diagnostics gated by substrate policies.')
    parser.add_argument('--input', required=True, help='JSON/JSONL task file with base_files/repo and patch objectives.')
    parser.add_argument('--generated-tasks-out', default='results/llm_e2e/generated_patch_tasks.jsonl')
    parser.add_argument('--conditions', default=DEFAULT_CONDITIONS)
    parser.add_argument('--out', default='results/llm_e2e/e2e_patch_test.json')
    parser.add_argument('--work-dir', default='runs/llm_e2e_patch_test')
    parser.add_argument('--generation-log', default='results/llm_e2e/llm_patch_generation.jsonl')
    parser.add_argument('--provider', default='')
    parser.add_argument('--model', default='')
    parser.add_argument('--max-output-tokens', type=int, default=2500)
    parser.add_argument('--timeout-s', type=int, default=120)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--task-types', default='', help='Comma-separated task types to generate/run, e.g. commitment_staleness or independent,commitment_staleness. Filtering happens before --limit.')
    parser.add_argument('--row-timeout-s', type=int, default=90)
    parser.add_argument('--isolate-rows', action='store_true')
    parser.add_argument('--mock-from-prepared', action='store_true', help='Do not call an LLM; copy prepared patch_file text into patch_text. Useful for smoke tests.')
    parser.add_argument('--output-mode', default='file_edits', choices=['file_edits', 'diff'], help='LLM patch proposal format. file_edits asks for complete final file contents and materializes patches in the harness; diff asks the model for raw unified diff.')
    parser.add_argument('--no-validate-generated-patches', action='store_true')
    parser.add_argument('--apply-reviewed', action='store_true')
    args = parser.parse_args(argv)
    ensure_dir(Path(args.generated_tasks_out).parent)
    llm_config = {
        'provider': clean(args.provider),
        'model': clean(args.model),
        'max_output_tokens': args.max_output_tokens,
        'timeout_seconds': args.timeout_s,
        'error_log_path': args.generation_log + '.provider_errors.jsonl',
    }
    gen = generate_llm_patch_tasks(
        input_path=args.input,
        out_path=args.generated_tasks_out,
        limit=args.limit,
        llm_config=llm_config,
        mock_from_prepared=args.mock_from_prepared,
        validate_patch=not args.no_validate_generated_patches,
        timeout_s=args.timeout_s,
        generation_log_path=args.generation_log,
        task_types=[x.strip() for x in args.task_types.split(',') if x.strip()],
        output_mode=args.output_mode,
    )
    payload = run_e2e_patch_tests(
        input_path=args.generated_tasks_out,
        conditions=args.conditions,
        out=args.out,
        work_dir=args.work_dir,
        apply_reviewed=args.apply_reviewed,
        timeout_s=args.timeout_s,
        limit=0,
        isolate_rows=args.isolate_rows,
        row_timeout_s=args.row_timeout_s,
    )
    summary = {'ok': True, 'generated_tasks_out': args.generated_tasks_out, 'generation': {k: gen.get(k) for k in ('tasks','patches','parse_success','validation_success','edit_parse_success','materialized_patch_valid','output_mode','mock_from_prepared')}, 'e2e_rows': len(payload.get('results', [])), 'out': args.out}
    write_json(str(Path(args.out).with_suffix('.summary.json')), summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
