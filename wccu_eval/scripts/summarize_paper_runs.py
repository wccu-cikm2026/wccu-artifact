from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from statistics import mean


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Summarize live-LLM WCCU ablation tables across paper runs.')
    ap.add_argument('--glob', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--out-md', required=True)
    args = ap.parse_args(argv)

    rows = []
    for path in sorted(glob.glob(args.glob)):
        tag = Path(path).parts[1] if len(Path(path).parts) > 1 else Path(path).stem
        for row in _read_csv(path):
            row = dict(row)
            row['run_tag'] = tag
            rows.append(row)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(args.out_csv).write_text('', encoding='utf-8')
        Path(args.out_md).write_text('No matching run-level ablation CSV files found.\n', encoding='utf-8')
        print('No matching run-level ablation CSV files found.')
        return 0

    keys = sorted(rows[0].keys())
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    group_fields = [k for k in ('family', 'condition', 'policy_mode') if k in rows[0]]
    metric_fields = [k for k in rows[0] if k not in set(group_fields + ['run_tag'])]
    numeric_metrics = []
    for m in metric_fields:
        try:
            [float(r[m]) for r in rows if r.get(m) not in ('', None)]
            numeric_metrics.append(m)
        except Exception:
            pass

    grouped = {}
    for r in rows:
        key = tuple(r.get(f, '') for f in group_fields) or ('all',)
        grouped.setdefault(key, []).append(r)

    lines = ['# Paper live-LLM run summary', '', f'Input tables: `{args.glob}`', '', f'Rows: {len(rows)}', '']
    if group_fields and numeric_metrics:
        show_metrics = numeric_metrics[:8]
        header = group_fields + ['runs'] + [f'mean_{m}' for m in show_metrics]
        lines.append('| ' + ' | '.join(header) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
        for key, rs in sorted(grouped.items()):
            vals = list(key) + [str(len({r["run_tag"] for r in rs}))]
            for m in show_metrics:
                nums = [float(r[m]) for r in rs if r.get(m) not in ('', None)]
                vals.append(f'{mean(nums):.4f}' if nums else '')
            lines.append('| ' + ' | '.join(vals) + ' |')
    Path(args.out_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {args.out_csv} and {args.out_md}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
