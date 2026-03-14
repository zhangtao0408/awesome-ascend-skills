# Benchmark Report

## Test Summary

| Item | Value |
|------|-------|
| Service | `{base_url}` |
| Model | `{model}` |
| Backend | `{backend}` |
| Dataset | `{dataset}` |
| Mode | {mode} |
| Time | {timestamp} |
| Environment | {environment} |

## Configuration

| Parameter | Value |
|-----------|-------|
| num_prompts | {num_prompts} |
| max_concurrency | {max_concurrency} |
| request_rate | {request_rate} |
| input_len | {input_len} |
| output_len | {output_len} |
| num_warmups | {num_warmups} |

## Results

{results_table}

## SLO Compliance

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| TTFT P99 | {slo_ttft_target} | {slo_ttft_actual} | {slo_ttft_status} |
| TPOT P99 | {slo_tpot_target} | {slo_tpot_actual} | {slo_tpot_status} |
| E2E P99 | {slo_e2e_target} | {slo_e2e_actual} | {slo_e2e_status} |
| Success Rate | {slo_success_target} | {slo_success_actual} | {slo_success_status} |

## Recommendation

{recommendation}

## Result Files

- Result directory: `{result_dir}`
- Result files:
{result_files_list}
