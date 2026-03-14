# Auto-Optimization Strategy

Find the maximum throughput or concurrency that satisfies user-defined SLO constraints.

---

## Problem Definition

Given:
- A running vLLM inference service
- A set of SLO constraints (e.g., TTFT P99 < 500ms, success rate > 95%)
- A search dimension (concurrency or request rate)

Find:
- The maximum value of the search dimension where all SLO constraints are satisfied

---

## Search Modes

### Mode A (Default): Search `--max-concurrency` only
- Do not set `--request-rate` (let requests dispatch as fast as possible)
- Probe sequence: 1, 2, 4, 8, 16, 32, 64, 128, 256...
- `num_prompts = concurrency × multiplier`

### Mode B: Search `--request-rate` only
- Do not set `--max-concurrency`
- Probe sequence: 1, 2, 4, 8, 16, 32, 64...
- `num_prompts = rate × multiplier`

### Mode C: Fixed `--max-concurrency`, search `--request-rate`
- User specifies a fixed concurrency (e.g., `--max-concurrency 32`)
- Search for optimal request rate under that concurrency cap
- Use case: user knows concurrency limit, wants best send rate

### Mode D: Fixed `--request-rate`, search `--max-concurrency`
- User specifies a fixed rate (e.g., `--request-rate 100`)
- Search for minimal concurrency that sustains that rate
- Use case: user has a target QPS, wants to know how many concurrent slots needed

### Mode E: Two-Dimensional Joint Search
1. First run Mode A to find optimal concurrency C*
2. Then fix concurrency near C* and run Mode C to find optimal rate R*
3. Optionally fine-tune: try (C*±δ, R*±δ) combinations
- Larger search space, significantly more iterations
- Agent must warn user about increased time cost before starting

---

## num_prompts Multiplier

The number of requests per iteration should scale with the search variable to ensure enough samples for meaningful P99 statistics.

| Phase | Multiplier Range | Default | Purpose |
|-------|-----------------|---------|---------|
| Warmup | — | 50 (fixed) | Just warm up the service |
| Coarse Probe | 2~4× | 3× | Speed priority, find rough boundaries |
| Fine Search | 5~8× | 6× | Accuracy priority, narrow down |
| Validation | 8~10× | 10× | High confidence, confirm final result |

**Formula:** `num_prompts = max(search_value × multiplier, minimum_floor)`
- Coarse minimum floor: 50
- Fine minimum floor: 100
- Validation minimum floor: 200

Agent should confirm multiplier preferences with user before starting. If user emphasizes speed, use lower multipliers. If user emphasizes accuracy, use higher multipliers.

---

## Algorithm: 4-Phase Search

### Phase 1: Warmup

```
Run benchmark:
  --max-concurrency 1 (or --request-rate 1.0)
  --num-prompts 50
  --num-warmups 5

Check SLO compliance.
If SLO already fails → ABORT
  Report: "Service cannot meet SLO constraints even at minimum load."
  Suggest: Check service health, reduce SLO strictness, or optimize service.
```

### Phase 2: Exponential Probe (find upper bound)

```
search_value = 1
last_good = None

loop (max 15 iterations):
  num_prompts = max(search_value × coarse_multiplier, 50)

  Run benchmark at search_value
  Parse results: extract TTFT P99, TPOT P99, E2E P99, success rate

  Check SLO compliance:
    ttft_p99 <= slo_ttft_p99  (if specified)
    tpot_p99 <= slo_tpot_p99  (if specified)
    e2e_p99  <= slo_e2e_p99   (if specified)
    success_rate >= slo_success_rate  (default 95%)

  if ALL SLOs met:
    last_good = search_value
    search_value = search_value × 2
  else:
    upper_bound = search_value
    lower_bound = last_good
    → proceed to Phase 3

if loop exhausted without SLO violation:
  Report: "Service not saturated at search_value={search_value}."
  The last tested (passing) value is the best we found.
  Suggest: Increase probe range or the service has excess capacity.
```

### Phase 3: Binary Search (find optimal point)

```
lower = lower_bound
upper = upper_bound

loop (max 8 iterations):
  mid = (lower + upper) / 2
  num_prompts = max(mid × fine_multiplier, 100)

  Run benchmark at mid
  Check SLO compliance (same rules as Phase 2)

  if ALL SLOs met:
    lower = mid
  else:
    upper = mid

  Convergence check:
    if (upper - lower) / lower < 0.05:  # 5% tolerance
      → proceed to Phase 4

optimal_candidate = lower  (last known good)
```

### Phase 4: Validation

```
num_prompts = max(optimal_candidate × validation_multiplier, 200)

Run benchmark at optimal_candidate

if ALL SLOs met:
  → CONFIRMED OPTIMAL
else:
  retry_value = optimal_candidate × 0.9  # reduce by 10%
  Retry validation (max 3 retries)

  if still failing after 3 retries:
    Report: "Could not confirm stable optimal point."
    Return last confirmed good value from Phase 3.
```

---

## SLO Compliance Check

For each benchmark iteration, extract metrics and check:

```python
def check_slo(results, slo_targets):
    """
    results: parsed benchmark output
    slo_targets: dict of {metric: threshold}
    Returns: (passed: bool, violations: list)
    """
    violations = []

    if 'ttft_p99' in slo_targets:
        actual = results['percentiles_ttft_ms']['p99']
        if actual > slo_targets['ttft_p99']:
            violations.append(f"TTFT P99: {actual:.1f}ms > {slo_targets['ttft_p99']}ms")

    if 'tpot_p99' in slo_targets:
        actual = results['percentiles_tpot_ms']['p99']
        if actual > slo_targets['tpot_p99']:
            violations.append(f"TPOT P99: {actual:.1f}ms > {slo_targets['tpot_p99']}ms")

    if 'e2e_p99' in slo_targets:
        actual = results['percentiles_e2el_ms']['p99']
        if actual > slo_targets['e2e_p99']:
            violations.append(f"E2E P99: {actual:.1f}ms > {slo_targets['e2e_p99']}ms")

    if 'success_rate' in slo_targets:
        total = results['completed'] + results['failed']
        actual = results['completed'] / total * 100 if total > 0 else 0
        if actual < slo_targets['success_rate']:
            violations.append(f"Success rate: {actual:.1f}% < {slo_targets['success_rate']}%")

    return len(violations) == 0, violations
```

---

## Edge Cases

### Service saturated at minimum load
- Phase 1 warmup fails SLO
- Action: abort immediately, report that service cannot meet constraints
- Suggest checking service health, available resources, or relaxing SLOs

### Service never saturates
- Phase 2 probe reaches very high values (e.g., concurrency 256+) without SLO violation
- Action: report the maximum tested value as the result
- Note: "Service has excess capacity at tested range"

### Unstable metrics
- P99 values fluctuate >20% between identical runs
- Detection: if the same search_value gives different SLO pass/fail on retry
- Action: increase `num_prompts` multiplier, add cooldown between runs (5-10s)

### Service crash during benchmark
- Benchmark process exits with non-zero code
- Action: wait 10s, probe service health. If recovered, retry once. If not, abort.

### Partial SLO failure
- One SLO passes, another fails (e.g., TTFT OK but TPOT too high)
- Action: treat as SLO violation (ALL constraints must be met)

---

## Output Format

### optimization_report.json

```json
{
  "search_mode": "A",
  "search_dimension": "max_concurrency",
  "optimal_value": 16,
  "slo_targets": {
    "ttft_p99": 500,
    "tpot_p99": 50,
    "success_rate": 95
  },
  "metrics_at_optimal": {
    "request_throughput": 45.2,
    "output_throughput": 5780,
    "ttft_p99": 423.5,
    "tpot_p99": 42.1,
    "e2e_p99": 2830.0,
    "success_rate": 99.2
  },
  "slo_compliance": [
    {"metric": "TTFT P99", "target": "<=500ms", "actual": "423.5ms", "pass": true},
    {"metric": "TPOT P99", "target": "<=50ms", "actual": "42.1ms", "pass": true},
    {"metric": "Success Rate", "target": ">=95%", "actual": "99.2%", "pass": true}
  ],
  "exploration_history": [
    {"phase": "warmup", "value": 1, "num_prompts": 50, "slo_pass": true, "file": "warmup.json"},
    {"phase": "probe", "value": 1, "num_prompts": 3, "slo_pass": true, "file": "probe_001_c1.json"},
    {"phase": "probe", "value": 2, "num_prompts": 6, "slo_pass": true, "file": "probe_002_c2.json"},
    {"phase": "probe", "value": 4, "num_prompts": 12, "slo_pass": true, "file": "probe_003_c4.json"},
    {"phase": "probe", "value": 8, "num_prompts": 24, "slo_pass": true, "file": "probe_004_c8.json"},
    {"phase": "probe", "value": 16, "num_prompts": 48, "slo_pass": true, "file": "probe_005_c16.json"},
    {"phase": "probe", "value": 32, "num_prompts": 96, "slo_pass": false, "file": "probe_006_c32.json"},
    {"phase": "search", "value": 24, "num_prompts": 144, "slo_pass": true, "file": "search_001_c24.json"},
    {"phase": "search", "value": 28, "num_prompts": 168, "slo_pass": false, "file": "search_002_c28.json"},
    {"phase": "search", "value": 26, "num_prompts": 156, "slo_pass": false, "file": "search_003_c26.json"},
    {"phase": "validation", "value": 24, "num_prompts": 240, "slo_pass": true, "file": "validation.json"}
  ],
  "recommendation": "Optimal max_concurrency = 24. At this concurrency, the service achieves 45.2 req/s throughput while keeping TTFT P99 at 423.5ms (target: <=500ms) and TPOT P99 at 42.1ms (target: <=50ms)."
}
```
