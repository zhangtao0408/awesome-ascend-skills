---
name: vllm-bench-serve
description: >-
  Interactive online benchmark orchestrator for vLLM inference services using
  `vllm bench serve`. Supports single benchmarks, multi-case batch execution
  with result aggregation, and auto-optimization to find optimal
  concurrency/throughput under latency SLO constraints (TTFT, TPOT, P99,
  success rate). Use this skill whenever the user wants to benchmark, stress
  test, or measure performance of a running LLM/multimodal/embedding inference
  service — even if they don't say "vllm bench serve" explicitly. Do NOT use
  for offline inference throughput, service deployment/startup, profiling/tracing,
  health checks only, or analyzing existing benchmark results without running
  new tests.
keywords:
  - vllm bench serve
  - benchmark
  - online benchmark
  - 在线性能测试
  - 推理服务性能
  - throughput
  - latency
  - TTFT
  - TPOT
  - P99
  - concurrency
  - 并发测试
  - auto-optimize
  - 寻优
  - SLO
  - request rate
  - stress test
  - 压力测试
---

# vllm-bench-serve — Online Benchmark Orchestrator

## 1. Scope & Boundaries

**This skill does:**
- Execute single or batch `vllm bench serve` online benchmarks against running inference services
- Aggregate and compare results across multiple test cases
- Auto-optimize: search for optimal concurrency/throughput given SLO constraints

**This skill does NOT do (defer to other skills or decline):**
- Start/deploy vLLM services → use `vllm-ascend-server`
- Offline batch inference throughput tests → use `vllm-ascend`
- Profiling / tracing (torch profiler, perfetto, NPU profiling) → out of scope
- Health checks only (just checking `/v1/models`) → simple curl, no skill needed
- Download/clean/convert datasets without running benchmarks → out of scope
- Analyze existing benchmark results without running new tests → out of scope

**Related skills:**
- `vllm-ascend-server`: Deploy and manage vLLM inference services
- `remote-server-guide`: SSH connection, remote server access, container management

---

## 2. Trigger Examples

### Should Trigger

1. "帮我测试一下 0.0.0.0:6531 的推理服务的性能。"
   → Probe service, discover model, confirm test case, run benchmark.

2. "帮我依次测试 0.0.0.0:1234 和 0.0.0.0:4321 这两个推理服务在 1,2,4,8,16,32 并发下的性能。"
   → Batch mode: multiple services × multiple concurrency levels, aggregate results.

3. "我想要知道 qwen3-30b 这个模型的最优吞吐是多少。"
   → Ask for service address, dataset, SLO constraints, then auto-optimize.

4. "帮我用 sharegpt 数据集测一下这个服务在不同 request rate 下的时延表现。"
   → Batch mode: sweep request rates, compare latency metrics.

5. "我需要找到 TTFT P99 在 500ms 以下的最大并发数。"
   → Auto-optimize with TTFT P99 SLO constraint.

### Should NOT Trigger

1. "帮我采集 qwen3-8b 的 profiling。" → Profiling, not benchmark.
2. "帮我跑一下这个模型的离线吞吐测试。" → Offline benchmark, use `vllm-ascend`.
3. "帮我把 vLLM 服务启动起来再测。" → Service deployment, use `vllm-ascend-server`.
4. "帮我看看这个 benchmark 结果文件怎么解读。" → Result analysis only, no new test.
5. "我已经有一批 vllm bench 的 JSON 结果了，帮我生成测试报告。" → Report from existing data only.

---

## 3. Workflow Overview

Execute phases in order. Generate a plan at the start with all phases listed. Mark each phase complete before proceeding. Check results at each phase boundary.

```
Phase 0: Execution Environment  →  Can we run vllm bench serve here?
    ↓
Phase 1: Service & Model        →  What service? What model?
    ↓
Phase 2: Mode Selection          →  Single / Batch / Auto-Optimize?
    ↓
Phase 3: Backend Selection       →  Text / Multimodal / Embedding / Reranking?
    ↓
Phase 4: Dataset Selection       →  Which dataset? What parameters?
    ↓
Phase 5: Benchmark Parameters    →  Concurrency? Rate? Prompts? SLOs?
    ↓
Phase 6: Parameter Validation    →  Are all combinations compatible?
    ↓
Phase 7: Command Gen & Execute   →  Generate commands, run benchmarks
    ↓
Phase 8: Results & Report        →  Aggregate, compare, recommend
```

For **Auto-Optimize**, Phase 7 and 8 form an iterative loop (probe → search → validate).

---

## 4. Phase 0 — Execution Environment

Before anything else, determine WHERE the benchmark client will run. The benchmark client sends HTTP requests to the service, so it needs network access to the target service and `vllm` installed.

### Decision Tree

1. **Can we access the target service from the current environment?**
   - If user gave `ip:port`, try: `curl -s --connect-timeout 5 http://ip:port/health`
   - If reachable → candidate for local execution
   - If not reachable → need remote execution environment

2. **Is `vllm bench serve` available here?**
   - Run: `scripts/check_bench_env.sh` (or manually: `python3 -c "import vllm; print(vllm.__version__)"` then `vllm bench serve --help`)
   - If available → proceed locally
   - If not → need a different environment

3. **If local execution is not viable:**
   - Ask user where benchmark should run (remote server? container?)
   - Use `remote-server-guide` skill to establish connection
   - Once in the remote environment, re-run environment checks
   - All subsequent phases execute in that environment

4. **Check writable directory** for result storage
   - Test: `touch /tmp/bench_test_write && rm /tmp/bench_test_write`
   - If not writable, ask user for an alternative result directory

> **Remote/Container execution**: When benchmark runs remotely, construct `vllm bench serve` commands directly in the remote shell rather than transferring scripts. Read results via `cat`/`jq` in the remote session. See `references/environment-checks.md` for details.

---

## 5. Phase 1 — Service & Model

### Service Address
- If user provided `ip:port` or `base_url` → use it
- If user only gave a model name → ask for the service address
- Normalize to `base_url` format: `http://ip:port`

### Model Discovery
Run the service probe (or manually: `curl -s http://ip:port/v1/models`):
```bash
# Using the probe script (local execution):
bash <skill-path>/scripts/probe_service.sh --base-url http://ip:port

# Or manually:
curl -s http://ip:port/v1/models | python3 -c "
import sys, json
d = json.load(sys.stdin)
for m in d['data']:
    print(f\"id={m['id']}  root={m['root']}\")
"
```

- If exactly one model → confirm with user
- If multiple models → let user choose
- If no models → service may not be ready, report error

### Understanding `--model` vs `--served-model-name`

These two parameters serve different purposes in `vllm bench serve`:

| Parameter | Purpose | Used For |
|-----------|---------|----------|
| `--model` | Model weight path / tokenizer identifier | Tokenizer initialization (needed for `random`, `random-mm`, `random-rerank` datasets that generate synthetic tokens) |
| `--served-model-name` | The model name used in API requests | The `"model"` field in HTTP request bodies sent to the service |

**How `vllm bench serve` resolves these internally:**
- If `--model` is NOT specified: auto-fetches from `/v1/models` endpoint — `id` → `served_model_name`, `root` → `model` (weight path for tokenizer)
- If `--model` IS specified: uses it for tokenizer. If `--served-model-name` is also given, uses that for API requests; otherwise falls back to using `--model` value as the API model name

**When to use each:**
- If the service is accessible and benchmark runs on the same network: **omit `--model`** to let auto-fetch handle both (simplest approach). Confirm with user that the auto-detected model is correct.
- If auto-fetch is not available, or the `/v1/models` `root` field does not point to a valid tokenizer path:
  - `--model` should be the weight path (e.g., `/path/to/Qwen3-30B-Instruct`) — ask the user for it
  - `--served-model-name` should be the name the service recognizes (e.g., `qwen3-30b`) — can be read from `/v1/models` `id` field
- For `--skip-tokenizer-init` scenarios (embedding/reranking backends): `--model` is less critical but `--served-model-name` must still match the service

**Always confirm with the user** which model identifier to use, especially when the weight path and served name differ.

### Output of This Phase
- `base_url`: confirmed service URL
- `model`: weight path / tokenizer identifier (for `--model`)
- `served_model_name`: API model name (for `--served-model-name`), may be the same as `model`

---

## 6. Phase 2 — Mode Selection

Determine the benchmark mode from user intent:

| User Intent | Mode | Description |
|-------------|------|-------------|
| "测试一下性能" / single test | **Single** | One benchmark run with one parameter set |
| "多个并发/参数/服务对比" | **Batch** | Multiple cases, aggregate comparison table |
| "找最优并发/吞吐" / "寻优" | **Auto-Optimize** | Iterative search for optimal configuration |

If ambiguous, ask the user which mode they need. Explain the three options briefly.

---

## 7. Phase 3 — Backend Selection

The backend determines the request format. **You must also set `--endpoint` to the matching API path** — `vllm bench serve` defaults `--endpoint` to `/v1/completions`, which is only correct for `openai` and `vllm` backends. All other backends will fail without the correct `--endpoint`.

| Model Type | Recommended Backend | Required `--endpoint` |
|------------|-------------------|-----------------------|
| Text LLM (chat) | `openai-chat` | `/v1/chat/completions` |
| Text LLM (completion) | `openai` or `vllm` | `/v1/completions` (default) |
| Multimodal (VLM) | `openai-chat` | `/v1/chat/completions` |
| Embedding | `openai-embeddings` | `/v1/embeddings` |
| Reranking | `vllm-rerank` | `/v1/rerank` |
| Audio/ASR | `openai-audio` | `/v1/audio/transcriptions` |

For most LLM benchmarks, **`openai-chat` is the recommended default** because it matches real production usage patterns.

> **CRITICAL**: When using `--backend openai-chat`, you MUST also pass `--endpoint /v1/chat/completions`. Without it, the benchmark will fail with a URL validation error. The `scripts/generate_bench_cmd.py` and `scripts/auto_optimize.py` auto-inject the correct endpoint, but when constructing commands manually, always include `--endpoint`.

Present the options to the user with short explanations. If the user doesn't have a preference, use `openai-chat` for text LLMs.

> For the full backend list including `openai-embeddings-chat`, `openai-embeddings-clip`, `openai-embeddings-vlm2vec`, `infinity-embeddings`, `infinity-embeddings-clip`, `vllm-pooling`, see `references/backend-mapping.md`.

---

## 8. Phase 4 — Dataset Selection

Use a **two-step interaction**: first select the dataset category, then confirm specific parameters.

### Step 1: Dataset Category

| Scenario | Dataset | Key Parameters |
|----------|---------|---------------|
| Quick synthetic test | `random` | `--random-input-len`, `--random-output-len` |
| Random length range | `random` | + `--random-range-ratio` (0~1, e.g. 0.5 = ±50%) |
| Prefix cache test | `random` + `--random-prefix-len` | or `prefix_repetition` |
| Real conversation distribution | `sharegpt` | `--dataset-path` (auto-downloads if not local) |
| Burst traffic simulation | `burstgpt` | `--dataset-path` |
| Custom workload (JSONL) | `custom` | `--dataset-path` (JSONL with "prompt" field) |
| Multimodal (images) | `random-mm` or `custom_mm` | MM-specific params |
| Embedding benchmark | `random` | with embedding backend, `--random-batch-size` |
| Reranking benchmark | `random-rerank` | must use `vllm-rerank` backend |
| HuggingFace dataset | `hf` | `--dataset-path HF_ID`, `--hf-split`, `--hf-subset` |
| Speculative decoding test | `spec_bench` | `--spec-bench-category` |

### Step 2: Dataset-Specific Parameters

After the user selects a category, confirm the specific parameters. Different datasets need different parameters — do not assume a universal template.

**For `random` dataset** (most common for quick benchmarks):
- `--random-input-len` (default 1024): input token count
- `--random-output-len` (default 128): output token count
- `--random-range-ratio` (default 0.0): length randomization range, 0=fixed, 0.5=±50%
- `--random-prefix-len` (default 0): fixed prefix length for prefix cache testing

> Note: `--input-len` and `--output-len` are convenience aliases that map to dataset-specific parameters (e.g., `--random-input-len`, `--sonnet-input-len`). Prefer using dataset-specific parameter names to be explicit.

**For `sharegpt` dataset**:
- `--dataset-path`: path to ShareGPT JSON file (will auto-download from HuggingFace if not provided)
- `--sharegpt-output-len`: optional output length override

**For `custom` dataset**:
- `--dataset-path`: path to JSONL file, each line must have `"prompt"` field, optional `"output_tokens"`
- `--custom-output-len` (default 256): output length if not in data

> For the complete dataset compatibility matrix and all dataset-specific parameters, see `references/dataset-guide.md`.

---

## 9. Phase 5 — Benchmark Parameters

### Core Parameters (always confirm)

**Load Control** (choose one approach):
- `--max-concurrency N`: limit concurrent requests (recommended for most tests)
- `--request-rate R`: requests per second, Poisson arrival (default: inf = all at once)
- These can also be combined (e.g., fixed rate with concurrency cap)
- For auto-optimize mode, the search dimension determines which to vary

**Request Count:**
- `--num-prompts N` (default 1000): total requests to send
- For auto-optimize, this is calculated as `search_value × multiplier` (see Phase 8)

**Warmup:**
- `--num-warmups N` (default 0): warmup requests before measurement
- Recommend 3-10 for graph-mode compiled services

### SLO & Metrics (for auto-optimize or SLO-aware benchmarks)

- `--goodput "METRIC:VALUE_MS"`: SLO targets, e.g. `--goodput "ttft:500" --goodput "tpot:50"`
  - Valid metrics: `ttft`, `tpot`, `e2el` (values in milliseconds)
- `--percentile-metrics "ttft,tpot,itl,e2el"`: which metrics to report percentiles for
- `--metric-percentiles "50,90,95,99"`: which percentiles to compute

### Sampling Parameters (usually defaults are fine)
- `--temperature`, `--top-p`, `--top-k`, etc.
- Only effective with OpenAI-compatible backends (`openai`, `openai-chat`, `vllm`)
- Typically leave at defaults unless user has specific requirements

> For the exhaustive parameter list, see `references/param-reference.md`.

---

## 10. Phase 6 — Parameter Validation

Before executing, validate parameter compatibility. Use `scripts/validate_params.py` (local) or check manually:

**Critical rules:**
1. **`--endpoint` must match the backend** — e.g., `openai-chat` requires `--endpoint /v1/chat/completions`. Only `openai`/`vllm` can use the default `/v1/completions`. The scripts auto-inject this, but verify when constructing commands manually.
2. `random-rerank` dataset MUST use `vllm-rerank` backend
3. Multimodal datasets (`random-mm`, `custom_mm`) require `openai-chat` backend
4. `sharegpt`/`custom`/`custom_mm` require `--dataset-path`
5. `--request-rate` and `--max-concurrency` CAN be combined, but understand the interaction
6. `--goodput` metric names must be `ttft`, `tpot`, or `e2el`
7. If `--num-prompts` < 100, warn about insufficient statistical significance for P99

If validation fails, explain the issue to the user and suggest corrections. Do not proceed until all errors are resolved.

---

## 11. Phase 7 — Command Generation & Execution

### Result Archival (ALWAYS applied)

Every `vllm bench serve` command MUST include these flags:
```
--save-result --save-detailed --result-dir <dir> --result-filename <name>
```

**Naming convention:**
```
bench_{model_short}_{dataset}_{backend}_{YYYYMMDD_HHMMSS}.json
```
- `model_short`: last segment of model name, max 30 chars, `/` → `-`

**Directory structure:**
```
bench_results/
├── single/        # Single runs
├── batch/         # Batch sessions (subdirectory per batch)
│   └── batch_YYYYMMDD_HHMMSS/
├── optimize/      # Auto-optimize sessions (subdirectory per session)
│   └── opt_YYYYMMDD_HHMMSS/
└── logs/          # Execution logs
```

### Single Execution

```bash
# Generate command (local) — auto-injects correct --endpoint:
python3 <skill-path>/scripts/generate_bench_cmd.py \
  --base-url http://ip:port --model /path/to/weights --served-model-name MODEL_NAME \
  --backend openai-chat \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --max-concurrency 8 --num-prompts 500

# Execute:
bash <skill-path>/scripts/run_bench.sh "<generated_command>"
```

Or construct the command directly (**note `--endpoint` is required**):
```bash
vllm bench serve \
  --base-url http://ip:port --model /path/to/weights --served-model-name MODEL_NAME \
  --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --max-concurrency 8 --num-prompts 500 \
  --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
  --save-result --save-detailed \
  --result-dir ./bench_results/single \
  --result-filename bench_MODEL_random_openai-chat_TIMESTAMP.json
```

> Note: If `--model` is omitted, `vllm bench serve` auto-fetches both the weight path and served name from the service's `/v1/models` endpoint. This is the simplest approach when the service is accessible. See Phase 1 for details on `--model` vs `--served-model-name`.

### Batch Execution

For each case in the batch, generate and execute sequentially. Track pass/fail per case. After all cases complete, aggregate results.

```bash
# Using batch script (local):
bash <skill-path>/scripts/run_batch.sh \
  --config batch_cases.jsonl \
  --result-dir ./bench_results/batch/batch_TIMESTAMP \
  --common-args "--base-url http://ip:port --model /path/to/weights --served-model-name MODEL_NAME --backend openai-chat --endpoint /v1/chat/completions --dataset-name random --random-input-len 1024 --random-output-len 128"
```

### Error Handling
- If a benchmark command fails (non-zero exit), capture stderr and log
- In batch mode, continue with remaining cases; mark failed cases in summary
- If service becomes unreachable mid-benchmark, probe health before retrying
- Set reasonable timeout for each benchmark run

### Remote Execution
When running in a remote/container environment (determined in Phase 0):
- Construct commands directly in the remote shell
- Set `--result-dir` to a writable path in the remote environment (e.g., `/tmp/bench_results/`)
- After completion, read results via `cat` / `jq` in the remote session
- If user needs files locally, transfer via `scp` / `docker cp`

---

## 12. Phase 8 — Auto-Optimize

Auto-optimization finds the maximum throughput or concurrency that satisfies user-defined SLO constraints.

### Prerequisites
The user MUST provide at least one SLO constraint. If none given, ask:
- "What is your TTFT P99 target? (e.g., 500ms)"
- "What is your TPOT P99 target? (e.g., 50ms)"
- "What is your minimum acceptable success rate? (e.g., 95%)"

Do NOT define "optimal" on behalf of the user without explicit constraints.

### Search Modes

| Mode | Search Variable | Fixed Variable | Default? |
|------|----------------|---------------|----------|
| A | `--max-concurrency` | (no `--request-rate`) | **Yes** |
| B | `--request-rate` | (no `--max-concurrency`) | |
| C | `--request-rate` | `--max-concurrency` (user-specified) | |
| D | `--max-concurrency` | `--request-rate` (user-specified) | |
| E | Both (two-stage) | — | |

Confirm the search mode with the user. Default to Mode A if not specified.

### num_prompts Multiplier

The number of requests per iteration should be a multiple of the search variable to ensure statistical significance:

| Phase | Multiplier | Purpose |
|-------|-----------|---------|
| Coarse probe (Phase 2) | 2~4× | Speed priority |
| Fine search (Phase 3) | 5~8× | Accuracy priority |
| Validation (Phase 4) | 8~10× | Confidence priority |

Confirm multiplier preference with user before starting. Use defaults if no preference.

### Algorithm (4 Phases)

See `references/optimization-strategy.md` for the complete algorithm. Summary:

1. **Warmup**: Run at minimal load (concurrency=1, 50 prompts, num_warmups=5). If SLO already fails → abort with "service cannot meet SLO at minimum load".

2. **Exponential Probe**: Double the search variable each iteration (1→2→4→8→16...) until SLO is violated. This finds the upper bound.

3. **Binary Search**: Search between last-good and first-bad values. Converge when `(upper - lower) / lower < 5%` or max 8 iterations.

4. **Validation**: Run at the optimal point with higher `num_prompts` to confirm SLO compliance. If validation fails, reduce by 10% and retry (max 3 retries).

### SLO Compliance Check (after each iteration)
```
ttft_p99 <= slo_ttft_p99  (if specified)
tpot_p99 <= slo_tpot_p99  (if specified)
e2e_p99  <= slo_e2e_p99   (if specified)
success_rate >= slo_success_rate  (default 95%)
```

### Edge Cases
- **Service saturated at concurrency=1**: Report "cannot meet SLO at minimum load"
- **Service never saturates** (probe reaches very high values): Report maximum tested point
- **Unstable metrics** (>20% variance between runs): Increase `num_prompts` multiplier

### Result Archival for Auto-Optimize
Every iteration is saved individually:
```
bench_results/optimize/opt_TIMESTAMP/
├── warmup.json
├── probe_001_c1.json
├── probe_002_c2.json
├── probe_003_c4.json
├── ...
├── search_001_c12.json
├── search_002_c10.json
├── ...
├── validation.json
└── optimization_report.json   # Final summary
```

---

## 13. Results & Output

### Default Output Format: Comparison Table + Recommendation

For every benchmark run (single, batch, or auto-optimize), present results as:

**Comparison Table** (Markdown):
```
| Case | Concurrency | Rate | Prompts | Req/s | Out tok/s | TTFT mean | TTFT P99 | TPOT mean | TPOT P99 | E2E P99 | Success% | SLO? |
|------|------------|------|---------|-------|-----------|-----------|----------|-----------|----------|---------|----------|------|
| ...  | ...        | ...  | ...     | ...   | ...       | ...       | ...      | ...       | ...      | ...     | ...      | ...  |
```

**Recommendation** (after the table):
- For single: summary of key metrics, any concerns
- For batch: which configuration performed best and why
- For auto-optimize: the optimal point, SLO compliance evidence, search history summary

### Additional Output (on request)
- Raw `vllm bench serve` commands used
- Paths to result JSON files
- Short test report (using `assets/report_template.md`)
- Per-request detailed data analysis

### Aggregation Script (local execution)
```bash
python3 <skill-path>/scripts/aggregate_results.py \
  --result-dir ./bench_results/batch/batch_TIMESTAMP \
  --format markdown
```

---

## 14. Scripts & References Index

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/check_bench_env.sh` | Verify vllm installation and bench serve availability |
| `scripts/probe_service.sh` | Probe running service: health, models, backends |
| `scripts/validate_params.py` | Validate parameter combinations before execution |
| `scripts/generate_bench_cmd.py` | Generate complete benchmark command with archival flags |
| `scripts/run_bench.sh` | Execute single benchmark with error handling |
| `scripts/run_batch.sh` | Execute batch of benchmark cases sequentially |
| `scripts/aggregate_results.py` | Aggregate result JSONs into comparison table |
| `scripts/auto_optimize.py` | Auto-optimization driver (probe + binary search) |

### References

| Reference | Content |
|-----------|---------|
| `references/param-reference.md` | Complete `vllm bench serve` CLI parameter mapping |
| `references/dataset-guide.md` | Dataset types, parameters, backend compatibility matrix |
| `references/backend-mapping.md` | Backend → endpoint → model type mapping |
| `references/scenario-cookbook.md` | 10 ready-to-use benchmark scenario examples |
| `references/optimization-strategy.md` | Auto-optimization algorithm in full detail |
| `references/environment-checks.md` | Execution environment prerequisites and checks |
