# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AutoTools — speculative execution for agent tool calls. Uses DeepSeek's prefix continuation API to predict tool calls before the main model finishes, reducing latency and token consumption. Evaluated on BFCL v4 benchmark.

## Getting Started

```bash
uv sync                              # install dependencies
$env:DEEPSEEK_API_KEY="sk-xxx"       # set API key
uv run experiment.py --category simple_python --max-samples 10
uv run experiment.py --category multi_turn_base --max-samples 5
uv run experiment.py --category all --concurrency 10
```

## Architecture

- `provider.py` — DeepSeek API wrapper: `chat()`, `prefix_complete()`, `fim_complete()`
- `tools.py` — `ToolRegistry` with 5 built-in tools + BFCL tool registration. `_sanitize_schema()` handles non-standard JSON Schema types (tuple→array, dict→object, float→number, any→omit)
- `speculative/` — `SpeculativeExecutor`: predicts tool calls via prefix continuation, caches results. Supports Prefix and Pre+FIM levels
- `agent.py` — Agent loop with speculative execution support
- `config.py` — API key, model name, base URL
- `main.py` — Interactive REPL

## Benchmark

- `experiment.py` — **Main entry point.** Runs BFCL categories, compares Normal vs Speculative on accuracy/latency/tokens. Supports `--category`, `--max-samples`, `--concurrency`. Saves results to `experiments/`.
- `benchmark_bfcl.py` — Library: data loading, AST matching, tool registration, multi-turn helpers. Imported by `experiment.py`.

## Key Findings (BFCL v4, DeepSeek V4 Flash, 2026-05)

| Category | Main Acc | Spec Acc | Speedup | Token Ratio |
|---|---|---|---|---|
| simple_python (400) | 92.2% | 66.0% | 1.8x | 27% |
| multi_turn_base (200) | 2.5% | 91.0% | 11.1x | 36% |
| irrelevance (240) | 70.0% | 44.6% | 3.9x | 18% |

**Single-turn**: Main wins on parameter precision. Spec loses arguments from missing parameter descriptions.

**Multi-turn**: Main collapses into exploration loops (pwd/ls/get_status dominate 37% of calls). Spec bypasses the "verify-then-act" habit — predicts task-relevant tools directly. 11x speedup.

**Irrelevance**: Hardest for Spec — false positive rate 55% (predicts tools when none needed). Main also struggles (30% false positive).

## Design Notes

- **Prefix continuation** — optimal speculative approach. FIM cannot standalone, only works as Stage2.
- **`{"tool": null}` protocol** — models output null when no tool needed (irrelevance detection)
- **Schema sanitization** — BFCL uses non-standard types (tuple, dict, float, any) that must be mapped to valid JSON Schema before sending to API
- **Multi-turn tool selection** — main model's failure mode is not "can't use tools" but "prefers to verify state before acting" when many tools are available
