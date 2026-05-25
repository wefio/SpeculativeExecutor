"""BFCL experiment: Normal vs Speculative execution.

Single entry point for running BFCL benchmark experiments.
Compares main model (chat) vs speculative (prefix continuation) on:
- Accuracy (AST matching against ground truth)
- Latency (time to tool call)
- Token consumption

Supports single-turn and multi-turn BFCL v4 categories.

Usage:
    uv run experiment.py --category simple_python --max-samples 10
    uv run experiment.py --category multi_turn_base --max-samples 5
    uv run experiment.py --category all --concurrency 5
"""

import argparse
import asyncio
import json
import os
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from provider import DeepSeekProvider
from tools import ToolRegistry
from speculative import SpeculativeExecutor, SpecConfig
from benchmark_bfcl import (
    ALL_CATEGORIES, MULTI_TURN_CATEGORIES,
    load_bfcl, load_multi_turn_func_docs, match_func_docs,
    register_bfcl_tools, ast_match, parse_python_call,
)
from experiments.save import save


# ── Helpers ──────────────────────────────────────────────────────────────

def _extract_user_msg(question) -> str:
    """Extract user message from BFCL question field."""
    if isinstance(question, list) and len(question) > 0:
        first = question[0]
        if isinstance(first, list) and len(first) > 0:
            return first[0].get("content", str(first))
        elif isinstance(first, dict):
            return first.get("content", str(first))
        return str(first)
    return str(question)


def _token_dict(usage) -> dict | None:
    """Convert OpenAI usage object to dict."""
    if not usage:
        return None
    return {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens, "total": usage.total_tokens}


# ── Single-Turn Runner ───────────────────────────────────────────────────

async def run_single(provider, item: dict, speculative: bool) -> dict:
    """Run one single-turn BFCL case. Returns tool_calls, timing, tokens."""
    tools = ToolRegistry()
    register_bfcl_tools(tools, item.get("function", []))
    name_map = tools.get_name_map()
    user_msg = _extract_user_msg(item["question"])
    messages = [
        {"role": "system", "content": "You are an assistant with tools."},
        {"role": "user", "content": user_msg},
    ]

    spec_raw = []
    prefix_time = None

    if speculative:
        spec = SpeculativeExecutor(provider, tools, SpecConfig())
        try:
            await spec._run(messages, user_msg)
        except Exception:
            pass
        if spec.stats:
            prefix_time = round(sum(s.prefix_duration for s in spec.stats) / len(spec.stats), 3)
        seen = set()
        for cached in spec._caches:
            name = name_map.get(cached.tool_name, cached.tool_name)
            if name in seen:
                continue
            seen.add(name)
            args = cached.arguments
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                pass
            spec_raw.append({"name": name, "arguments": args})

    t0 = time.perf_counter()
    resp = await provider.chat(messages=messages, tools=tools.get_definitions())
    latency = time.perf_counter() - t0

    main_calls = []
    if resp.tool_calls:
        for tc in resp.tool_calls:
            name = name_map.get(tc.function.name, tc.function.name)
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = tc.function.arguments
            main_calls.append({"name": name, "arguments": args})

    return {
        "main_calls": main_calls,
        "spec_raw": spec_raw,
        "latency": round(latency, 3),
        "prefix_time": prefix_time,
        "main_tokens": _token_dict(provider.chat_usage),
        "prefix_tokens": _token_dict(provider.prefix_usage),
    }


# ── Multi-Turn Runner ────────────────────────────────────────────────────

def _build_multi_turn_tools(item: dict, func_docs: dict) -> tuple[ToolRegistry, dict]:
    """Build tool registry for multi-turn case from involved_classes."""
    involved = item.get("involved_classes", [])
    excluded = set(item.get("excluded_function", []))
    funcs = match_func_docs(involved, excluded, func_docs)

    tools = ToolRegistry()
    for f in funcs:
        tools.register(
            name=f["name"], description=f.get("description", ""),
            parameters=f.get("parameters", {"type": "object", "properties": {}}),
            handler=lambda **kw: "OK",
        )
    return tools, {f["name"]: f.get("description", "") for f in funcs}


async def run_multi(provider, item: dict, func_docs: dict, speculative: bool) -> dict:
    """Run one multi-turn BFCL case. Returns accumulated tool_calls across turns."""
    tools, tool_descs = _build_multi_turn_tools(item, func_docs)
    name_map = tools.get_name_map()
    initial_config = item.get("initial_config", {})

    spec = None
    all_spec_names = set()

    if speculative:
        spec = SpeculativeExecutor(provider, tools, SpecConfig())
        if initial_config:
            spec.context = f"Current environment state:\n{json.dumps(initial_config, indent=2)}"

    tool_desc_lines = "\n".join(f"- {n}: {d}" for n, d in tool_descs.items())
    system_prompt = (
        "You are an assistant with tools. "
        "The environment state is already provided — do NOT explore.\n"
        "Do NOT use ls, pwd, find, or other exploratory tools. "
        "Directly call the tools needed for the task.\n\n"
        f"Available tools:\n{tool_desc_lines}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    if initial_config:
        messages.append({
            "role": "assistant",
            "content": f"Current environment state:\n{json.dumps(initial_config, indent=2)}",
        })

    all_calls = []
    total_latency = 0.0
    prefix_time = None

    for turn_msgs in item["question"]:
        user_msg = _extract_user_msg(turn_msgs)
        messages.append({"role": "user", "content": user_msg})

        if spec:
            try:
                await spec._run(messages, user_msg)
                if not prefix_time and spec.stats:
                    prefix_time = round(sum(s.prefix_duration for s in spec.stats) / len(spec.stats), 3)
                for cached in spec._caches:
                    all_spec_names.add(name_map.get(cached.tool_name, cached.tool_name))
            except Exception:
                pass

        t0 = time.perf_counter()
        resp = await provider.chat(messages=messages, tools=tools.get_definitions())
        elapsed = time.perf_counter() - t0
        total_latency += elapsed

        turn_calls = []
        if resp.tool_calls:
            for tc in resp.tool_calls:
                name = name_map.get(tc.function.name, tc.function.name)
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = tc.function.arguments
                turn_calls.append({"name": name, "arguments": args})
            all_calls.extend(turn_calls)
            calls_text = ", ".join(
                f'{c["name"]}({json.dumps(c["arguments"])})' for c in turn_calls
            )
            messages.append({"role": "assistant", "content": f"Called: {calls_text}"})
            messages.append({
                "role": "user",
                "content": f"Tool results: OK. Current state updated after {calls_text}",
            })
        else:
            messages.append({"role": "assistant", "content": resp.content or ""})

        if spec:
            spec.clear()

    if spec and spec.stats:
        prefix_time = round(sum(s.prefix_duration for s in spec.stats) / len(spec.stats), 3)

    return {
        "main_calls": all_calls,
        "spec_names": sorted(all_spec_names),
        "latency": round(total_latency, 3),
        "prefix_time": prefix_time,
        "main_tokens": _token_dict(provider.chat_usage),
        "prefix_tokens": _token_dict(provider.prefix_usage),
    }


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_single(result: dict, gt, is_irrelevance: bool) -> tuple[bool | None, bool | None]:
    """Evaluate single-turn: (main_correct, spec_correct)."""
    if "error" in result:
        return None, None
    main = result.get("main_calls", [])
    spec = result.get("spec_raw", [])
    if is_irrelevance:
        return len(main) == 0, len(spec) == 0
    if gt is None:
        return None, None
    return ast_match(main, gt), ast_match(spec, gt) if spec else False


def evaluate_multi(result: dict, gt) -> tuple[bool | None, bool | None]:
    """Evaluate multi-turn: subset match for main, name-only match for spec."""
    if "error" in result:
        return None, None

    main_names = set(c["name"] for c in result.get("main_calls", []))
    main_ok = all(
        set(parse_python_call(c)["name"] for c in turn).issubset(main_names)
        for turn in gt
    )

    spec_names = set(result.get("spec_names", []))
    spec_ok = any(
        set(parse_python_call(c)["name"] for c in turn) & spec_names
        for turn in gt
    ) if spec_names else None

    return main_ok, spec_ok


# ── Report ───────────────────────────────────────────────────────────────

def print_header(category: str, n: int, concurrency: int):
    print(f"\n{'━' * 60}")
    print(f"  {category} ({n} samples, concurrency={concurrency})")
    print(f"{'━' * 60}")


def print_summary(category: str, records: list[dict], failures: list[dict]):
    n = len(records) + len(failures)
    ok = [r for r in records if r.get("status") == "ok"]

    n_failed = len(failures)
    n_main_ok = sum(1 for r in ok if r.get("main_correct"))
    n_main_err = sum(1 for r in ok if r.get("main_correct") is False)
    n_spec_ok = sum(1 for r in ok if r.get("spec_correct") is True)
    n_spec_err = sum(1 for r in ok if r.get("spec_correct") is False)

    main_lat = [r["main_latency"] for r in ok if r.get("main_latency")]
    spec_lat = [r["prefix_latency"] for r in ok if r.get("prefix_latency")]
    avg_main = sum(main_lat) / len(main_lat) if main_lat else 0
    avg_spec = sum(spec_lat) / len(spec_lat) if spec_lat else 0

    main_tok = [r["main_tokens"]["total"] for r in ok if r.get("main_tokens")]
    spec_tok = [r["prefix_tokens"]["total"] for r in ok if r.get("prefix_tokens")]
    total_main_tok = sum(main_tok)
    total_spec_tok = sum(spec_tok)

    print(f"\n{'─' * 60}")
    print(f"  Results: {category}")
    print(f"{'─' * 60}")
    print(f"  Total samples:       {n}")
    print(f"  API failures (客观):  {n_failed}")
    print(f"  Completed:           {len(ok)}")
    print(f"")
    print(f"  {'':<25} {'Count':>6}  {'Rate':>8}")
    print(f"  {'  Main correct':<25} {n_main_ok:>6}  {n_main_ok/len(ok)*100:>7.1f}%")
    print(f"  {'  Main wrong (模型)':<25} {n_main_err:>6}  {n_main_err/len(ok)*100:>7.1f}%")
    print(f"  {'  Spec correct':<25} {n_spec_ok:>6}  {n_spec_ok/len(ok)*100:>7.1f}%")
    print(f"  {'  Spec wrong (参数)':<25} {n_spec_err:>6}  {n_spec_err/len(ok)*100:>7.1f}%")
    print(f"")
    print(f"  {'':<25} {'Main':>10}  {'Spec':>10}  {'Delta':>10}")
    print(f"  {'  Avg Latency':<25} {avg_main:>9.2f}s  {avg_spec:>9.2f}s  {avg_main/avg_spec:>9.1f}x")
    print(f"  {'  Tokens (total)':<25} {total_main_tok:>9,}  {total_spec_tok:>9,}  {total_spec_tok/total_main_tok*100:>9.0f}%")
    min_m = min(main_lat) if main_lat else 0
    max_m = max(main_lat) if main_lat else 0
    min_s = min(spec_lat) if spec_lat else 0
    max_s = max(spec_lat) if spec_lat else 0
    print(f"  {'  Latency range':<25} {min_m:.1f}-{max_m:.1f}s  {min_s:.1f}-{max_s:.1f}s")
    print(f"{'─' * 60}")


def print_analysis(category: str, records: list[dict], gt_data: list[dict], is_multi: bool):
    """Analyze tool distribution: expected vs actual, failed cases breakdown."""
    from collections import Counter

    ok = [r for r in records if r.get("status") == "ok"]
    wrong_main = [r for r in ok if r.get("main_correct") is False]
    wrong_spec = [r for r in ok if r.get("spec_correct") is False]

    if not wrong_main and not wrong_spec:
        return None

    expected_counter = Counter()      # per-case: which tools expected
    main_counter = Counter()          # all main calls
    main_wrong_counter = Counter()    # main calls in failed cases
    unexpected_counter = Counter()    # calls NOT in that case's GT
    missed_counter = Counter()        # expected tools model didn't call
    spec_counter = Counter()
    spec_unexpected = Counter()

    # Build GT lookup
    gt_map = {}
    if gt_data:
        for g in gt_data:
            gt_map[g["id"]] = g.get("ground_truth", [])

    for r in ok:
        rid = r["id"]
        gt = gt_map.get(rid, [])

        # Extract expected names for this case
        if is_multi:
            expected = set()
            for turn in gt:
                for call in turn:
                    expected.add(call.split("(")[0])
        else:
            expected = set()
            for entry in gt:
                expected.update(entry.keys())

        for en in expected:
            expected_counter[en] += 1

        # Main model calls
        main_names = []
        for c in r.get("main_calls", []):
            name = c["name"] if isinstance(c, dict) else c
            main_names.append(name)
            main_counter[name] += 1

        is_wrong = r.get("main_correct") is False
        if is_wrong:
            for n in main_names:
                main_wrong_counter[n] += 1
            for n in main_names:
                if n not in expected:
                    unexpected_counter[n] += 1
            for en in expected:
                if en not in set(main_names):
                    missed_counter[en] += 1

        # Spec predictions
        spec_data = r.get("spec_calls") or r.get("spec_names") or []
        for item in spec_data:
            name = item["name"] if isinstance(item, dict) else item
            spec_counter[name] += 1
            if name not in expected:
                spec_unexpected[name] += 1

    n_wrong = len(wrong_main)
    n_wrong_spec = len(wrong_spec)

    print(f"\n{'─' * 60}")
    print(f"  Failure Analysis: {category}")
    print(f"{'─' * 60}")

    if n_wrong > 0:
        print(f"\n  Main failures: {n_wrong} cases")
        if unexpected_counter:
            print(f"  Unexpected tools called (not in GT):")
            for name, count in unexpected_counter.most_common(8):
                print(f"    {name}: {count}×")
        if missed_counter:
            print(f"  Expected tools missed:")
            for name, count in missed_counter.most_common(8):
                print(f"    {name}: {count}×")
        if main_wrong_counter and unexpected_counter:
            ratio = sum(unexpected_counter.values()) / sum(main_wrong_counter.values()) * 100
            print(f"  Unexpected call ratio: {sum(unexpected_counter.values())}/{sum(main_wrong_counter.values())} ({ratio:.0f}%)")

    if n_wrong_spec > 0:
        print(f"\n  Spec failures: {n_wrong_spec} cases")
        if spec_unexpected:
            print(f"  Spec predicted tools not in GT:")
            for name, count in spec_unexpected.most_common(8):
                print(f"    {name}: {count}×")

    print(f"{'─' * 60}")

    return {
        "main_failures": n_wrong,
        "unexpected_calls": dict(unexpected_counter.most_common()),
        "missed_tools": dict(missed_counter.most_common()),
        "wrong_case_calls": dict(main_wrong_counter.most_common()),
        "spec_failures": n_wrong_spec,
        "spec_unexpected": dict(spec_unexpected.most_common()),
    }


# ── Main ─────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="BFCL Experiment: Normal vs Speculative")
    parser.add_argument("--category", default="simple_python",
                        choices=ALL_CATEGORIES + MULTI_TURN_CATEGORIES + ["all", "multi_turn"])
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-dir", default="experiments")
    args = parser.parse_args()

    if args.category == "all":
        categories = ALL_CATEGORIES + MULTI_TURN_CATEGORIES
    elif args.category == "multi_turn":
        categories = MULTI_TURN_CATEGORIES
    else:
        categories = [args.category]

    func_docs = None

    for category in categories:
        is_multi = category in MULTI_TURN_CATEGORIES
        is_irr = (category == "irrelevance")

        test_data, gt_data = load_bfcl(category)
        if args.max_samples > 0:
            test_data = test_data[:args.max_samples]
            if gt_data:
                gt_data = gt_data[:args.max_samples]
        n = len(test_data)

        if is_multi and func_docs is None:
            func_docs = load_multi_turn_func_docs()
            print(f"  Loaded {len(func_docs)} function doc files")

        print_header(category, n, args.concurrency)

        # Run cases in batches
        async def run_one(i, item):
            gt = gt_data[i].get("ground_truth") if gt_data and i < len(gt_data) else None
            if is_multi:
                pn, ps = DeepSeekProvider(), DeepSeekProvider()
                nr, sr = await asyncio.gather(
                    run_multi(pn, item, func_docs, speculative=False),
                    run_multi(ps, item, func_docs, speculative=True),
                )
            else:
                pn, ps = DeepSeekProvider(), DeepSeekProvider()
                nr, sr = await asyncio.gather(
                    run_single(pn, item, speculative=False),
                    run_single(ps, item, speculative=True),
                )
            return i, gt, nr, sr

        all_results = []
        for batch_start in range(0, n, args.concurrency):
            batch = test_data[batch_start:batch_start + args.concurrency]
            tasks = [run_one(batch_start + j, item) for j, item in enumerate(batch)]
            batch_results = await asyncio.gather(*tasks)
            new_items = sorted(batch_results, key=lambda x: x[0])
            all_results.extend(new_items)

            # Evaluate batch incrementally for progress display
            batch_ok = 0
            batch_main_ok = 0
            batch_spec_ok = 0
            for i, gt, nr, sr in new_items:
                n_err = nr.get("error")
                s_err = sr.get("error")
                if n_err or s_err:
                    continue
                batch_ok += 1
                if is_multi:
                    n_ok, _ = evaluate_multi(nr, gt)
                    _, s_ok = evaluate_multi(sr, gt)
                else:
                    n_ok, _ = evaluate_single(nr, gt, is_irr)
                    _, s_ok = evaluate_single(sr, gt, is_irr)
                if n_ok:
                    batch_main_ok += 1
                if s_ok:
                    batch_spec_ok += 1

            done = len(all_results)
            print(f"  {done}/{n} done | main={batch_main_ok}/{batch_ok} spec={batch_spec_ok}/{batch_ok} in batch",
                  flush=True)

        # Final evaluation and display
        records = []
        failures = []

        for i, gt, nr, sr in all_results:
            n_err = nr.get("error")
            s_err = sr.get("error")
            if n_err or s_err:
                failures.append({"id": test_data[i]["id"], "normal_error": n_err, "spec_error": s_err})
                print(f"  [{i+1}/{n}] {test_data[i]['id']}  FAILED: {n_err or s_err}")
                records.append({"id": test_data[i]["id"], "status": "failed",
                                "normal_error": n_err, "spec_error": s_err})
                continue

            if is_multi:
                n_ok, _ = evaluate_multi(nr, gt)
                _, s_ok = evaluate_multi(sr, gt)
            else:
                n_ok, _ = evaluate_single(nr, gt, is_irr)
                _, s_ok = evaluate_single(sr, gt, is_irr)

            spec_time = sr.get("prefix_time") or 0
            print(f"  [{i+1}/{n}] {test_data[i]['id']}  "
                  f"N={'✓' if n_ok else '✗'} S={'✓' if s_ok else '✗'}  "
                  f"main={nr.get('latency', 0):.1f}s "
                  f"prefix={spec_time:.1f}s")

            records.append({
                "id": test_data[i]["id"],
                "status": "ok",
                "main_latency": nr.get("latency"),
                "main_tokens": nr.get("main_tokens"),
                "main_calls": nr.get("main_calls"),
                "main_correct": n_ok,
                "prefix_latency": sr.get("prefix_time"),
                "prefix_tokens": sr.get("prefix_tokens"),
                "spec_calls": sr.get("spec_raw") if not is_multi else None,
                "spec_names": sr.get("spec_names") if is_multi else None,
                "spec_correct": s_ok,
            })

        print_summary(category, records, failures)

        # Failure analysis
        analysis = print_analysis(category, records, gt_data, is_multi)

        # Save
        save(f"bfcl_{category}", {
            "category": category,
            "samples": n,
            "concurrency": args.concurrency,
            "completed": len([r for r in records if r.get("status") == "ok"]),
            "failed": len(failures),
            "results": records,
            "failures": failures,
            "analysis": analysis,
        })


if __name__ == "__main__":
    asyncio.run(main())
