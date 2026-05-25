"""BFCL (Berkeley Function Calling Leaderboard) data loader and evaluation.

Provides loading, tool registration, and AST matching for BFCL v4 datasets.
Used as a library by experiment.py.

Usage:
    from benchmark_bfcl import load_bfcl, register_bfcl_tools, ast_match
"""

import json
import os
import re

from tools import ToolRegistry

DATA_DIR = os.path.join(os.path.dirname(__file__),
    "temp_gorilla", "berkeley-function-call-leaderboard", "bfcl_eval", "data")

ALL_CATEGORIES = [
    "simple_python", "simple_java", "simple_javascript",
    "multiple", "parallel", "parallel_multiple",
    "irrelevance",
]

MULTI_TURN_CATEGORIES = [
    "multi_turn_base", "multi_turn_long_context",
    "multi_turn_miss_func", "multi_turn_miss_param",
]


# ── Data Loading ─────────────────────────────────────────────────────────

def load_bfcl(category: str) -> tuple[list[dict], list[dict] | None]:
    """Load BFCL test data and ground truth (JSONL format)."""
    test_file = os.path.join(DATA_DIR, f"BFCL_v4_{category}.json")
    answer_file = os.path.join(DATA_DIR, "possible_answer", f"BFCL_v4_{category}.json")

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    ground_truth = None
    if os.path.exists(answer_file):
        with open(answer_file, "r", encoding="utf-8") as f:
            ground_truth = [json.loads(line) for line in f if line.strip()]

    return test_data, ground_truth


def load_multi_turn_func_docs() -> dict[str, list[dict]]:
    """Load all multi-turn function definitions. Returns {class_name: [func_def]}."""
    doc_dir = os.path.join(DATA_DIR, "multi_turn_func_doc")
    result = {}
    for fname in os.listdir(doc_dir):
        if not fname.endswith(".json"):
            continue
        class_name = fname.replace(".json", "")
        with open(os.path.join(doc_dir, fname), "r", encoding="utf-8") as f:
            result[class_name] = [json.loads(line) for line in f if line.strip()]
    return result


# ── Tool Registration ────────────────────────────────────────────────────

def register_bfcl_tools(registry: ToolRegistry, functions: list[dict]):
    """Register BFCL function definitions as noop tools."""
    for func in functions:
        registry.register(
            name=func["name"],
            description=func.get("description", ""),
            parameters=func.get("parameters", {"type": "object", "properties": {}}),
            handler=lambda **kwargs: "OK",
        )


# ── AST Matching ─────────────────────────────────────────────────────────

def normalize_value(val):
    """Normalize a value for comparison."""
    if isinstance(val, str):
        return val.strip()
    return val


def value_matches(predicted, valid_values: list) -> bool:
    """Check if predicted value matches any valid value in the list."""
    pred = normalize_value(predicted)
    for valid in valid_values:
        if pred == normalize_value(valid):
            return True
        if str(pred) == str(normalize_value(valid)):
            return True
    return False


def ast_match(predicted_calls: list[dict], ground_truth: list[dict]) -> bool:
    """Match predicted tool calls against BFCL ground truth.

    predicted_calls: [{"name": "func", "arguments": {"param": val}}]
    ground_truth: [{"func_name": {"param": [valid_val1, valid_val2]}}]

    Returns True if all predictions match (set matching, order-independent).
    """
    if len(predicted_calls) != len(ground_truth):
        return False
    if len(predicted_calls) == 0:
        return True

    gt_list = []
    for gt_entry in ground_truth:
        for func_name, params in gt_entry.items():
            gt_list.append((func_name, params))

    used = set()
    for pred in predicted_calls:
        pred_name = pred.get("name", "")
        pred_args = pred.get("arguments", {})
        if isinstance(pred_args, str):
            try:
                pred_args = json.loads(pred_args)
            except json.JSONDecodeError:
                pred_args = {}

        matched = False
        for i, (gt_name, gt_params) in enumerate(gt_list):
            if i in used:
                continue
            if pred_name != gt_name:
                continue
            all_params_ok = True
            for param_name, valid_values in gt_params.items():
                if param_name in pred_args:
                    if not value_matches(pred_args[param_name], valid_values):
                        all_params_ok = False
                        break
                else:
                    if "" not in valid_values and None not in valid_values:
                        all_params_ok = False
                        break
            if all_params_ok:
                used.add(i)
                matched = True
                break

        if not matched:
            return False

    return True


# ── Multi-Turn Helpers ───────────────────────────────────────────────────

def parse_python_call(call_str: str) -> dict:
    """Parse Python call syntax: cd(folder='document') -> {"name": "cd", "arguments": {"folder": "document"}}"""
    import ast
    try:
        tree = ast.parse(call_str.strip(), mode="eval")
        call = tree.body
        name = call.func.id if isinstance(call.func, ast.Name) else call.func.attr
        args = {}
        for kw in call.keywords:
            args[kw.arg] = ast.literal_eval(kw.value)
        for i, arg in enumerate(call.args):
            args[f"arg_{i}"] = ast.literal_eval(arg)
        return {"name": name, "arguments": args}
    except Exception:
        m = re.match(r'(\w+)\((.*)\)', call_str.strip())
        if not m:
            return {"name": call_str, "arguments": {}}
        name = m.group(1)
        args = {}
        for part in re.findall(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|\d+|True|False|None)", m.group(2)):
            val = part[1]
            if val.startswith(("'", '"')):
                args[part[0]] = val[1:-1]
            elif val == "True":
                args[part[0]] = True
            elif val == "False":
                args[part[0]] = False
            else:
                try:
                    args[part[0]] = int(val)
                except Exception:
                    args[part[0]] = val
        return {"name": name, "arguments": args}


def match_func_docs(involved_classes: list[str], excluded_functions: set[str],
                    func_docs: dict[str, list[dict]]) -> list[dict]:
    """Match involved classes to function definitions. Returns matched function list."""
    all_funcs = []
    matched = set()
    for cls_name in involved_classes:
        cls_norm = cls_name.lower().replace("_", "")
        for doc_name, funcs in func_docs.items():
            doc_norm = doc_name.lower().replace("_", "")
            if cls_norm in doc_norm or doc_norm in cls_norm:
                all_funcs.extend(funcs)
                matched.add(cls_name)
                break
        if cls_name not in matched:
            for doc_name, funcs in func_docs.items():
                if any(cls_name.lower() in f.get("description", "").lower() for f in funcs):
                    all_funcs.extend(funcs)
                    matched.add(cls_name)
                    break
    return [f for f in all_funcs if f["name"] not in excluded_functions]
