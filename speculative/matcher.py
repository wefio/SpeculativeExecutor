"""AST-style matching for tool calls.

Compares predicted vs actual tool calls with:
- Exact function name match
- Subset argument match (predicted params must all exist in actual)
- Flexible value comparison (10 == 10.0, "hello" == "hello")
- BFCL ground truth support (multiple valid values per param)
"""
import json


def match(predicted_name: str, predicted_args: str,
          actual_name: str, actual_args: str) -> bool:
    """Check if predicted tool call matches actual. Fast path first."""
    if predicted_name != actual_name:
        return False
    if predicted_args == actual_args:
        return True
    try:
        pred = json.loads(predicted_args)
        actual = json.loads(actual_args)
    except (json.JSONDecodeError, TypeError):
        return False
    return _args_match(pred, actual)


def match_ground_truth(predicted_name: str, predicted_args: str,
                       ground_truth: list[dict]) -> bool:
    """Check if predicted tool call matches any entry in BFCL ground truth.

    ground_truth format: [{"func_name": {"param": [val1, val2, ...]}}]
    """
    try:
        pred = json.loads(predicted_args)
    except (json.JSONDecodeError, TypeError):
        return False

    for entry in ground_truth:
        for func_name, params in entry.items():
            if predicted_name != func_name:
                continue
            if all(
                k in pred and _value_in(pred[k], v)
                for k, v in params.items()
            ):
                return True
    return False


def _args_match(pred: dict, actual: dict) -> bool:
    """Subset match: all predicted params must exist in actual with same value."""
    for k, v in pred.items():
        if k not in actual:
            return False
        av = actual[k]
        if v == av:
            continue
        if str(v) == str(av):
            continue
        if isinstance(v, (int, float)) and isinstance(av, (int, float)) and v == av:
            continue
        return False
    return True


def _value_in(value, valid_values: list) -> bool:
    """Check if value matches any in a list of valid values (BFCL format)."""
    for valid in valid_values:
        if value == valid:
            return True
        if str(value) == str(valid):
            return True
        try:
            if float(value) == float(valid):
                return True
        except (ValueError, TypeError):
            pass
    return False
