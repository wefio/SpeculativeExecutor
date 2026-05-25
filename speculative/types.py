"""Types for speculative execution."""
import json


class PredictionResult:
    """A single speculative prediction."""
    __slots__ = ("tool_name", "arguments", "arguments_normalized", "result", "level")

    def __init__(self, tool_name: str, arguments: str, result: str, level: str):
        self.tool_name = tool_name
        self.arguments = arguments
        self.result = result
        self.level = level
        self.arguments_normalized = self._normalize(arguments)

    @staticmethod
    def _normalize(args_str: str) -> str:
        return json.dumps(json.loads(args_str), sort_keys=True, ensure_ascii=False)

    def matches(self, name: str, args: str) -> bool:
        """AST-style matching: name exact, args subset match."""
        if self.tool_name != name:
            return False
        if self.arguments == args:
            return True
        try:
            pred = json.loads(self.arguments)
            actual = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return self.arguments_normalized == self._normalize(args)
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


class PredictionStats:
    """Timing and metadata for one prediction."""
    __slots__ = ("prefix_duration", "fim_duration", "total_duration",
                 "prefix_text", "fim_text", "tool_name", "args_json",
                 "level", "error")

    def __init__(self):
        self.prefix_duration = 0.0
        self.fim_duration = 0.0
        self.total_duration = 0.0
        self.prefix_text = ""
        self.fim_text = ""
        self.tool_name = ""
        self.args_json = ""
        self.level = ""
        self.error = ""

    def __repr__(self):
        if self.error:
            return f"PredictionStats(error={self.error[:50]})"
        return (f"PredictionStats(tool={self.tool_name}, "
                f"prefix={self.prefix_duration:.3f}s, total={self.total_duration:.3f}s)")
