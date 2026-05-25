import io
import json
import subprocess
import sys
import urllib.request
import urllib.error


def _sanitize_name(name: str) -> str:
    """Replace dots with underscores for API compatibility."""
    return name.replace(".", "_")


def _sanitize_schema(schema: dict) -> dict:
    """Convert non-standard types for API compatibility (dict→object, float→number)."""
    TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "list": "array"}
    VALID_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
    if not isinstance(schema, dict):
        return schema
    result = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            if v == "any":
                pass  # omit type field to accept any type
            else:
                result[k] = TYPE_MAP.get(v, v if v in VALID_TYPES else "string")
        elif isinstance(v, dict):
            result[k] = _sanitize_schema(v)
        elif isinstance(v, list):
            result[k] = [_sanitize_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            result[k] = v
    return result


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, callable] = {}
        self._name_map: dict[str, str] = {}  # sanitized → original

    def register(self, name: str, description: str, parameters: dict, handler):
        sanitized = _sanitize_name(name)
        self._name_map[sanitized] = name
        self._tools[sanitized] = {
            "type": "function",
            "function": {
                "name": sanitized,
                "description": description,
                "parameters": _sanitize_schema(parameters),
            },
        }
        self._handlers[sanitized] = handler

    def get_definitions(self) -> list[dict]:
        return list(self._tools.values())

    def get_original_name(self, sanitized: str) -> str:
        """Map sanitized name back to original."""
        return self._name_map.get(sanitized, sanitized)

    def get_name_map(self) -> dict[str, str]:
        """Return full sanitized → original mapping."""
        return dict(self._name_map)

    def execute(self, name: str, arguments: str) -> str:
        if name not in self._handlers:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            result = self._handlers[name](**args)
            return str(result)
        except Exception as e:
            return json.dumps({"error": str(e)})


def _python_execute(code: str) -> str:
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        exec(code, {})
        output = buffer.getvalue()
        return output if output else "(no output)"
    finally:
        sys.stdout = old_stdout


def _shell_execute(command: str) -> str:
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=30
    )
    parts = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr}")
    if result.returncode != 0:
        parts.append(f"[exit code: {result.returncode}]")
    return "\n".join(parts) if parts else "(no output)"


def _file_read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _file_write(path: str, content: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written {len(content)} chars to {path}"


def _http_request(url: str, method: str = "GET", body: str | None = None) -> str:
    headers = {"Content-Type": "application/json"}
    data = body.encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")[:4000]
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.read().decode('utf-8')[:2000]}"


def create_default_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(
        name="python_execute",
        description="Execute Python code and return stdout output.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        },
        handler=_python_execute,
    )

    reg.register(
        name="shell_execute",
        description="Execute a shell command and return output.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"}
            },
            "required": ["command"],
        },
        handler=_shell_execute,
    )

    reg.register(
        name="file_read",
        description="Read a file and return its content.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"],
        },
        handler=_file_read,
    )

    reg.register(
        name="file_write",
        description="Write content to a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
        handler=_file_write,
    )

    reg.register(
        name="http_request",
        description="Make an HTTP request and return the response body.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to request"},
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "default": "GET",
                },
                "body": {
                    "type": "string",
                    "description": "Request body (for POST)",
                },
            },
            "required": ["url"],
        },
        handler=_http_request,
    )

    return reg
