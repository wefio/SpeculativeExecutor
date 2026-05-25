"""Agent strategies: Simple, ReAct, Plan-and-Solve."""

SIMPLE_PROMPT = (
    'You are an assistant with tools. Use tools when needed.\n'
    'If no tool is needed, respond with: {"tool": null}\n'
    'Available tools (call as JSON: {"tool": "name", "arguments": {...}}):'
)

REACT_PROMPT = (
    'You are an assistant that solves tasks step-by-step using the Thought/Action/Observation pattern.\n\n'
    'For each step:\n'
    '1. **Thought**: Reason about what to do next\n'
    '2. **Action**: Call a tool (or say "Final Answer:" if done)\n'
    '3. **Observation**: (provided by the system after tool execution)\n\n'
    'Repeat until you have enough information to give a Final Answer.\n\n'
    'If no tool is needed, skip directly to Final Answer.\n\n'
    'Available tools (call as JSON: {"tool": "name", "arguments": {...}}):'
)

PLAN_SOLVE_PROMPT = (
    'You are an assistant that plans before acting.\n\n'
    'When given a task:\n'
    '1. First, output a **Plan** — a numbered list of steps\n'
    '2. Then execute each step, calling tools as needed\n'
    '3. After all steps, give a **Summary**\n\n'
    'If no tool is needed, respond with: {"tool": null}\n\n'
    'Available tools (call as JSON: {"tool": "name", "arguments": {...}}):'
)


def get_system_prompt(strategy: str, tool_signatures: str) -> str:
    """Build system prompt for the given strategy with tool signatures appended."""
    prompts = {
        "simple": SIMPLE_PROMPT,
        "react": REACT_PROMPT,
        "plan_solve": PLAN_SOLVE_PROMPT,
    }
    base = prompts.get(strategy, SIMPLE_PROMPT)
    return f"{base}\n{tool_signatures}"


def get_max_iterations(strategy: str) -> int:
    """ReAct and Plan-and-Solve may need more iterations."""
    return {
        "simple": 10,
        "react": 15,
        "plan_solve": 15,
    }.get(strategy, 10)
