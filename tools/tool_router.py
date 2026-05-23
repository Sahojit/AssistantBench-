"""
Tool router — runs all tools against user input and augments the message when
a tool fires.  Called at the start of every assistant chat() method before
guardrails so the model receives computed results alongside the question.
"""

from tools.tools import run_calculator, run_datetime

_TOOLS = [run_calculator, run_datetime]


def check_tools(user_input: str) -> str:
    """
    Run all registered tools against user input in priority order.

    If a tool fires, the returned string prepends the tool result so the model
    can use it to ground its answer:

        [Tool Result]: Calculator result: 12 * 8 = 96

        User question: What is 12 times 8?

    If no tool fires, the original input is returned unchanged.

    Args:
        user_input: The raw user message before guardrail filtering.

    Returns:
        Augmented message string (or original if no tool triggered).
    """
    for tool_fn in _TOOLS:
        triggered, result = tool_fn(user_input)
        if triggered:
            return f"[Tool Result]: {result}\n\nUser question: {user_input}"
    return user_input
