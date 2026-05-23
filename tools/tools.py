"""
Tool implementations: Calculator and DateTime.

Each tool exposes a run_*(user_input) function that returns
(triggered: bool, result_string: str).
"""

import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

# Characters allowed in a safe math expression
_SAFE_CHARS_RE = re.compile(r"^[\d\s+\-*/%.()]+$")

# Trigger words / phrases
_CALC_WORDS_RE = re.compile(
    r"\b(calculate|compute|evaluate|solve)\b|what\s+is\s+\d",
    re.IGNORECASE,
)

# Extract a candidate expression from a longer sentence
_EXPR_EXTRACT_RE = re.compile(r"[\d][\d\s+\-*/%.()]*")


def run_calculator(user_input: str) -> tuple[bool, str]:
    """
    Attempt to evaluate a mathematical expression from user input.

    Triggers when the input contains calculator keywords or is itself a
    math expression.  Only characters in [0-9 + - * / % . ( ) space] are
    allowed through to eval(); anything else is rejected before evaluation.

    Args:
        user_input: The raw user message.

    Returns:
        (triggered, result_string) — triggered is False when no math was found.
    """
    stripped = user_input.strip()

    # Trigger 1: entire input is a bare math expression
    whole_is_expr = bool(_SAFE_CHARS_RE.match(stripped)) and any(
        c.isdigit() for c in stripped
    )

    # Trigger 2: contains explicit calculator keyword
    has_keyword = bool(_CALC_WORDS_RE.search(user_input))

    if not (whole_is_expr or has_keyword):
        return False, ""

    # Extract the expression to evaluate
    if whole_is_expr:
        expr = stripped
    else:
        match = _EXPR_EXTRACT_RE.search(user_input)
        if not match:
            return False, ""
        expr = match.group().strip()

    # Final safety gate — reject if any unsafe character slipped through
    if not _SAFE_CHARS_RE.match(expr):
        return False, ""

    try:
        result = eval(expr, {"__builtins__": {}}, {})  # nosec — whitelist validated above
        return True, f"Calculator result: {expr} = {result}"
    except Exception:
        return False, ""


# ---------------------------------------------------------------------------
# DateTime
# ---------------------------------------------------------------------------

_DATETIME_TRIGGERS = [
    "what time",
    "what's the time",
    "whats the time",
    "current time",
    "today's date",
    "todays date",
    "what day",
    "what date",
    "current date",
]


def run_datetime(user_input: str) -> tuple[bool, str]:
    """
    Check whether the user is asking about the current time or date.

    Args:
        user_input: The raw user message.

    Returns:
        (triggered, result_string) — triggered is False when no datetime query detected.
    """
    lowered = user_input.lower()
    for trigger in _DATETIME_TRIGGERS:
        if trigger in lowered:
            now_str = datetime.now().strftime("%A, %B %d %Y, %H:%M:%S")
            return True, f"Current datetime: {now_str}"
    return False, ""
