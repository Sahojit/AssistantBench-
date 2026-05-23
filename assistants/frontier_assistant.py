"""Frontier assistant backed by DeepSeek-V4-Flash via NVIDIA NIM (OpenAI-compatible API)."""

import logging
import os
import time

from openai import OpenAI  # type: ignore

from guardrails.filters import check_input, check_output_toxicity, get_system_prompt, scan_output
from observability.tracer import trace_call
from tools.tool_router import check_tools

logger = logging.getLogger(__name__)

MODEL_ID = "deepseek-ai/deepseek-v4-flash"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_TURNS = 10


class FrontierAssistant:
    """
    Wraps DeepSeek-V4-Flash via NVIDIA NIM using the OpenAI-compatible API.

    Pipeline for chat():
        check_tools → check_input (guardrail) → model → scan_output → check_output_toxicity

    Pipeline for generate_for_eval():
        model → scan_output  (guardrails bypassed so adversarial prompts reach the model)
    """

    def __init__(self) -> None:
        """Initialise the NVIDIA NIM client and reset conversation state."""
        self.client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=os.getenv("NVIDIA_API_KEY"),
        )
        self.system_prompt = get_system_prompt()
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Public interface used by the Streamlit app
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> tuple[str, float, bool]:
        """
        Process a user message through the full pipeline.

        Order: tool augmentation → input guardrail → model → output scan → toxicity check.

        Returns:
            response_text: The assistant's reply (or a refusal/redacted message).
            latency_ms:    Wall-clock round-trip time in milliseconds.
            was_blocked:   True when the input guardrail blocked the request.
        """
        augmented = check_tools(user_message)

        is_safe, refusal = check_input(augmented)
        if not is_safe:
            return refusal, 0.0, True

        messages = self._build_messages(augmented)
        response_text, latency_ms = self._call_api(messages)

        response_text = scan_output(response_text)
        response_text, _ = check_output_toxicity(response_text, user_message)

        self._append_turn(user_message, response_text)
        trace_call(
            model_name=MODEL_ID,
            prompt=user_message,
            response=response_text,
            latency_ms=latency_ms,
            metadata={"turn_count": self.get_turn_count()},
        )
        return response_text, latency_ms, False

    def generate_for_eval(self, prompt: str) -> tuple[str, float]:
        """
        Send a single prompt directly to the model without input guardrails.

        Used by the evaluation runner so adversarial prompts reach the raw
        model rather than being blocked by the keyword filter. Layer 3
        output scanning is still applied; Layer 4 toxicity is skipped so
        judge scores reflect the model's actual output.

        Returns:
            response_text: The model's reply after output scanning.
            latency_ms:    Wall-clock round-trip time in milliseconds.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response_text, latency_ms = self._call_api(messages)
        response_text = scan_output(response_text)
        trace_call(
            model_name=MODEL_ID,
            prompt=prompt,
            response=response_text,
            latency_ms=latency_ms,
            metadata={"mode": "eval"},
        )
        return response_text, latency_ms

    def clear_history(self) -> None:
        """Reset the sliding conversation window."""
        self.history = []

    def get_turn_count(self) -> int:
        """Return the number of complete turns currently stored in memory."""
        return len(self.history) // 2

    def estimate_tokens(self) -> int:
        """Rough token estimate for the stored history (words × 1.3)."""
        total_words = sum(len(m["content"].split()) for m in self.history)
        return int(total_words * 1.3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, user_message: str) -> list[dict]:
        """Construct the messages list with sliding-window history."""
        window = self.history[-(MAX_TURNS * 2):]
        return (
            [{"role": "system", "content": self.system_prompt}]
            + window
            + [{"role": "user", "content": user_message}]
        )

    def _call_api(self, messages: list[dict]) -> tuple[str, float]:
        """Call the NVIDIA NIM API and return (response_text, latency_ms)."""
        t0 = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=512,
                temperature=0.7,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            return completion.choices[0].message.content.strip(), latency_ms
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error("FrontierAssistant (NVIDIA NIM) API error: %s", exc)
            return "I encountered an error processing your request. Please try again.", latency_ms

    def _append_turn(self, user_text: str, assistant_text: str) -> None:
        """Append a turn to history and trim to the sliding window."""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        if len(self.history) > MAX_TURNS * 2:
            self.history = self.history[-(MAX_TURNS * 2):]
