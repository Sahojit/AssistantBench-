"""
LLM Assistant Benchmark — Streamlit application.

Run with:
    streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import streamlit as st  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from assistants.frontier_assistant import FrontierAssistant  # noqa: E402
from assistants.oss_assistant import OSSAssistant  # noqa: E402
from guardrails.filters import check_input  # noqa: E402
from observability.tracer import configure_session, get_session_stats  # noqa: E402

_PROJECT_ROOT = Path(__file__).parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python3"
_PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Assistant Benchmark",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state bootstrap
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
    configure_session(st.session_state.session_id)
if "oss" not in st.session_state:
    st.session_state.oss = OSSAssistant()
if "frontier" not in st.session_state:
    st.session_state.frontier = FrontierAssistant()
if "oss_msgs" not in st.session_state:
    st.session_state.oss_msgs = []
if "frontier_msgs" not in st.session_state:
    st.session_state.frontier_msgs = []


def _estimate_tokens(msgs: list[dict]) -> int:
    """Estimate token usage from a message list using the word × 1.3 heuristic."""
    return int(sum(len(m["content"].split()) for m in msgs) * 1.3)



# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("LLM Benchmark")
    st.caption("Llama 3.1 8B vs Llama 3.3 70B (Groq)")
    st.divider()

    st.subheader("OSS Assistant (Llama)")
    st.metric("Memory window", f"{st.session_state.oss.get_turn_count()} / 10 turns")
    st.metric("Est. tokens this session", _estimate_tokens(st.session_state.oss_msgs))

    st.divider()

    st.subheader("Frontier Assistant (Llama 70B)")
    st.metric("Memory window", f"{st.session_state.frontier.get_turn_count()} / 10 turns")
    st.metric("Est. tokens this session", _estimate_tokens(st.session_state.frontier_msgs))

    st.divider()

    st.subheader("Session Stats")
    _stats = get_session_stats()
    st.metric("Total API calls", _stats["total_calls"])
    st.metric("Avg latency (ms)", _stats["avg_latency_ms"])
    st.metric("Errors logged", _stats["error_count"])
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    st.divider()

    st.subheader("Evaluation")
    st.caption("Runs all 30 prompts against both models and saves JSON results.")

    if st.button("Run Full Evaluation", use_container_width=True):
        progress_bar = st.progress(0, text="Starting evaluation…")
        status_area = st.empty()
        try:
            proc = subprocess.Popen(
                [_PYTHON, str(_PROJECT_ROOT / "evaluation" / "runner.py")],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            total_prompts = 30
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.rstrip()
                status_area.text(line)
                if line.startswith("Running prompt"):
                    parts = line.split()
                    try:
                        num = int(parts[2].split("/")[0])
                        progress_bar.progress(num / total_prompts, text=f"Prompt {num}/{total_prompts}…")
                    except (IndexError, ValueError):
                        pass
            proc.wait()
            if proc.returncode == 0:
                progress_bar.progress(1.0, text="Complete!")
                st.success("Done — see Evaluation Results tab.")
                st.rerun()
            else:
                st.error("Evaluation failed. Check terminal for details.")
        except Exception as exc:
            st.error(f"Could not start evaluation: {exc}")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
st.title("LLM Assistant Benchmark")
st.caption(
    "Compare **Llama 3.1 8B Instant** (OSS) vs **Llama 3.3 70B Versatile** "
    "(frontier) — both via Groq. Identical guardrails, memory, and system prompts."
)

tab_oss, tab_frontier = st.tabs([
    "OSS Assistant — Llama 3.1 8B",
    "Frontier Assistant — Llama 3.3 70B",
])


def _render_chat_tab(tab, assistant_key, msgs_key, model_label, input_key, clear_key):
    """Render a complete chat interface inside a Streamlit tab."""
    with tab:
        header_col, btn_col = st.columns([5, 1])
        with header_col:
            st.subheader(model_label)
        with btn_col:
            if st.button("Clear", key=clear_key, help="Reset conversation history"):
                st.session_state[msgs_key] = []
                st.session_state[assistant_key].clear_history()
                st.rerun()

        for msg in st.session_state[msgs_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("latency_ms") is not None:
                    st.caption(f"**Model:** {model_label} &nbsp;|&nbsp; **Latency:** {msg['latency_ms']:.0f} ms")

        if user_input := st.chat_input(placeholder=f"Ask {model_label} anything…", key=input_key):
            is_safe, refusal = check_input(user_input)
            if not is_safe:
                st.warning(f"**Guardrail blocked your message:** {refusal}", icon="🛡️")
            else:
                st.session_state[msgs_key].append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        response_text, latency_ms, _ = st.session_state[assistant_key].chat(user_input)
                    st.markdown(response_text)
                    st.caption(f"**Model:** {model_label} &nbsp;|&nbsp; **Latency:** {latency_ms:.0f} ms")
                st.session_state[msgs_key].append({"role": "assistant", "content": response_text, "latency_ms": latency_ms})
                st.rerun()


_render_chat_tab(tab_oss, "oss", "oss_msgs", "Llama 3.1 8B Instant", "input_oss", "clear_oss")
_render_chat_tab(tab_frontier, "frontier", "frontier_msgs", "Llama 3.3 70B Versatile", "input_frontier", "clear_frontier")
