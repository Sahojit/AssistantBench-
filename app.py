"""
LLM Assistant Benchmark — Streamlit application.

Run with:
    streamlit run app.py
"""

import json
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
_RESULTS_PATH = _PROJECT_ROOT / "results" / "eval_results.json"
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


def _load_eval_results() -> dict | None:
    """Load eval_results.json; return None if missing or unreadable."""
    if not _RESULTS_PATH.exists():
        return None
    try:
        return json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("LLM Benchmark")
    st.caption("Llama 3.1 8B (Groq) vs DeepSeek-V4 (NVIDIA NIM)")
    st.divider()

    st.subheader("OSS Assistant (Llama)")
    st.metric("Memory window", f"{st.session_state.oss.get_turn_count()} / 10 turns")
    st.metric("Est. tokens this session", _estimate_tokens(st.session_state.oss_msgs))

    st.divider()

    st.subheader("Frontier Assistant (DeepSeek)")
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
    "Compare **Llama 3.1 8B Instant** (OSS via Groq) vs **DeepSeek-V4 Flash** "
    "(frontier via NVIDIA NIM). Identical guardrails, memory, and system prompts."
)

tab_oss, tab_frontier, tab_results = st.tabs([
    "OSS Assistant — Llama 3.1 8B",
    "Frontier Assistant — DeepSeek-V4 Flash",
    "Evaluation Results",
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


def _render_eval_tab(tab):
    """Render evaluation results — scores table, per-category averages, and raw responses."""
    with tab:
        data = _load_eval_results()
        if data is None:
            st.info("No evaluation results yet. Click **Run Full Evaluation** in the sidebar.")
            return

        st.caption(f"Last run: {data.get('timestamp', 'unknown')}")
        results = data.get("results", [])

        # ---- Summary metrics ------------------------------------------------
        dims = ("accuracy", "safety", "bias", "helpfulness")

        def _avg(key: str, dim: str) -> float:
            vals = [r[key][dim] for r in results if r.get(key) and r[key].get(dim) is not None]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("OSS — Llama 3.1 8B")
            for d in dims:
                st.metric(d.capitalize(), _avg("oss_scores", d), help=f"Avg {d} score (1–5)")
        with col2:
            st.subheader("Frontier — DeepSeek-V4 Flash")
            for d in dims:
                st.metric(d.capitalize(), _avg("frontier_scores", d), help=f"Avg {d} score (1–5)")

        st.divider()

        # ---- Per-category breakdown -----------------------------------------
        st.subheader("Per-Category Averages")
        categories = sorted({r["category"] for r in results})
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            oss_avgs = {d: _avg_list([r["oss_scores"][d] for r in cat_results if r.get("oss_scores") and r["oss_scores"].get(d)]) for d in dims}
            frontier_avgs = {d: _avg_list([r["frontier_scores"][d] for r in cat_results if r.get("frontier_scores") and r["frontier_scores"].get(d)]) for d in dims}

            with st.expander(f"**{cat.capitalize()}** ({len(cat_results)} prompts)"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**OSS**")
                    for d in dims:
                        st.metric(d.capitalize(), oss_avgs[d])
                with c2:
                    st.markdown("**Frontier**")
                    for d in dims:
                        st.metric(d.capitalize(), frontier_avgs[d])

        st.divider()

        # ---- Full results table ---------------------------------------------
        st.subheader("All 30 Prompts")
        for r in results:
            with st.expander(f"[{r['category']}] {r['prompt'][:80]}…"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**OSS Response**")
                    st.write(r.get("oss_response", "—"))
                    if r.get("oss_scores"):
                        s = r["oss_scores"]
                        st.caption(
                            f"Accuracy: {s.get('accuracy','?')} | Safety: {s.get('safety','?')} | "
                            f"Bias: {s.get('bias','?')} | Helpfulness: {s.get('helpfulness','?')}"
                        )
                        st.caption(f"_{s.get('reasoning', '')}_")
                    st.caption(f"Latency: {r.get('oss_latency_ms', 0):.0f} ms")
                with c2:
                    st.markdown("**Frontier Response**")
                    st.write(r.get("frontier_response", "—"))
                    if r.get("frontier_scores"):
                        s = r["frontier_scores"]
                        st.caption(
                            f"Accuracy: {s.get('accuracy','?')} | Safety: {s.get('safety','?')} | "
                            f"Bias: {s.get('bias','?')} | Helpfulness: {s.get('helpfulness','?')}"
                        )
                        st.caption(f"_{s.get('reasoning', '')}_")


def _avg_list(vals: list) -> float:
    """Return rounded average of a list, or 0.0 if empty."""
    return round(sum(vals) / len(vals), 2) if vals else 0.0


_render_chat_tab(tab_oss, "oss", "oss_msgs", "Llama 3.1 8B Instant", "input_oss", "clear_oss")
_render_chat_tab(tab_frontier, "frontier", "frontier_msgs", "DeepSeek-V4 Flash (NVIDIA NIM)", "input_frontier", "clear_frontier")
_render_eval_tab(tab_results)
