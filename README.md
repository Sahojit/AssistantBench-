# LLM Assistant Benchmark

A side-by-side comparison of two AI assistants — an open-source model (Llama 3.1 8B via Groq) and a frontier model (Llama 3.3 70B via Groq) — with shared guardrails, memory, tool use, observability, a 30-prompt evaluation suite judged by an LLM, and automated PDF report generation.

**Live demo:** https://llm-assistant-benchmark.onrender.com

---

## Architecture Decisions

| Component | Choice | Reason |
|---|---|---|
| **OSS model** | Llama 3.1 8B Instant | Lightweight open-source model; free via Groq's inference API; low latency |
| **Frontier model** | Llama 3.3 70B Versatile | 70B vs 8B creates a meaningful quality gap to benchmark; same provider keeps latency comparable |
| **UI framework** | Streamlit | Minimal boilerplate for chat interfaces; built-in chat primitives |
| **Guardrails** | 4-layer custom pipeline | Layer 1: keyword/PII input filter → Layer 2: system prompt injection → Layer 3: output PII redaction → Layer 4: LLM toxicity check |
| **Tool use** | Calculator + datetime | Augments user messages before guardrail check; never blocks |
| **Observability** | LangFuse v4 | Open-source LLM tracing; session-level stats surfaced in the sidebar |
| **LLM judge** | Llama 3.3 70B via Groq | Scores responses on accuracy, safety, bias, helpfulness (1–5); structured JSON output |
| **PDF generation** | ReportLab + Matplotlib | Pure-Python; bar charts per dimension + summary table |

---

## Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/Sahojit/OLLive-LLm-assistantbenchmark.git
   cd OLLive-LLm-assistantbenchmark
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   # .venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and fill in:

   | Variable | Where to get it |
   |---|---|
   | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys |
   | `NVIDIA_API_KEY` | [build.nvidia.com](https://build.nvidia.com) → Get API Key *(optional — not used in current config)* |
   | `LANGFUSE_PUBLIC_KEY` | [cloud.langfuse.com](https://cloud.langfuse.com) *(optional)* |
   | `LANGFUSE_SECRET_KEY` | Same *(optional)* |
   | `LANGFUSE_HOST` | `https://us.cloud.langfuse.com` |

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## How to Run

### Chat app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Three tabs:
- **OSS Assistant** — chat with Llama 3.1 8B
- **Frontier Assistant** — chat with Llama 3.3 70B
- **Evaluation Results** — view scores from the last eval run

### Full evaluation (30 prompts)
```bash
python evaluation/runner.py
```
Runs all 30 prompts (factual / bias / adversarial) against both models, scores each with the LLM judge, and writes `results/eval_results.json`. Expect ~5 minutes.

You can also trigger this from the **Run Full Evaluation** button in the sidebar.

### PDF report
```bash
python report/generate_report.py
```
Reads `results/eval_results.json` and produces bar charts + a full PDF at `report/evaluation_report.pdf`.

---

## Project Structure

```
llm-benchmark/
├── app.py                        # Streamlit app — chat UI + eval results tab
├── assistants/
│   ├── oss_assistant.py          # Llama 3.1 8B via Groq
│   └── frontier_assistant.py     # Llama 3.3 70B via Groq
├── evaluation/
│   ├── prompts.py                # 30 eval prompts (factual/bias/adversarial)
│   ├── judge.py                  # LLM-as-judge (Llama 3.3 70B, scores 1–5)
│   └── runner.py                 # Runs all prompts, saves results + latency report
├── guardrails/
│   └── filters.py                # 4-layer safety pipeline
├── tools/
│   ├── tools.py                  # Calculator + datetime tools
│   └── tool_router.py            # Prepends tool results to user messages
├── observability/
│   └── tracer.py                 # LangFuse v4 tracing + session stats
├── report/
│   └── generate_report.py        # Reads results, outputs PDF + charts
├── hf_space/
│   └── app.py                    # Standalone OSS-only app for HF Spaces deploy
├── results/                      # Written by runner.py (gitignored)
├── .env.example                  # Template — copy to .env and fill in keys
├── render.yaml                   # Render.com deploy config
└── requirements.txt
```

---

## Evaluation Suite

30 prompts across 3 categories (10 each):

| Category | What it tests |
|---|---|
| **Factual** | Accuracy on capital cities, Nobel prizes, scientific constants, historical dates |
| **Bias** | Gender, race, and religion stereotypes; political balance |
| **Adversarial** | Jailbreak attempts, roleplay bypasses, harmful content requests |

The judge scores each response on four dimensions (1–5):
- **Accuracy** — factual correctness
- **Safety** — correct refusal of harmful requests
- **Bias** — balanced, fair responses
- **Helpfulness** — useful and on-topic

---

## Tradeoffs Made

- **Same provider for both models.** Using Groq for both OSS (8B) and frontier (70B) keeps infrastructure simple and latency comparable, but means the "frontier" model is still open-source. A paid API like Claude or GPT-4 would give a sharper OSS vs. proprietary comparison; the NVIDIA NIM free tier (DeepSeek) was tested but hit rate limits during bulk evaluation.

- **Keyword-based guardrails, not semantic.** The input filter catches exact keyword matches but can be evaded by paraphrasing. A production system would add an embedding-based classifier or LlamaGuard as an additional layer.

- **Judge model = frontier model.** Using the same model family as a benchmark participant and as the judge could introduce implicit bias. An independent judge (different model family) would remove this conflict.

---

## What I Would Improve With More Time

- **Swap frontier model for a paid proprietary API** (Claude Sonnet or GPT-4.1) to get a true OSS vs. closed-source comparison — the most meaningful split for real-world deployment decisions.

- **Streaming responses in the Streamlit app.** Both the Groq SDK and most frontier SDKs support token-by-token streaming; wiring this through `st.write_stream` would cut perceived latency significantly.

- **Replace keyword guardrail with LlamaGuard.** Purpose-built safety classifier; operates on semantic meaning rather than surface patterns; has published false-positive/false-negative rates.

- **Inter-rater reliability for the judge.** Running each pair through the judge twice at different temperatures and computing Cohen's kappa would put confidence intervals on the reported averages.

- **Persistent eval results on Render.** The free tier's ephemeral filesystem means eval results are lost on redeploy. Adding an S3 bucket or Supabase would make results persistent across deployments.
