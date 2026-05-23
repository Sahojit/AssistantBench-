# LLM Assistant Benchmark

A side-by-side comparison framework for two AI assistants — an open-source model (Qwen2.5-0.5B-Instruct via HuggingFace Inference API) and a frontier model (Gemini 1.5 Flash) — with shared guardrails, memory, observability, a 30-prompt evaluation suite judged by an LLM, and automated PDF report generation.

---

## Architecture Decisions

| Component | Choice | Reason |
|---|---|---|
| **OSS model** | Qwen2.5-0.5B-Instruct | Smallest Qwen2.5 instruction-tuned model; freely available via HF Serverless Inference with no local GPU required |
| **Frontier model** | Gemini 1.5 Flash | Best latency/quality trade-off in the Gemini family; free tier available; 1M-token context window |
| **UI framework** | Streamlit | Minimal boilerplate for data-app prototypes; built-in chat primitives match the use case |
| **Guardrails** | Custom regex + keyword lists | Zero-latency, no extra API call; deterministic; tunable without redeployment |
| **Observability** | LangFuse | Open-source LLM tracing platform with a generous cloud free tier; SDK wraps any model |
| **LLM judge** | Gemini 1.5 Flash | Same frontier model already in the dependency graph; prompt-engineered to return structured JSON |
| **PDF generation** | ReportLab | Pure-Python, no headless browser needed; gives precise layout control for charts + tables |

---

## Setup Instructions

1. **Clone / enter the project directory**
   ```bash
   cd llm-benchmark
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   # .venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Copy and populate the environment file**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and fill in:
   - `GOOGLE_API_KEY` — from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - `HF_TOKEN` — from [HuggingFace Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` — from [LangFuse Cloud](https://cloud.langfuse.com) *(optional; app works without these)*

5. **Verify the environment (optional quick check)**
   ```bash
   python -c "from assistants.oss_assistant import OSSAssistant; print('OK')"
   ```

---

## How to Run the App

```bash
streamlit run app.py
```

Open the URL printed in the terminal (typically `http://localhost:8501`).

- Use the **OSS Assistant** tab to chat with Qwen2.5-0.5B-Instruct.
- Use the **Frontier Assistant** tab to chat with Gemini 1.5 Flash.
- The sidebar shows live memory window size and estimated token usage per assistant.
- The **Run Full Evaluation** button in the sidebar streams progress and saves results on completion.

---

## How to Run the Evaluation

```bash
python evaluation/runner.py
```

This runs all 30 prompts against both assistants, scores each response with the Gemini Flash judge, and writes `results/eval_results.json`.  Expect approximately 5–10 minutes for the full run (30 prompts × 4 API calls × 1 s delay between calls).

---

## How to Generate the Report

```bash
python report/generate_report.py
```

Reads `results/eval_results.json` and produces:
- `report/charts/chart_accuracy.png` — accuracy bar chart
- `report/charts/chart_safety.png` — safety bar chart
- `report/charts/chart_bias.png` — bias bar chart
- `report/charts/chart_helpfulness.png` — helpfulness bar chart
- `report/evaluation_report.pdf` — full PDF with all charts, summary table, and a Gemini-generated recommendation paragraph

---

## Tradeoffs Made

- **Qwen2.5-0.5B is very small for a fair comparison.** At 0.5 billion parameters, Qwen2.5-0.5B will almost always underperform Gemini 1.5 Flash, which has orders-of-magnitude more capacity. The benchmark is most useful for characterising the gap rather than declaring a competitive race.

- **Guardrails are keyword-based, not semantic.** The input filter catches adversarial prompts that contain exact keyword matches but can be evaded by paraphrasing. A production system would add an embedding-based classifier or a dedicated safety model (e.g., LlamaGuard) as an additional layer.

- **The LLM judge is the same model as one of the assistants.** Using Gemini Flash both as a benchmark participant and as the judge introduces potential self-serving bias (the judge may implicitly favour responses that resemble its own style). An independent judge model would remove this conflict.

---

## What I Would Improve With More Time

- **Replace the keyword guardrail with LlamaGuard or ShieldLM.** These models are purpose-built for safety classification, operate on semantic meaning rather than surface patterns, and have published false-positive and false-negative rates — giving the guardrail layer a measurable specification rather than ad-hoc keyword lists.

- **Add a streaming response mode to the Streamlit app.** Both HF InferenceClient and the Gemini SDK support token-by-token streaming; wiring this through `st.write_stream` would dramatically improve perceived latency for longer responses and make the two assistants easier to compare qualitatively in real time.

- **Implement an inter-rater reliability check for the judge.** Running each (prompt, response) pair through the judge twice with different temperature settings and computing Cohen's kappa would quantify how consistent the scoring is, allowing us to set confidence intervals on the reported averages rather than treating them as exact measurements.
