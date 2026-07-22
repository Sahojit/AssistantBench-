"""
Evaluation runner.

Usage:
    python evaluation/runner.py

Runs all 30 prompts against both assistants, judges each response with
an LLM judge, saves results/eval_results.json, and additionally computes
per-call latency/cost stats saved to results/latency_report.json and
cost_latency_table.md.
"""

import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from assistants.frontier_assistant import FrontierAssistant  # noqa: E402
from assistants.oss_assistant import OSSAssistant  # noqa: E402
from evaluation.judge import Judge  # noqa: E402
from evaluation.prompts import get_all_prompts  # noqa: E402
from observability.redis_client import get_client as get_redis_client  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_PATH = _PROJECT_ROOT / "results" / "eval_results.json"
LATENCY_REPORT_PATH = _PROJECT_ROOT / "results" / "latency_report.json"
COST_TABLE_PATH = _PROJECT_ROOT / "cost_latency_table.md"
INTER_CALL_DELAY = 2.0

_RUN_KEY_PREFIX = "llm_benchmark:eval_run:"


def _run_key(run_id: str) -> str:
    return f"{_RUN_KEY_PREFIX}{run_id}"


def _save_result_to_redis(redis_client, run_id: str, result: dict) -> None:
    """Persist a single prompt's result to Redis immediately after it's scored.

    Uses a hash keyed by run_id with one field per prompt_id, so a crash
    mid-run only loses the prompt currently in flight — every completed
    prompt is already durable. Non-fatal if Redis is unreachable.
    """
    try:
        redis_client.hset(_run_key(run_id), str(result["prompt_id"]), json.dumps(result))
    except Exception as exc:
        logger.warning("Redis write failed for prompt %s (non-fatal): %s", result["prompt_id"], exc)


def _load_results_from_redis(redis_client, run_id: str, fallback: list[dict]) -> list[dict]:
    """Read back all persisted results for a run, sorted by prompt_id.

    Falls back to the in-memory results list if Redis is unavailable or the
    hash is empty for any reason.
    """
    try:
        raw = redis_client.hgetall(_run_key(run_id))
        if not raw:
            return fallback
        records = [json.loads(v) for v in raw.values()]
        records.sort(key=lambda r: r["prompt_id"])
        return records
    except Exception as exc:
        logger.warning("Redis read-back failed, using in-memory results: %s", exc)
        return fallback


def _safe_generate(assistant, prompt: str, label: str) -> tuple[str, float]:
    """
    Call assistant.generate_for_eval and return (text, latency_ms).

    Returns ("ERROR", 0.0) on failure so the caller can handle gracefully.

    Args:
        assistant: An OSSAssistant or FrontierAssistant instance.
        prompt:    The raw evaluation prompt (no guardrail filtering applied).
        label:     Human-readable label used in log messages.
    """
    try:
        return assistant.generate_for_eval(prompt)
    except Exception as exc:
        logger.error("%s failed: %s", label, exc)
        return "ERROR", 0.0


def _safe_score(judge: Judge, category: str, prompt: str, response: str, label: str):
    """
    Call judge.score and return the scores dict, or None on failure.

    Args:
        judge:    Initialised Judge instance.
        category: Prompt category string.
        prompt:   The original evaluation prompt.
        response: The model response to score.
        label:    Human-readable label used in log messages.
    """
    if response == "ERROR":
        return None
    try:
        return judge.score(category, prompt, response)
    except Exception as exc:
        logger.error("Judging %s failed: %s", label, exc)
        return None


def _estimate_tokens(prompt: str, response: str) -> int:
    """Estimate combined token count using word × 1.3 heuristic."""
    words = len(prompt.split()) + len(response.split())
    return round(words * 1.3)


def _build_latency_report(oss_latencies: list[float], oss_tokens: list[int]) -> dict:
    """
    Compute summary statistics for the OSS model's eval run.

    Args:
        oss_latencies: Per-call latency in milliseconds (excluding ERROR calls).
        oss_tokens:    Per-call estimated token counts.

    Returns:
        Dict matching the latency_report.json schema.
    """
    valid = [l for l in oss_latencies if l > 0]
    avg_lat = statistics.mean(valid) if valid else 0.0
    p50 = statistics.median(valid) if valid else 0.0
    p95 = sorted(valid)[int(len(valid) * 0.95)] if valid else 0.0
    avg_tok = round(statistics.mean(oss_tokens)) if oss_tokens else 0

    return {
        "model": "Llama 3.1 8B Instant (OSS) vs Llama 3.3 70B Versatile (Frontier)",
        "provider": "Groq (free tier)",
        "total_calls": len(oss_latencies),
        "avg_latency_ms": round(avg_lat, 1),
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
        "avg_tokens_per_call": avg_tok,
        "cost_per_1k_tokens": "$0.00 (free tier)",
        "total_estimated_cost": "$0.00",
    }


def _write_cost_table(report: dict) -> str:
    """
    Render a markdown table from the latency report and write it to disk.

    Args:
        report: The latency report dict from _build_latency_report().

    Returns:
        The markdown table as a string (also printed to console).
    """
    rows = [
        ("Model", report["model"]),
        ("Provider", report["provider"]),
        ("Total calls", str(report["total_calls"])),
        ("Avg latency (ms)", str(report["avg_latency_ms"])),
        ("P50 latency (ms)", str(report["p50_latency_ms"])),
        ("P95 latency (ms)", str(report["p95_latency_ms"])),
        ("Avg tokens / call", str(report["avg_tokens_per_call"])),
        ("Cost per 1k tokens", report["cost_per_1k_tokens"]),
        ("Total estimated cost", report["total_estimated_cost"]),
    ]

    lines = [
        "# Cost & Latency Report",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for metric, value in rows:
        lines.append(f"| {metric} | {value} |")

    table = "\n".join(lines) + "\n"
    COST_TABLE_PATH.write_text(table, encoding="utf-8")
    return table


def run_evaluation() -> dict:
    """
    Execute the full benchmark: all 30 prompts × 2 assistants × judge scoring.

    Additionally tracks per-call latency for the OSS model and writes:
    - results/eval_results.json
    - results/latency_report.json
    - cost_latency_table.md  (also printed to console)
    """
    oss = OSSAssistant()
    frontier = FrontierAssistant()
    judge = Judge()
    prompts = get_all_prompts()

    run_id = str(uuid4())
    redis_client = get_redis_client()
    if redis_client is not None:
        logger.info("Persisting per-prompt results to Redis (run_id=%s).", run_id)

    results = []
    oss_latencies: list[float] = []
    oss_tokens: list[int] = []

    for idx, ep in enumerate(prompts, start=1):
        print(f"Running prompt {idx}/{len(prompts)} — [{ep.category}] {ep.prompt[:70]}...")

        # ---- OSS response ------------------------------------------------
        oss_response, oss_latency = _safe_generate(oss, ep.prompt, f"OSS prompt {ep.id}")
        oss_latencies.append(oss_latency)
        oss_tokens.append(_estimate_tokens(ep.prompt, oss_response))
        time.sleep(INTER_CALL_DELAY)

        # ---- Frontier response -------------------------------------------
        frontier_response, _ = _safe_generate(
            frontier, ep.prompt, f"Frontier prompt {ep.id}"
        )
        time.sleep(INTER_CALL_DELAY)

        # ---- Judge: OSS --------------------------------------------------
        oss_scores = _safe_score(
            judge, ep.category, ep.prompt, oss_response, f"OSS scores prompt {ep.id}"
        )
        time.sleep(INTER_CALL_DELAY)

        # ---- Judge: Frontier ---------------------------------------------
        frontier_scores = _safe_score(
            judge,
            ep.category,
            ep.prompt,
            frontier_response,
            f"Frontier scores prompt {ep.id}",
        )
        time.sleep(INTER_CALL_DELAY)

        result = {
            "prompt_id": ep.id,
            "category": ep.category,
            "prompt": ep.prompt,
            "oss_response": oss_response,
            "oss_latency_ms": round(oss_latency, 1),
            "frontier_response": frontier_response,
            "oss_scores": oss_scores,
            "frontier_scores": frontier_scores,
        }
        results.append(result)

        if redis_client is not None:
            _save_result_to_redis(redis_client, run_id, result)

    # ---- Save eval results -------------------------------------------------
    # Sourced from Redis when available so a crash mid-run doesn't lose
    # completed prompts; falls back to the in-memory list otherwise.
    final_results = (
        _load_results_from_redis(redis_client, run_id, results)
        if redis_client is not None
        else results
    )
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": final_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    # ---- Latency report --------------------------------------------------
    latency_report = _build_latency_report(oss_latencies, oss_tokens)
    with open(LATENCY_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(latency_report, fh, indent=2)

    # ---- Cost/latency table ----------------------------------------------
    table = _write_cost_table(latency_report)
    print("\n" + table)
    print(f"Evaluation complete.")
    print(f"  Results  -> {RESULTS_PATH}")
    print(f"  Latency  -> {LATENCY_REPORT_PATH}")
    print(f"  Table    -> {COST_TABLE_PATH}")
    return output


if __name__ == "__main__":
    run_evaluation()
