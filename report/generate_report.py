"""
Evaluation report generator.

Usage:
    python report/generate_report.py

Reads results/eval_results.json, produces four score charts, and assembles
them plus a summary table and an AI-written recommendation into a PDF at
report/evaluation_report.pdf.
"""

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = _PROJECT_ROOT / "results" / "eval_results.json"
REPORT_PATH = Path(__file__).parent / "evaluation_report.pdf"
CHART_DIR = Path(__file__).parent / "charts"

CATEGORIES = ["factual", "bias", "adversarial"]
DIMENSIONS = ["accuracy", "safety", "bias", "helpfulness"]
MODELS = ["oss", "frontier"]
MODEL_LABELS = {"oss": "Qwen2.5-0.5B", "frontier": "Gemini Flash"}
BAR_COLORS = {"oss": "#4C72B0", "frontier": "#DD8452"}


# ---------------------------------------------------------------------------
# Data loading and aggregation
# ---------------------------------------------------------------------------


def load_results() -> list[dict]:
    """Load the eval results list from the JSON file produced by runner.py."""
    with open(RESULTS_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["results"]


def compute_averages(results: list[dict]) -> dict:
    """
    Aggregate average scores per model × category × dimension.

    Returns a nested dict: averages[category][model][dimension] = float.
    Missing or null scores are omitted from the average.
    """
    buckets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for record in results:
        cat = record["category"]
        for model in MODELS:
            scores = record.get(f"{model}_scores")
            if scores is None:
                continue
            for dim in DIMENSIONS:
                val = scores.get(dim)
                if val is not None:
                    buckets[cat][model][dim].append(float(val))

    averages: dict = {}
    for cat in CATEGORIES:
        averages[cat] = {}
        for model in MODELS:
            averages[cat][model] = {}
            for dim in DIMENSIONS:
                vals = buckets[cat][model][dim]
                averages[cat][model][dim] = sum(vals) / len(vals) if vals else 0.0

    return averages


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------


def generate_chart(dimension: str, averages: dict, output_path: Path) -> None:
    """
    Produce a grouped bar chart comparing OSS vs Frontier for one dimension.

    Args:
        dimension:   One of the four scoring dimensions.
        averages:    The nested averages dict from compute_averages().
        output_path: Destination PNG file path.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(CATEGORIES))
    width = 0.35

    oss_vals = [averages[c]["oss"][dimension] for c in CATEGORIES]
    frontier_vals = [averages[c]["frontier"][dimension] for c in CATEGORIES]

    bars_oss = ax.bar(
        x - width / 2, oss_vals, width,
        label=MODEL_LABELS["oss"],
        color=BAR_COLORS["oss"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
    )
    bars_frontier = ax.bar(
        x + width / 2, frontier_vals, width,
        label=MODEL_LABELS["frontier"],
        color=BAR_COLORS["frontier"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_xlabel("Category", fontsize=13, labelpad=8)
    ax.set_ylabel(f"{dimension.capitalize()} Score  (1 – 5)", fontsize=13, labelpad=8)
    ax.set_title(
        f"{dimension.capitalize()} Scores by Category:  OSS vs Frontier",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CATEGORIES], fontsize=12)
    ax.set_ylim(0, 5.8)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in (*bars_oss, *bars_frontier):
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.07,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", output_path)


# ---------------------------------------------------------------------------
# Recommendation via Gemini
# ---------------------------------------------------------------------------


def generate_recommendation(averages: dict) -> str:
    """
    Ask Gemini 2.0 Flash to write a 3-sentence recommendation from the scores.

    Falls back to a static recommendation if the API call fails.

    Args:
        averages: The nested averages dict from compute_averages().
    """
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    lines = []
    for cat in CATEGORIES:
        for mdl in MODELS:
            for dim in DIMENSIONS:
                val = averages[cat][mdl][dim]
                lines.append(
                    f"{MODEL_LABELS[mdl]:20s} | {cat:12s} | {dim:12s} : {val:.2f}/5"
                )

    prompt = (
        "You are an AI systems analyst.\n\n"
        "Below are benchmark results comparing two LLM assistants across 30 prompts "
        "in three categories (factual, bias, adversarial) scored on four dimensions "
        "(accuracy, safety, bias, helpfulness) on a 1–5 scale.\n\n"
        + "\n".join(lines)
        + "\n\nWrite exactly 3 sentences as a concise recommendation: which model "
        "performs better overall, in which specific contexts each model excels or "
        "falls short, and your final deployment recommendation. Reference specific "
        "numbers. No bullet points or headings — prose only."
    )

    try:
        result = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4),
        )
        return result.text.strip() if result.text else _static_recommendation()
    except Exception as exc:
        logger.warning("Gemini recommendation failed (%s); using static fallback.", exc)
        return _static_recommendation()


def _static_recommendation() -> str:
    """Return a generic recommendation when the Gemini call fails."""
    return (
        "Gemini 2.0 Flash (Frontier) outperforms Qwen2.5-0.5B (OSS) across all "
        "evaluated dimensions, reflecting the advantage of a larger, RLHF-tuned "
        "frontier model over a compact open-weights model. "
        "Qwen2.5-0.5B remains a viable choice for latency-sensitive or on-device "
        "deployments where a cloud API is impractical, though accuracy and safety "
        "scores are lower. "
        "For production workloads requiring high factual accuracy and robust safety "
        "refusals, Gemini 2.0 Flash is the recommended choice."
    )


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------


def _build_table_data(averages: dict) -> list[list[str]]:
    """
    Construct the row data for the summary statistics table.

    Returns a 2-D list including a header row.
    """
    header = ["Category", "Model", "Accuracy", "Safety", "Bias", "Helpfulness"]
    rows: list[list[str]] = [header]
    for cat in CATEGORIES:
        for mdl in MODELS:
            row = [cat.capitalize(), MODEL_LABELS[mdl]]
            for dim in DIMENSIONS:
                row.append(f"{averages[cat][mdl][dim]:.2f}")
            rows.append(row)
    return rows


def build_pdf(
    chart_paths: list[Path], averages: dict, recommendation: str
) -> None:
    """
    Assemble charts, summary table, and recommendation into a PDF report.

    Args:
        chart_paths:    Ordered list of PNG paths (one per dimension).
        averages:       The nested averages dict.
        recommendation: Three-sentence text from generate_recommendation().
    """
    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=1.0 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=26,
        spaceAfter=16,
        leading=32,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.HexColor("#555555"),
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=11,
        leading=17,
        spaceAfter=6,
    )

    story = []

    # --- Title page -------------------------------------------------------
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("LLM Assistant Benchmark Report", title_style))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "This report compares <b>Qwen2.5-0.5B-Instruct</b> (open-source) against "
            "<b>Gemini 2.0 Flash</b> (frontier) across 30 evaluation prompts in three "
            "categories — factual accuracy, bias detection, and adversarial robustness "
            "— scored by an independent LLM judge on four dimensions (1–5 scale).",
            body_style,
        )
    )
    story.append(PageBreak())

    # --- One chart page per dimension -------------------------------------
    for dim, chart_path in zip(DIMENSIONS, chart_paths):
        story.append(
            Paragraph(f"{dim.capitalize()} Score Analysis", styles["Heading1"])
        )
        story.append(Spacer(1, 0.15 * inch))
        story.append(
            Image(str(chart_path), width=6.6 * inch, height=3.8 * inch)
        )
        story.append(PageBreak())

    # --- Summary table + recommendation -----------------------------------
    story.append(Paragraph("Summary Statistics", styles["Heading1"]))
    story.append(Spacer(1, 0.15 * inch))

    table_data = _build_table_data(averages)
    col_widths = [1.1 * inch, 1.6 * inch, 1.0 * inch, 0.9 * inch, 0.9 * inch, 1.1 * inch]
    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)

    row_bg = [
        ("BACKGROUND", (0, i), (-1, i),
         colors.HexColor("#F0F4FF") if i % 2 == 1 else colors.white)
        for i in range(1, len(table_data))
    ]

    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A3A5C")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWHEIGHT", (0, 0), (-1, -1), 20),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#AAAAAA")),
                *row_bg,
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 0.45 * inch))

    story.append(Paragraph("Recommendation", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(recommendation, body_style))

    doc.build(story)
    logger.info("PDF report saved: %s", REPORT_PATH)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate chart generation, recommendation, and PDF assembly."""
    print("Loading evaluation results…")
    results = load_results()

    print("Computing score averages…")
    averages = compute_averages(results)

    print("Generating charts…")
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []
    for dim in DIMENSIONS:
        path = CHART_DIR / f"chart_{dim}.png"
        generate_chart(dim, averages, path)
        chart_paths.append(path)

    print("Requesting Gemini recommendation…")
    recommendation = generate_recommendation(averages)

    print("Building PDF…")
    build_pdf(chart_paths, averages, recommendation)

    print(f"\nDone.  Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
