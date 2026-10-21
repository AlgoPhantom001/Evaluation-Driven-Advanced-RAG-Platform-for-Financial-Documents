from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent

sys.path.insert(0, str(_ROOT / "src"))

os.environ.setdefault("SECRAG_DATA_DIR", str(_ROOT / "data"))
os.environ.setdefault("SECRAG_ENABLE_SPLADE", "false")

import gradio as gr

from secrag.core.types import QueryRequest
from secrag.engine import build_engine

try:
    import spaces

    @spaces.GPU(duration=10)
    def gpu_diagnostic() -> str:
        try:
            import onnxruntime

            providers = onnxruntime.get_available_providers()
        except Exception:
            providers = []
        return f"ONNX Runtime providers visible to this worker: {providers}"

except ImportError:
    gpu_diagnostic = None


ENGINE = build_engine()
ENGINE.warmup()
STATS = ENGINE.stats()

EXAMPLES = [
    "What supply chain risks does Apple disclose?",
    "What was Apple's gross margin in fiscal 2024?",
    "What does NVIDIA say about export controls?",
    "How much did Microsoft's revenue grow from 2022 to 2024?",
    "What credit risks does JPMorgan disclose?",
    "What was Walmart's revenue in fiscal 2026?",
]

SECTION_LABELS = {
    "item_1_business": "Item 1 Business",
    "item_1a_risk_factors": "Item 1A Risk Factors",
    "item_3_legal_proceedings": "Item 3 Legal Proceedings",
    "item_7_mda": "Item 7 MD&A",
    "item_7a_market_risk": "Item 7A Market Risk",
    "item_8_financial_statements": "Item 8 Financial Statements",
    "item_9a_controls": "Item 9A Controls",
    "other": "Other",
}


def _figures_md(results: list[Any]) -> str:
    if not results:
        return ""
    rows = ["### Verified figures", "", "_Computed from filed XBRL data, not generated._", ""]
    for r in results:
        if r.value is None:
            continue
        value = f"{r.value:,.2f}%" if r.unit == "percent" else f"{r.value:,.0f} {r.unit}"
        rows.append(f"**{r.label}**")
        rows.append(f"# {value}")
        inputs = "  ".join(f"`{k}` = {v:,.0f}" for k, v in r.inputs.items())
        rows.append(f"`{r.formula}`" + (f"<br>{inputs}" if inputs else ""))
        rows.append("")
    return "\n".join(rows) if len(rows) > 4 else ""


def _sources_md(contexts: list[Any]) -> str:
    if not contexts:
        return "_No passages retrieved._"
    parts = ["### Sources", ""]
    for i, scored in enumerate(contexts, start=1):
        chunk = scored.chunk
        section = SECTION_LABELS.get(chunk.section.value, chunk.section.value)
        arms = (
            " + ".join(k for k in scored.component_scores if not k.endswith("_rank"))
            or scored.stage
        )
        body = chunk.text.strip()
        if len(body) > 700:
            body = body[:700] + " ..."
        parts.append(
            f"<details><summary><b>[{i}] {chunk.ticker} FY{chunk.fiscal_year} "
            f"{section}</b> <code>{arms}</code> "
            f"score {scored.score:.3f}</summary>\n\n"
            f"{body}\n\n[View filing]({chunk.source_url})\n\n</details>"
        )
    return "\n".join(parts)


def _diagnostics_md(response: Any) -> str:
    route = response.route
    answer = response.answer
    rows = [
        "### Diagnostics",
        "",
        "| | |",
        "|---|---|",
        f"| Intent | `{route.intent.value}` ({route.confidence:.2f}) |" if route else "",
        f"| Status | `{answer.status.value}` |",
        f"| Groundedness | {answer.groundedness:.3f} |",
        f"| Citations | {len(answer.citations)} |",
        f"| Passages retrieved | {len(response.contexts)} |",
        f"| Served from cache | {response.cached} |",
        f"| Latency | {response.latency_ms:,.0f} ms |",
    ]
    if answer.refusal_reason:
        rows += ["", f"> **Withheld:** {answer.refusal_reason}"]
    return "\n".join(r for r in rows if r)


async def ask(
    question: str, company: str, reranker: str, use_cache: bool
) -> tuple[str, str, str, str]:
    if not question or not question.strip():
        return "Ask a question to begin.", "", "", ""

    request = QueryRequest(
        question=question.strip(),
        top_k=6,
        companies=[] if company == "All" else [company],
        reranker=reranker,
        use_reranker=reranker != "none",
        use_cache=use_cache,
    )

    try:
        response = await ENGINE.answer(request)
    except Exception as exc:
        return f"**Something went wrong.**\n\n```\n{type(exc).__name__}: {exc}\n```", "", "", ""

    answer = response.answer.text
    if response.answer.citations:
        answer += "\n\n---\n\n**Citations**\n\n"
        for c in response.answer.citations:
            answer += f"- **[{c.marker}]** {c.label} _(support {c.support_score:.2f})_\n"

    return (
        answer,
        _figures_md(response.numeric_results),
        _sources_md(response.contexts),
        _diagnostics_md(response),
    )


DESCRIPTION = f"""
# sec-rag-platform

**Evaluation-driven retrieval-augmented generation over SEC 10-K filings.**

Hybrid retrieval across dense and lexical arms, fused by Reciprocal Rank Fusion and reranked.
Financial figures are **computed from filed XBRL data rather than generated by the model**,
and every answer is verified against the passages it cites before being returned.

`{STATS["corpus_chunks"]:,}` chunks | `{STATS["xbrl_rows"]:,}` XBRL facts |
{" ".join(STATS["tickers"])} | provider `{(STATS["providers"] or ["offline"])[0]}`

[Source on GitHub](https://github.com/AlgoPhantom001/Evaluation-Driven-Advanced-RAG-Platform-for-Financial-Documents) |
Measured: nDCG@6 0.763, groundedness 0.781, numeric accuracy 1.000
"""

with gr.Blocks(title="sec-rag-platform") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        question = gr.Textbox(
            label="Question",
            placeholder="Ask about the indexed 10-K filings...",
            scale=5,
            autofocus=True,
        )
        submit = gr.Button("Ask", variant="primary", scale=1)

    with gr.Row():
        company = gr.Dropdown(
            choices=["All", *STATS["tickers"]], value="All", label="Company", scale=1
        )
        reranker = gr.Dropdown(
            choices=[
                ("cross-encoder (most accurate)", "cross_encoder"),
                ("learning-to-rank (fastest)", "ltr"),
                ("none (fusion order only)", "none"),
            ],
            value="ltr",
            label="Reranker",
            scale=2,
        )
        use_cache = gr.Checkbox(value=True, label="Semantic cache", scale=1)

    gr.Examples(examples=EXAMPLES, inputs=question, label="Try one")

    with gr.Row():
        with gr.Column(scale=3):
            answer_out = gr.Markdown(label="Answer", value="Ask a question to begin.")
            figures_out = gr.Markdown()
        with gr.Column(scale=2):
            sources_out = gr.Markdown()
            diagnostics_out = gr.Markdown()

    gr.Markdown(
        "---\n"
        "Filing data from [SEC EDGAR](https://www.sec.gov/edgar). "
        "This is a technical demonstration, not investment advice."
    )

    inputs = [question, company, reranker, use_cache]
    outputs = [answer_out, figures_out, sources_out, diagnostics_out]
    submit.click(ask, inputs=inputs, outputs=outputs)
    question.submit(ask, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.queue(max_size=16).launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
    )
