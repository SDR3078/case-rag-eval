"""Small Gradio UI for the EU Taxonomy FAQ RAG chatbot.

Reuses ``app.answer_question`` for the full retrieve->rerank->generate path. If
``ANTHROPIC_API_KEY`` is not set we fall back to a retrieval-only view so the UI
is still useful for reviewers without a key.

Run::

    .venv/bin/python ui.py
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

import app
from retriever import Retriever

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "experiments" / "default.yaml"

NO_KEY_BANNER = (
    "**Set `ANTHROPIC_API_KEY` to enable answer generation.** "
    "Showing retrieved FAQ entries only."
)
NO_KEY_ANSWER = "[generation disabled - set ANTHROPIC_API_KEY]"
FLOOR_NOTE = (
    "_Refusal floor tripped: top-1 similarity below `tau_low` - "
    "no Claude call was made._"
)


def _load_config() -> dict:
    """Load the default experiment YAML used to wire up the pipeline."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_chunks(pairs: list[tuple[str, str]]) -> str:
    """Render ``(section, question)`` tuples as a markdown bullet list."""
    if not pairs:
        return "_No FAQ entries retrieved._"
    return "\n".join(f'- **{section}** / "{question}"' for section, question in pairs)


class UIState:
    """Per-process runtime: config + eagerly-built retriever.

    Retriever is built once at startup so the no-key path works straight away.
    Reranker and Generator are constructed inside ``app.answer_question`` per
    call - reusing app's wiring keeps the CLI and UI in lock-step.
    """

    def __init__(self) -> None:
        self.config = _load_config()
        self.retriever = Retriever(self.config["name"])

    def has_key(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))


def ask(state: UIState, query: str) -> tuple[str, str, str]:
    """Run one query. Returns ``(banner_md, answer_md, retrieved_md)``."""
    query = (query or "").strip()
    if not query:
        return ("", "_Please enter a question._", "")

    if not state.has_key():
        try:
            results = state.retriever.retrieve(query, k=5, method="dense")
        except ValueError as e:
            return ("", f"_error: {e}_", "")
        pairs = [(r.chunk.section, r.chunk.question) for r in results]
        top1 = results[0].score if results else 0.0
        retrieved = _format_chunks(pairs) + f"\n\n_Top-1 similarity: {top1:.3f}_"
        return (NO_KEY_BANNER, NO_KEY_ANSWER, retrieved)

    try:
        result = app.answer_question(query, state.config)
    except (ValueError, TypeError) as e:
        return ("", f"_error: {e}_", "")

    answer = result["answer"]
    retrieved = _format_chunks(result["cited_chunks"])
    retrieved += f"\n\n_Top-1 similarity: {result['top1_score']:.3f}_"
    if result["refused_at_floor"]:
        return ("", answer + "\n\n" + FLOOR_NOTE, retrieved)
    return ("", answer, retrieved)


def build_app(state: UIState):
    """Construct the Gradio Blocks app.

    Gradio is imported lazily so ``import ui`` succeeds even when gradio is not
    installed (it lives in ``requirements-optional.txt``).
    """
    import gradio as gr

    banner_value = "" if state.has_key() else NO_KEY_BANNER

    with gr.Blocks(title="EU Taxonomy FAQ chatbot") as demo:
        gr.Markdown("# EU Taxonomy FAQ chatbot")
        gr.Markdown(
            "Ask a question about the EU Taxonomy Navigator FAQs. "
            "Answers are grounded in the retrieved FAQ entries shown below."
        )
        banner = gr.Markdown(banner_value)
        question = gr.Textbox(
            label="Question",
            placeholder="e.g. What does the EU Taxonomy require for Article 8 reporting?",
            lines=2,
            max_lines=6,
        )
        ask_btn = gr.Button("Ask", variant="primary")
        answer_box = gr.Markdown(label="Answer")
        with gr.Accordion("Retrieved FAQs", open=False):
            retrieved_box = gr.Markdown()

        handler = lambda q: ask(state, q)
        outs = [banner, answer_box, retrieved_box]
        ask_btn.click(fn=handler, inputs=question, outputs=outs)
        question.submit(fn=handler, inputs=question, outputs=outs)
    return demo


def main() -> None:
    state = UIState()
    demo = build_app(state)
    demo.launch()


if __name__ == "__main__":
    main()
