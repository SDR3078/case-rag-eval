"""Anthropic Claude generation step.

Loads a system prompt from `prompts/`, formats retrieved chunks into a `<context>`
block, and calls Claude. Prompt caching is enabled on the LAST text block of the
system message — context changes per query, so we don't cache it.

Refusal-floor short-circuit: if `top1_score < tau_low`, this module is bypassed
and the canonical refusal is returned without a model call. The caller in
`app.py` and `evals/run.py` handles that branch; `generate()` here always calls
the API.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic

from retriever import RetrievalResult

PROJECT_ROOT = Path(__file__).resolve().parent

CANONICAL_REFUSAL = "I don't know based on the provided FAQs."


def _format_context(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a `<context>` block.

    Each chunk is wrapped in a `<doc>` element carrying `id`, `section`, and
    `question` attributes so the model can copy them verbatim into its citation.
    XML-style tags are used because Claude is well-trained to attend to them.
    """
    parts = ["<context>"]
    for r in results:
        c = r.chunk
        # Escape double quotes in attributes; FAQ headings rarely contain them
        # but we still want to be safe against malformed source data.
        section_attr = c.section.replace('"', "&quot;")
        question_attr = c.question.replace('"', "&quot;")
        parts.append(
            f'<doc id="{c.id}" section="{section_attr}" question="{question_attr}">'
        )
        parts.append(c.text)
        parts.append("</doc>")
    parts.append("</context>")
    return "\n".join(parts)


class Generator:
    """Wraps the Anthropic call: load prompt, format context, generate.

    The prompt file is read once at construction. Caching is enabled by default
    on the system prompt; pass `prompt_caching=False` to disable for a control run.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        prompt_path: str = "prompts/system_v1.md",
        max_tokens: int = 600,
        temperature: float = 0.0,
        prompt_caching: bool = True,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        prompt_file = (PROJECT_ROOT / prompt_path).resolve()
        if not prompt_file.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        self.model = model
        self.prompt_path = prompt_path
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.prompt_caching = bool(prompt_caching)
        self._system_prompt = prompt_file.read_text(encoding="utf-8").strip()

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY env var is required. "
                "Set it before calling Generator.generate()."
            )
        self._client = anthropic.Anthropic()

    # -- public ----------------------------------------------------------

    def generate(self, query: str, results: list[RetrievalResult]) -> dict:
        """Run a single Claude call and return the answer + metadata.

        Returns a dict:
          - `answer` (str): the model's text reply.
          - `cited_ids` (list[str]): the chunk IDs we passed in. The model
            cites by section/question text, not by ID, so this is the *candidate*
            set, not what was actually cited.
          - `stop_reason` (str): from the API response (`end_turn`, `max_tokens`, etc.).
          - `usage` (dict): the API's input/output token counts (incl. cache stats).

        Side effects: one HTTP call to the Anthropic API.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(results, list):
            raise TypeError("results must be a list of RetrievalResult")

        system_blocks = self._build_system_blocks()
        context_block = _format_context(results)
        user_text = f"{context_block}\n\nQuestion: {query.strip()}"

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_blocks,
            messages=[{"role": "user", "content": user_text}],
        )

        answer_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        answer = "".join(answer_parts).strip()

        return {
            "answer": answer,
            "cited_ids": [r.chunk.id for r in results],
            "stop_reason": resp.stop_reason,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cache_creation_input_tokens": getattr(
                    resp.usage, "cache_creation_input_tokens", 0
                ),
                "cache_read_input_tokens": getattr(
                    resp.usage, "cache_read_input_tokens", 0
                ),
            },
        }

    # -- internal --------------------------------------------------------

    def _build_system_blocks(self) -> list[dict]:
        """Build the `system` parameter as a list of TextBlockParam dicts.

        With caching enabled, we put the entire prompt in a single block and
        attach `cache_control={"type": "ephemeral"}` to the LAST block so all
        of the system content above it is cacheable.
        """
        block: dict = {"type": "text", "text": self._system_prompt}
        if self.prompt_caching:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]
