"""LLM-as-judge for faithfulness + correctness on a generated answer.

Uses the same OpenAI-compatible SDK pattern as `generator.py` so the judge
runs on any provider the generator runs on (set `model` and `base_url`
explicitly on the judge to use a different / stronger model than the one
under test).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from retriever import RetrievalResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "judge_v1.md"


def _format_context(retrieved: list[RetrievalResult]) -> str:
    """Embed the retrieved chunks verbatim so the judge can verify groundedness."""
    parts = ["<context>"]
    for r in retrieved:
        c = r.chunk
        section_attr = c.section.replace('"', "&quot;")
        question_attr = c.question.replace('"', "&quot;")
        parts.append(
            f'<doc section="{section_attr}" question="{question_attr}">'
        )
        parts.append(c.text)
        parts.append("</doc>")
    parts.append("</context>")
    return "\n".join(parts)


class Judge:
    """Wraps a separate chat-completions call to score a generated answer.

    `model` and `base_url` are independent of the generator under test — to
    avoid self-evaluation bias, point the judge at a stronger or different
    model when budget permits. With both omitted the judge uses the same
    model the generator does (cheap, simple, biased).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        prompt_path: str = "prompts/judge_v1.md",
        max_tokens: int = 200,
        temperature: float = 0.0,
        base_url: str | None = None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        prompt_file = (PROJECT_ROOT / prompt_path).resolve()
        if not prompt_file.is_file():
            raise FileNotFoundError(f"Judge prompt not found: {prompt_file}")

        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._prompt = prompt_file.read_text(encoding="utf-8").strip()

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY env var is required for the judge."
            )

        client_kwargs: dict[str, Any] = {}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        # max_retries=10 absorbs rate-limit bursts on free-tier providers.
        # See generator.py for the rationale.
        self._client = OpenAI(max_retries=10, **client_kwargs)

    def score(
        self,
        question: str,
        retrieved: list[RetrievalResult],
        answer: str,
        expected_themes: list[str],
    ) -> dict:
        """Return `{faithfulness, correctness, rationale}`.

        Faithfulness and correctness are integers in [0, 5], or `-1` if the
        judge returned malformed JSON. Rationale is a one-sentence string.
        """
        if not isinstance(answer, str):
            raise TypeError("answer must be a string")
        themes_block = (
            "\n".join(f"- {t}" for t in expected_themes)
            if expected_themes
            else "(none — this is an out-of-scope case; the answer should refuse)"
        )
        user = (
            f"USER QUESTION:\n{question.strip()}\n\n"
            f"RETRIEVED CONTEXT:\n{_format_context(retrieved)}\n\n"
            f"GENERATED ANSWER:\n{answer.strip()}\n\n"
            f"EXPECTED THEMES:\n{themes_block}\n"
        )

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": user},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {
                "faithfulness": -1,
                "correctness": -1,
                "rationale": f"judge parse error: {content[:120]}",
            }

        def _clip(x: Any) -> int:
            try:
                v = int(x)
            except (TypeError, ValueError):
                return -1
            return max(-1, min(5, v))

        return {
            "faithfulness": _clip(parsed.get("faithfulness")),
            "correctness": _clip(parsed.get("correctness")),
            "rationale": str(parsed.get("rationale", ""))[:200],
        }
