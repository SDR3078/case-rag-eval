"""OpenAI-compatible chat-completions generation step.

Loads a system prompt from `prompts/`, formats retrieved chunks into a
`<context>` block, and calls an OpenAI-compatible chat-completions endpoint.

Provider-agnostic: `base_url` is configurable so the same code talks to OpenAI
proper, Azure OpenAI, OpenRouter, Together, Groq, LM Studio, Ollama, vLLM, or
any other OpenAI-compatible server. Reviewers swap providers via the
`OPENAI_BASE_URL` env var or the `generation.base_url` YAML field — no code
change.

Prompt caching: OpenAI-compatible providers handle prefix caching
automatically (OpenAI itself caches prompts at >=1024 tokens; others vary).
The system prompt is the natural cacheable prefix; the retrieved-context
block sits in the user message and varies per query.

Refusal-floor short-circuit: if `top1_score < tau_low`, `app.py` /
`evals/run.py` short-circuit and emit the canonical refusal without ever
calling this module. `generate()` always issues the API request.
"""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

from retriever import RetrievalResult

PROJECT_ROOT = Path(__file__).resolve().parent

CANONICAL_REFUSAL = "I don't know based on the provided FAQs."


def _format_context(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a `<context>` block.

    Each chunk is wrapped in a `<doc>` element carrying `id`, `section`, and
    `question` attributes so the model can copy them verbatim into its citation.
    XML-style tags travel well across providers — most instruction-tuned chat
    models attend to tag boundaries.
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
    """Wraps the OpenAI-compatible chat call: load prompt, format context, generate.

    The prompt file is read once at construction. `base_url` lets you point at
    any OpenAI-compatible endpoint; if omitted the SDK uses the default OpenAI
    URL. `OPENAI_API_KEY` is required; some providers accept a stand-in value
    (e.g. local LM Studio takes any non-empty string).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        prompt_path: str = "prompts/system_v1.md",
        max_tokens: int = 600,
        temperature: float = 0.0,
        base_url: str | None = None,
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
        # YAML base_url overrides env; both can be unset (= default OpenAI URL).
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._system_prompt = prompt_file.read_text(encoding="utf-8").strip()

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY env var is required. "
                "Set it before calling Generator.generate()."
            )

        client_kwargs: dict = {}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        # max_retries=10 lets the SDK's exponential backoff absorb provider
        # rate limits (Groq's free tier in particular) and transient
        # connection errors without manual retry plumbing.
        self._client = OpenAI(max_retries=10, **client_kwargs)

    # -- public ----------------------------------------------------------

    def generate(self, query: str, results: list[RetrievalResult]) -> dict:
        """Run a single chat-completion call and return the answer + metadata.

        Returns a dict:
          - `answer` (str): the model's text reply.
          - `cited_ids` (list[str]): the chunk IDs we passed in. The model
            cites by section/question text, not by ID, so this is the *candidate*
            set, not what was actually cited.
          - `stop_reason` (str): from the API response (`stop`, `length`, etc.).
          - `usage` (dict): prompt/completion/total token counts plus
            `cached_tokens` when the provider reports prefix-cache hits.

        Side effects: one HTTP call to the configured OpenAI-compatible endpoint.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(results, list):
            raise TypeError("results must be a list of RetrievalResult")

        context_block = _format_context(results)
        user_text = f"{context_block}\n\nQuestion: {query.strip()}"

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_text},
            ],
        )

        choice = resp.choices[0]
        answer = (choice.message.content or "").strip()

        usage_dict: dict = {}
        if resp.usage is not None:
            usage_dict = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            # OpenAI proper exposes prompt_tokens_details.cached_tokens; many
            # providers don't. Surface it when available so eval logs can see
            # cache hit rates.
            details = getattr(resp.usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", None)
                if cached is not None:
                    usage_dict["cached_tokens"] = cached

        return {
            "answer": answer,
            "cited_ids": [r.chunk.id for r in results],
            "stop_reason": choice.finish_reason,
            "usage": usage_dict,
        }
