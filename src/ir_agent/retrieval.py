from __future__ import annotations

import math
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Small synchronous contract for optional semantic retrieval."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector for every input text."""


class OpenAIEmbeddingProvider:
    """OpenAI-compatible embedding adapter, enabled only when configured."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is not installed; install the base project dependencies."
            ) from exc

        client_kwargs: dict[str, object] = {"timeout": timeout}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization
        self._client = OpenAI(**client_kwargs)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        data = sorted(response.data, key=lambda item: item.index)
        vectors = [list(map(float, item.embedding)) for item in data]
        if len(vectors) != len(texts):
            raise ValueError("Embedding provider returned an unexpected vector count")
        return vectors


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))
