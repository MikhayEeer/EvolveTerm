"""Embedding client abstraction for code snippets."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np
from openai import OpenAI

from .config import load_json_config
from .exceptions import EmbeddingUnavailableError


class EmbeddingClient(ABC):
    """Base interface for embedding generators."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError


class APIEmbeddingClient(EmbeddingClient):
    """Embedding client implemented via the official OpenAI SDK."""

    def __init__(self, config_name: str = "embed_config.json"):
        config = load_json_config(config_name)
        dimension = int(config.get("dimension", 0))
        if dimension <= 0:
            raise EmbeddingUnavailableError("Embedding dimension must be positive")
        super().__init__(dimension)
        self.base_url = config.get("base_url") or config.get("baseurl")
        self.api_key = config.get("api_key")
        self.model = config.get("model")
        self.payload_template = config.get("payload_template", {})
        if not self.base_url or not self.api_key:
            raise EmbeddingUnavailableError("Embedding base_url or API key missing")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed(self, text: str) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                **self.payload_template,
            )
        except Exception as exc:  # pragma: no cover - network path
            raise EmbeddingUnavailableError(f"Embedding provider error: {exc}") from exc

        if not response.data:
            raise EmbeddingUnavailableError("Embedding provider returned no data")
        vector = response.data[0].embedding
        if not vector:
            raise EmbeddingUnavailableError("Embedding provider returned empty vector")
        arr = np.array(vector, dtype=np.float32)
        if arr.shape[0] != self.dimension:
            raise EmbeddingUnavailableError(
                f"Expected dimension {self.dimension}, got {arr.shape[0]}"
            )
        return arr


class MockEmbeddingClient(EmbeddingClient):
    """Deterministic embedding generator for offline demos and tests."""

    def __init__(self, dimension: int = 64):
        super().__init__(dimension)

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        repeats = (self.dimension + len(digest) - 1) // len(digest)
        raw = (digest * repeats)[: self.dimension]
        vector = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        return vector


def build_embedding_client(config_name: str = "embed_config.json") -> EmbeddingClient:
    config = load_json_config(config_name)
    provider = config.get("provider", "mock").lower()
    if provider == "mock":
        dimension = int(config.get("dimension", 64))
        return MockEmbeddingClient(dimension=dimension)
    return APIEmbeddingClient(config_name=config_name)
