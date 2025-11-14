"""Embedding client abstraction for code snippets."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import requests

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
    """Generic HTTP-based embedding client.

    The config file must contain:
    - baseurl: URL for POST requests
    - api_key: credential passed via Authorization header
    - model: optional identifier included in the payload
    - dimension: embedding vector size
    - payload_template: optional dict merged into request body
    """

    def __init__(self, config_name: str = "embed_config.json"):
        config = load_json_config(config_name)
        dimension = int(config.get("dimension", 0))
        if dimension <= 0:
            raise EmbeddingUnavailableError("Embedding dimension must be positive")
        super().__init__(dimension)
        self.baseurl = config.get("baseurl")
        self.api_key = config.get("api_key")
        self.model = config.get("model")
        self.payload_template = config.get("payload_template", {})
        if not self.baseurl or not self.api_key:
            raise EmbeddingUnavailableError("Embedding baseurl or API key missing")

    def embed(self, text: str) -> np.ndarray:
        payload = {
            "model": self.model,
            "input": text,
        }
        payload.update(self.payload_template)

        response = requests.post(
            self.baseurl,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(payload),
            timeout=30,
        )
        if response.status_code >= 400:
            raise EmbeddingUnavailableError(
                f"Embedding provider error {response.status_code}: {response.text[:200]}"
            )
        data = response.json()
        vector = data.get("embedding") or (data.get("data") or [{}])[0].get("embedding")
        if not vector:
            raise EmbeddingUnavailableError("Embedding provider returned no vector")
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
