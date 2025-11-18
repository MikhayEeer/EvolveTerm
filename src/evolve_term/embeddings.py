"""Embedding client abstraction plus utility helpers for bulk vectorization."""

from __future__ import annotations

import argparse
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from openai import OpenAI

from .config import load_json_config
from .exceptions import EmbeddingUnavailableError

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_BULK_OUTPUT = DATA_DIR / "prebuilt_embeddings.json"


class EmbeddingClient(ABC):
    """Base interface for embedding generators."""

    def __init__(self, dimension: int, provider: str, model_name: str):
        self.dimension = dimension
        self.provider = provider
        self.model_name = model_name

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def signature(self) -> Dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model_name,
            "dimension": self.dimension,
        }


class APIEmbeddingClient(EmbeddingClient):
    """Embedding client implemented via the official OpenAI SDK."""

    def __init__(self, config_name: str = "embed_config.json"):
        config = load_json_config(config_name)
        dimension = int(config.get("dimension", 0))
        if dimension <= 0:
            raise EmbeddingUnavailableError("Embedding dimension must be positive")
        provider_name = config.get("provider", "api")
        model_name = config.get("model") or "unknown-model"
        super().__init__(dimension, provider=provider_name, model_name=model_name)
        self.base_url = config.get("base_url") or config.get("baseurl")
        self.api_key = config.get("api_key")
        self.payload_template = config.get("payload_template", {})
        if not self.base_url or not self.api_key:
            raise EmbeddingUnavailableError("Embedding base_url or API key missing")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed(self, text: str) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.dimension,
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

    def __init__(self, dimension: int = 64, model_name: str = "mock-sha1"):
        super().__init__(dimension, provider="mock", model_name=model_name)

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
        model_name = config.get("model", "mock-sha1")
        return MockEmbeddingClient(dimension=dimension, model_name=model_name)
    return APIEmbeddingClient(config_name=config_name)


def _demo_embedding(prompt: str = "int main() { return 0; }", config_name: str = "embed_config.json") -> None:
    """Quick manual check for embedding provider availability."""

    client = build_embedding_client(config_name=config_name)
    vector = client.embed(prompt)
    print(f"Embedding length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")


def bulk_vectorize_directory(
    source_dir: str | Path,
    output_path: str | Path = DEFAULT_BULK_OUTPUT,
    label: str = "unknown",
    explanation: str | None = None,
    config_name: str = "embed_config.json",
) -> Dict[str, object]:
    """Vectorize every *.c file under ``source_dir`` and dump to JSON.

    The payload records the embedding provider/model/dimension to ensure
    downstream components can rebuild indexes with the correct metadata.
    """

    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist")

    files = sorted(p for p in source_path.rglob("*.c") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .c files found under '{source_dir}'")

    client = build_embedding_client(config_name=config_name)
    signature = client.signature()

    cases: List[Dict[str, object]] = []
    for path in files:
        code = path.read_text(encoding="utf-8")
        vector = client.embed(code).astype(float).tolist()
        relative = path.relative_to(source_path)
        case_id = relative.with_suffix("").as_posix().replace("/", "-").replace("\\", "-")
        cases.append(
            {
                "case_id": case_id,
                "code": code,
                "label": label,
                "explanation": explanation or f"Bulk embedded from {relative.as_posix()}",
                "loops": [],
                "embedding": vector,
                "metadata": {
                    "source_file": str(path),
                    "relative_path": relative.as_posix(),
                    "embedding_provider": signature["provider"],
                    "embedding_model": signature["model"],
                    "embedding_dimension": signature["dimension"],
                },
            }
        )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "embedding_info": signature,
        "source_dir": str(source_path),
        "cases": cases,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "files_processed": len(files),
        "output_path": str(output_path),
        "embedding_info": signature,
    }


def _run_cli() -> None:  # pragma: no cover - manual helper
    parser = argparse.ArgumentParser(description="Embedding utilities")
    parser.add_argument("--config", default="embed_config.json", 
                        help="Path to embedding config JSON, default is \nconfig=<config/embed_config.json>")
    parser.add_argument("--bulk", action="store_true", 
                        help="Run bulk vectorization instead of single demo, no --bulk signal will run a tiny test demo.")
    parser.add_argument("--source-dir", type=Path, 
                        help="Directory that holds .c files for bulk mode")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BULK_OUTPUT,
        help=f"Where to write the generated JSON when --bulk is set, default json now is \n output='{DEFAULT_BULK_OUTPUT}'",
    )
    parser.add_argument("--label", default="NeedReview", 
                        help="Label to assign to bulk cases, \
                            default is \n label='NeedReview',\nthat means developers should\
                            check the vectorized code snipates before using in RAG.")
    parser.add_argument(
        "--explanation",
        default="Bulk embedded via evolve_term.embeddings",
        help="Explanation text recorded for each case",
    )

    args = parser.parse_args()

    if args.bulk:
        if args.source_dir is None:
            parser.error("--source-dir is required when --bulk is enabled")
        summary = bulk_vectorize_directory(
            source_dir=args.source_dir,
            output_path=args.output,
            label=args.label,
            explanation=args.explanation,
            config_name=str(args.config),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"[Embedding Demo] Using {args.config}")
        try:
            _demo_embedding(config_name=str(args.config))
        except EmbeddingUnavailableError as exc:
            print(f"Embedding test failed: {exc}")


if __name__ == "__main__":
    _run_cli()
