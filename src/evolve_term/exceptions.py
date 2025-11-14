"""Custom exceptions for the EvolveTerm pipeline."""


class EvolveTermError(RuntimeError):
    """Base class for domain-specific runtime errors."""


class LLMUnavailableError(EvolveTermError):
    """Raised when the LLM backend cannot serve a request."""


class EmbeddingUnavailableError(EvolveTermError):
    """Raised when the embedding backend cannot serve a request."""


class PromptNotFoundError(EvolveTermError):
    """Raised when an expected prompt template cannot be loaded."""


class KnowledgeBaseError(EvolveTermError):
    """Raised when persisting or loading the knowledge base fails."""


class IndexNotReadyError(EvolveTermError):
    """Raised when a retrieval is requested but the HNSW index is missing."""
