class PRDGeneratorError(Exception):
    """Base exception for the PRD generator."""


class RAGError(PRDGeneratorError):
    """Raised when RAG operations fail."""


class LLMError(PRDGeneratorError):
    """Raised when LLM API calls fail."""


class WorkflowError(PRDGeneratorError):
    """Raised when the LangGraph workflow encounters an error."""


class ValidationError(PRDGeneratorError):
    """Raised when PRD JSON output fails schema validation."""


class ConfigError(PRDGeneratorError):
    """Raised when required configuration is missing."""
