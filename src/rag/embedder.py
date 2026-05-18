import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")

from sentence_transformers import SentenceTransformer

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model_name
        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self._dimension: int | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> list[float]:
        embedding = self._model.encode(query, normalize_embeddings=True)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._dimension
