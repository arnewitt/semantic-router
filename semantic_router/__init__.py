from typing import Optional, List, Any
from semantic_router.encoder.base import DenseEncoder
from semantic_router.embedding.vector import EmbeddingVector
__all__ = [
    "DenseEncoder",
]

class AutoEncoder:

    type: str
    name: Optional[str] = None
    model: DenseEncoder

    def __init__(self, type: str, name: Optional[str] = None):
        pass

    def __call__(self, objs: List[Any]) -> List[EmbeddingVector]:
        """
        Encodes a list of objects into a list of vectors.

        Args:
            objs (List[Any]): A list of objects to encode.

        Returns:
            List[EmbeddingVector]: A list of encoded vectors.
        """
        pass