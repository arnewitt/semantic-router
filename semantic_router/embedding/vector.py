
from dataclasses import dataclass
import numpy as np

@dataclass
class EmbeddingVector:
    """
    Base class for embedding vectors.

    Vector embeddings are numerical representations of data points, like words, 
        images, or other complex data, that capture their semantic meaning and 
        relationships within a multi-dimensional space.
    """
    vector: np.ndarray

    def normalize(self) -> np.ndarray:
        """
        Returns a normalized (unit length) version of the vector.
        """
        norm = np.linalg.norm(self.vector)
        if norm == 0:
            return self.vector
        return self.vector / norm