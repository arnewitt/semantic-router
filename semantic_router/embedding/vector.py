import json
from typing import Iterable, Union
from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingVector:
    """
    Base class for embedding vectors of single objects.

    Vector embeddings are numerical representations of data points, like words,
        images, or other complex data, that capture their semantic meaning and
        relationships within a multi-dimensional space.
    """

    vector: np.ndarray

    def __add__(self, other: "EmbeddingVector") -> "EmbeddingVector":
        """
        Adds two EmbeddingVector objects together.

        Args:
            other (EmbeddingVector): The other EmbeddingVector object to add.
        Returns:
            EmbeddingVector: A new EmbeddingVector object with the sum of the vectors.
        """

        return EmbeddingVector(self.vector + other.vector)

    def __sub__(self, other: "EmbeddingVector") -> "EmbeddingVector":
        """
        Subtracts another EmbeddingVector from this one.

        Args:
            other (EmbeddingVector): The EmbeddingVector to subtract.

        Returns:
            EmbeddingVector: A new EmbeddingVector representing the difference.
        """
        return EmbeddingVector(self.vector - other.vector)

    def __mul__(self, scalar: Union[int, float]) -> "EmbeddingVector":
        """
        Multiplies the embedding vector by a scalar value.

        Args:
            scalar (Union[int, float]): The scalar value to multiply the vector by.

        Returns:
            EmbeddingVector: A new EmbeddingVector instance with the scaled vector.
        """
        return EmbeddingVector(self.vector * scalar)

    def __eq__(self, other: object) -> bool:
        """
        Checks if two EmbeddingVector objects are equal.

        Args:
            other: The other object to compare to.

        Returns:
            True if the two objects are equal, False otherwise.
        """
        if not isinstance(other, EmbeddingVector):
            return NotImplemented
        return np.array_equal(self.vector, other.vector)

    __rmul__ = __mul__

    def project(self, matrix: np.ndarray) -> "EmbeddingVector":
        """
        Projects the embedding vector onto a new space using a transformation matrix.

        This method applies a linear transformation to the embedding vector, effectively
        changing its representation to a new coordinate system. This can be useful for
        dimensionality reduction, feature extraction, or aligning different embedding spaces.

        Args:
            matrix (np.ndarray): A 2D numpy array representing the transformation matrix.
            The number of columns in the matrix must match the dimensionality of the
            embedding vector.

        Returns:
            EmbeddingVector: A new EmbeddingVector object representing the projected vector.
        """
        return EmbeddingVector(matrix @ self.vector)

    def to_list(self) -> list:
        """
        Converts the vector to a list.

        Returns:
            list: The vector as a list.
        """
        return self.vector.tolist()

    @classmethod
    def from_list(cls, lst: list) -> "EmbeddingVector":
        """
        Converts a list to an EmbeddingVector.

        Args:
            lst (list): A list of numbers.

        Returns:
            EmbeddingVector: An EmbeddingVector created from the list.
        """
        return cls(np.asarray(lst, dtype=float))

    @classmethod
    def mean(cls, vectors: Iterable["EmbeddingVector"]) -> "EmbeddingVector":
        vecs = [v.vector for v in vectors]
        return cls(np.mean(vecs, axis=0))

    @classmethod
    def stack(cls, vectors: Iterable["EmbeddingVector"]) -> np.ndarray:
        return np.vstack([v.vector for v in vectors])

    def is_zero(self, tol: float = 1e-12) -> bool:
        return np.linalg.norm(self.vector) < tol

    def clip(self, min_val: float, max_val: float) -> "EmbeddingVector":
        return EmbeddingVector(np.clip(self.vector, min_val, max_val))

    def normalize(self) -> np.ndarray:
        """
        Returns a normalized (unit length) version of the vector.
        """
        norm = np.linalg.norm(self.vector)
        if norm == 0:
            return self.vector
        return self.vector / norm

    def magnitude(self) -> float:
        """
        Calculates the magnitude (or Euclidean norm) of the vector.

        Returns:
            float: The magnitude of the vector.
        """
        return float(np.linalg.norm(self.vector))

    def dot(self, other: "EmbeddingVector") -> float:
        """
        Compute the dot product between two embedding vectors.

        Args:
            other (EmbeddingVector): The other embedding vector.

        Returns:
            float: The dot product between the two vectors.
        """
        return float(np.dot(self.vector, other.vector))

    def cosine_similarity(self, other: "EmbeddingVector") -> float:
        """
        Calculates the cosine similarity between two EmbeddingVector objects.

        Args:
            other (EmbeddingVector): The other EmbeddingVector to compare to.

        Returns:
            float: The cosine similarity between the two vectors, a value between -1 and 1.
                Returns 0 if either vector has a magnitude of zero to avoid division by zero.
        """
        return self.dot(other) / (self.magnitude() * other.magnitude() + 1e-12)

    def distance(self, other: "EmbeddingVector", metric: str = "euclidean") -> float:
        """
        Calculates the distance between this embedding vector and another.

        Args:
            other: The other EmbeddingVector to calculate the distance to.
            metric: The distance metric to use. Options are "euclidean", "cosine", and "manhattan".
            Defaults to "euclidean".

        Returns:
            The distance between the two vectors as a float.

        Raises:
            ValueError: If an unsupported metric is specified.
        """

        if metric == "euclidean":
            return float(np.linalg.norm(self.vector - other.vector))
        if metric == "cosine":
            return 1.0 - self.cosine_similarity(other)
        if metric == "manhattan":
            return float(np.sum(np.abs(self.vector - other.vector)))
        raise ValueError(f"Unsupported metric '{metric}'")
