from typing import List, Union
from semantic_router.encoder.base import DenseEncoder
from sentence_transformers import SentenceTransformer

from semantic_router.embedding.vector import EmbeddingVector


class HuggingFaceEncoder(DenseEncoder):
    """
    A dense encoder that uses Hugging Face models to encode input texts into dense vectors.

    This class extends the DenseEncoder base class and implements the encode method
    using a Hugging Face model for encoding inputs.
    """

    def __init__(
        self,
        name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initializes the HuggingFaceEncoder with a specified model name.

        Args:
            name (str): The name of the Hugging Face model to use for encoding.
                        Defaults to "sentence-transformers/all-MiniLM-L6-v2".
        """

        super().__init__(name=name)
        self._model = self._initialize()

    def _initialize(self):
        """
        Initializes the SentenceTransformer model.

        Returns:
            SentenceTransformer: The initialized SentenceTransformer model.

        Raises:
            RuntimeError: If the model could not be loaded.
        """
        try:
            return SentenceTransformer(model_name_or_path=self.name)
        except Exception as e:
            raise RuntimeError(f"Could not load model '{self.name}': {e}") from e

    def encode(self, inputs: Union[str, List[str]]) -> List[EmbeddingVector]:
        """
        Encodes a list of strings into a list of embedding vectors using the Hugging Face transformer model.

        Args:
            inputs (Union[str, List[str]]): A string or a list of strings to encode.

        Returns:
            List[EmbeddingVector]: A list of embedding vectors, one for each input string.

        Raises:
            TypeError: If inputs is not a string or a list of strings.
            TypeError: If any element in inputs is not a string.
            RuntimeError: If the encoding fails.
        """

        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list):
            raise TypeError(
                f"inputs must be a list of strings or a single string, got {type(inputs).__name__}"
            )
        if not all(isinstance(x, str) for x in inputs):
            raise TypeError("all elements in inputs must be of type str")
        try:
            vectors = self._model.encode(inputs, convert_to_numpy=True)
            return [EmbeddingVector(vector=vector) for vector in vectors]

        except Exception as e:
            raise RuntimeError(f"Encoding failed: {e}") from e
