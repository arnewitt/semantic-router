from typing import List, Any
from pydantic import BaseModel, Field
from semantic_router.embedding.vector import EmbeddingVector


class DenseEncoder(BaseModel):
    """
    Base class for all dense encoders.

    A dense encoder is a function that takes an input (such as text, image, or other data)
        and returns  a fixed-size vector of numbers (a dense vector) that represents the input's meaning.

    Dense vectors are useful for tasks like similarity search, clustering, and classification,
        because they capture the semantic content of the input in a numerical form.
    """

    name: str = Field(
        default="dense_encoder",
        description="The name of the dense encoder. This is used to identify the encoder in logs and configurations.",
    )

    def _initialize(self):
        """
        Initializes the encoder. This method should be implemented by subclasses to set up any necessary resources.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def encode(self, inputs: List[Any]) -> List[EmbeddingVector]:
        """
        Encodes a list of inputs into dense representations.

        Args:
            inputs (List[Any]): A list of input objects to be encoded.

        Returns:
            List[EmbeddingVector]: A list of dense vector representations for each input.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
