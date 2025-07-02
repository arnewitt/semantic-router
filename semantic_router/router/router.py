import heapq
from typing import List, Union
from dataclasses import dataclass, field
from collections.abc import Sequence
from semantic_router.encoder.base import DenseEncoder
from semantic_router.embedding.vector import EmbeddingVector


@dataclass(slots=True, frozen=True)
class Route:
    """Initializes a Route object with a name and description."""

    name: str
    description: str
    examples: Sequence[str] = field(default_factory=tuple)


class SemanticRouter:
    """A semantic router that routes queries to the most relevant defined routes. Uses cosine similarity."""

    def __init__(
        self,
        encoder: DenseEncoder,
        routes: Sequence[Route],
        top_k: int = 5,
    ):
        self.encoder = encoder
        self.routes: tuple[Route, ...] = tuple(routes)
        self.top_k = top_k
        self.router_map: dict[str, dict[str, Union[Route, List[EmbeddingVector]]]] = {}

        self._initialize_router_map()

    def _initialize_router_map(self):
        """
        Initializes the router map by encoding route examples and storing them.

        Raises:
            ValueError: If no routes are provided or if duplicate route names are found.
            TypeError: If the encoder is not an instance of DenseEncoder.
        """
        # Return error if no routes are provided or if duplicate names are found
        if len(self.routes) == 0:
            raise ValueError("No routes provided for the SemanticRouter.")

        # Check for duplicate route names
        route_names = [route.name for route in self.routes]
        if len(route_names) != len(set(route_names)):
            raise ValueError(
                "Duplicate route names found. Each route must have a unique name."
            )

        # Check for duplicate route descriptions
        route_descriptions = [route.description for route in self.routes]
        if len(route_descriptions) != len(set(route_descriptions)):
            raise ValueError(
                "Duplicate route descriptions found. Each route must have a unique description."
            )

        # Check if the encoder is initialized correctly
        if not isinstance(self.encoder, DenseEncoder):
            raise TypeError("Encoder must be an instance of DenseEncoder.")

        # Encode each route's examples and store them in the router map
        for route in self.routes:
            encoded_examples = self._encode_query(route.examples)
            self.router_map[route.name] = {
                "route": route,
                "encoded_examples": encoded_examples,
            }

    def _encode_query(self, queries: Sequence[str]) -> List[EmbeddingVector]:
        """
        Encodes a sequence of queries into a list of embedding vectors.

        Args:
            queries (Sequence[str]): A sequence of string queries to encode.

        Returns:
            List[EmbeddingVector]: A list of embedding vectors, one for each query.
        """
        return self.encoder.encode(list(queries))

    def _find_top_k_closest_routes(self, queries: Sequence[str]) -> List[dict]:
        """
        Finds the top k closest routes to the given queries based on cosine similarity.

        Args:
            queries (Sequence[str]): A list of query strings.

        Returns:
            List[dict]: A list of dictionaries, each containing the rank, route name,
            and cosine similarity score for the top k routes. The list is sorted
            by rank, with the highest score (closest route) having rank 0.
        """

        encoded_queries = self._encode_query(queries)
        best_scores: dict[str, float] = {}

        for encoded_query in encoded_queries:
            for route_name, data in self.router_map.items():
                score = max(
                    encoded_query.cosine_similarity(example)
                    for example in data["encoded_examples"]
                )
                best_scores[route_name] = max(score, best_scores.get(route_name, 0.0))

        # Heapq is used to efficiently find the top k elements.
        # Heapq is more efficient than sorting the entire dataset.
        top_routes = heapq.nlargest(
            self.top_k, best_scores.items(), key=lambda item: item[1]
        )
        return [
            {"rank": idx, "route_name": name, "cosine_similarity": score}
            for idx, (name, score) in enumerate(top_routes)
        ]

    def route(self, queries: Union[str, Sequence[str]]) -> List[List[Route]]:
        """
        Routes a list of queries to the most relevant defined routes.

        Args:
            queries (Union[str, Sequence[str]]): A single query string or a sequence of query strings.
        Returns:
            List[List[Route]]: A list of lists, where each inner list contains the `Route` objects that
            are most relevant to the corresponding query.
            Returns a list of dictionaries, where each dictionary contains the original query and a list of `Route` objects.
        Raises:
            TypeError: If `queries` is not a string or a sequence of strings.
        """

        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, Sequence):
            raise TypeError("Queries must be a string or a sequence of strings.")
        return [
            {"query": query, "route": self._find_top_k_closest_routes([query])}
            for query in queries
        ]
