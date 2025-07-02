import pytest
from semantic_router.router.router import SemanticRouter, Route
from semantic_router.encoder.base import DenseEncoder


# Dummy embedding vector and encoder for testing
class DummyEmbeddingVector:
    def __init__(self, text: str):
        self._text = text

    def cosine_similarity(self, other):
        # Highest similarity if the texts match, otherwise zero
        return 1.0 if self._text == other._text else 0.0


class DummyEncoder(DenseEncoder):
    def encode(self, texts):
        return [DummyEmbeddingVector(t) for t in texts]


@pytest.fixture
def encoder():
    return DummyEncoder()


@pytest.fixture
def routes():
    return (
        Route("foo", "first route", ("hello",)),
        Route("bar", "second route", ("world",)),
    )


@pytest.fixture
def router(encoder, routes):
    return SemanticRouter(encoder, routes, top_k=1)


def test_init_no_routes(encoder):
    with pytest.raises(ValueError):
        SemanticRouter(encoder, (), top_k=1)


def test_init_duplicate_names(encoder, routes):
    dup = routes + (Route("foo", "third route", ("x",)),)
    with pytest.raises(ValueError):
        SemanticRouter(encoder, dup)


def test_init_duplicate_descriptions(encoder, routes):
    dup = routes + (Route("baz", "first route", ("x",)),)
    with pytest.raises(ValueError):
        SemanticRouter(encoder, dup)


def test_invalid_encoder_type(routes):
    with pytest.raises(TypeError):
        SemanticRouter(object(), routes)


def test_route_single_query(router):
    result = router.route("hello")[0]
    assert result["route"][0]["route_name"] == "foo"


def test_route_sequence_queries(router):
    res = router.route(["hello", "world"])
    assert {r["query"] for r in res} == {"hello", "world"}


def test_top_k_enforced(encoder, routes):
    extra = routes + (
        Route("baz", "third route", ("foo",)),
        Route("qux", "fourth route", ("bar",)),
    )
    router = SemanticRouter(encoder, extra, top_k=2)
    assert len(router.route("foo")[0]["route"]) == 2


def test_invalid_query_type(router):
    with pytest.raises(TypeError):
        router.route(123)  # type: ignore[arg-type]
