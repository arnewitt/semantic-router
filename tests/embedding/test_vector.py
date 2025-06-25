import numpy as np
import pytest

from semantic_router.embedding.vector import EmbeddingVector


def _vec(arr):
    return EmbeddingVector(np.asarray(arr, dtype=float))


# ── basic algebra ────────────────────────────────────────────────────────────
def test_add_sub_mul():
    a, b = _vec([1, 2, 3]), _vec([4, 5, 6])

    np.testing.assert_array_equal((a + b).vector, np.array([5, 7, 9]))
    np.testing.assert_array_equal((b - a).vector, np.array([3, 3, 3]))
    np.testing.assert_array_equal((a * 2).vector, np.array([2, 4, 6]))
    np.testing.assert_array_equal((3 * b).vector, np.array([12, 15, 18]))


def test_project():
    v = _vec([1, 0])
    rot = np.array([[0, -1], [1, 0]])  # 90° rotation
    np.testing.assert_array_equal(v.project(rot).vector, np.array([0, 1]))


# ── aggregation & utils ──────────────────────────────────────────────────────
def test_mean_stack():
    vs = [_vec([1, 2]), _vec([3, 4]), _vec([5, 6])]
    mean = EmbeddingVector.mean(vs)
    np.testing.assert_array_equal(mean.vector, np.array([3, 4]))

    stacked = EmbeddingVector.stack(vs)
    np.testing.assert_array_equal(stacked, np.array([[1, 2], [3, 4], [5, 6]]))


def test_is_zero_clip():
    z = _vec([0, 0, 0])
    assert z.is_zero()

    v = _vec([-2, 0.5, 3])
    clipped = v.clip(0, 2).vector
    np.testing.assert_array_equal(clipped, np.array([0, 0.5, 2]))


# ── geometry ─────────────────────────────────────────────────────────────────
def test_normalize_magnitude_dot_cosine():
    v = _vec([3, 4])
    np.testing.assert_allclose(v.magnitude(), 5.0)

    n = v.normalize()
    np.testing.assert_allclose(np.linalg.norm(n), 1.0)

    u = _vec([3, 4])
    assert v.dot(u) == 25
    np.testing.assert_allclose(v.cosine_similarity(u), 1.0)


@pytest.mark.parametrize(
    "a,b,metric,expected",
    [
        ([0, 0], [0, 0], "euclidean", 0.0),
        ([1, 0], [0, 1], "euclidean", np.sqrt(2)),
        ([1, 0], [0, 1], "manhattan", 2.0),
        ([1, 0], [0, 1], "cosine", 1.0),
    ],
)
def test_distance_metrics(a, b, metric, expected):
    d = _vec(a).distance(_vec(b), metric=metric)
    np.testing.assert_allclose(d, expected)


def test_distance_invalid_metric():
    with pytest.raises(ValueError):
        _vec([1, 2]).distance(_vec([3, 4]), metric="chebyshev")
