import torch
import pytest

from zennit.layer import KMeans


@pytest.mark.parametrize(
    "input, centroids, expected",
    [(torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0]]),
      torch.tensor([[1.0, 1.0], [5.0, 5.0]]), torch.tensor([0, 1, 1]))])
def test_kmeans(input, centroids, expected):
    model = KMeans(centroids)
    result = model(input)
    assert torch.equal(result, expected)


def test_kmeans_parameter():
    centroids = torch.tensor([[1.0, 1.0], [5.0, 5.0]])
    model = KMeans(centroids)
    assert torch.equal(model.centroids, centroids)
    assert isinstance(model.centroids, torch.nn.Parameter)
