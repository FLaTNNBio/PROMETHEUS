import torch

from causal_population_ranking.applications.contrastive_ranker import (
    TreatmentConditionedSiameseNetwork,
)


def test_network_outputs_score_and_normalized_projection():
    model = TreatmentConditionedSiameseNetwork(7, 3, hidden_dim=24, projection_dim=8)
    x = torch.randn(11, 7)
    treatment = torch.randint(0, 3, (11,))
    score, projection = model(x, treatment)
    assert score.shape == (11,)
    assert projection.shape == (11, 8)
    norms = torch.linalg.vector_norm(projection, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
