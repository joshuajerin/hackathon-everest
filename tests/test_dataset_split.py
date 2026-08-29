
from hackathon_everest.dataset import generate_probe_dataset
from hackathon_everest.estimation import train_estimator


def test_training_split_has_no_field_leakage():
    dataset = generate_probe_dataset(episodes=40, fields=10, seed=19)
    result = train_estimator(dataset, n_estimators=8, seed=19)
    train_fields = set(dataset.field_ids[result.train_indices])
    test_fields = set(dataset.field_ids[result.test_indices])
    assert train_fields.isdisjoint(test_fields)
    assert result.metrics["split"]["field_overlap"] == 0
