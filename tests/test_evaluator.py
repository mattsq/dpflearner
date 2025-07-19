import torch
from outdist.evaluation.evaluator import cross_validate, CVFoldResult, _split_dataset
from outdist.configs.model import ModelConfig
from outdist.models import register_model, BaseModel
from outdist.data.datasets import make_dataset

@register_model("dummy_eval")
class DummyModel(BaseModel):
    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.fc = torch.nn.Linear(1, n_bins)
        self.binner = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(name="dummy_eval", params={"n_bins": 10})


def test_cross_validate_runs():
    dataset, _, _ = make_dataset("synthetic", n_samples=20)

    def model_factory():
        return DummyModel(n_bins=10)

    results = cross_validate(model_factory, dataset, k_folds=2, metrics=["nll"], trainer_cfg=None)
    assert len(results) == 2
    assert all(isinstance(r, CVFoldResult) for r in results)
    assert all("nll" in r.metrics for r in results)


def test_cross_validate_default_metrics():
    """cross_validate should request default metrics when none provided."""
    dataset, _, _ = make_dataset("synthetic", n_samples=20)

    def model_factory():
        return DummyModel(n_bins=10)

    results = cross_validate(model_factory, dataset, k_folds=2)
    assert all({"nll", "crps"}.issubset(r.metrics.keys()) for r in results)


def test_split_dataset_even_and_uneven():
    """_split_dataset should partition all indices exactly once."""
    tensor_ds = torch.utils.data.TensorDataset(torch.arange(10))

    # Even split into 5 folds -> each val set has 2 items
    folds = list(_split_dataset(tensor_ds, 5))
    assert len(folds) == 5
    lengths = [len(val) for _, val in folds]
    assert lengths == [2, 2, 2, 2, 2]

    # Collect validation indices and ensure coverage
    all_val_idx = sorted(i for _, val in folds for i in val.indices)
    assert all_val_idx == list(range(10))

    # Uneven split into 3 folds -> last fold gets remainder
    folds = list(_split_dataset(tensor_ds, 3))
    lengths = [len(val) for _, val in folds]
    assert lengths == [3, 3, 4]
    all_val_idx = sorted(i for _, val in folds for i in val.indices)
    assert all_val_idx == list(range(10))
