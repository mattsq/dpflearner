import torch
from outdist.evaluation.evaluator import cross_validate, CVFoldResult
from outdist.configs.model import ModelConfig
from outdist.models import register_model, BaseModel
from outdist.data.datasets import make_dataset

@register_model("dummy_eval")
class DummyModel(BaseModel):
    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.fc = torch.nn.Linear(1, n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(name="dummy_eval", params={"n_bins": 10})


def test_cross_validate_runs():
    dataset, _, _ = make_dataset("dummy", n_samples=20)

    def model_factory():
        return DummyModel(n_bins=10)

    results = cross_validate(model_factory, dataset, k_folds=2, metrics=["nll"], trainer_cfg=None)
    assert len(results) == 2
    assert all(isinstance(r, CVFoldResult) for r in results)
    assert all("nll" in r.metrics for r in results)
