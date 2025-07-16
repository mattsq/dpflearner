import torch
import pytest
from outdist import cli
from outdist.models import BaseModel, register_model, MODEL_REGISTRY
from outdist.configs.model import ModelConfig
from outdist.data import binning as binning_scheme


@register_model("dummy_cli")
class DummyModel(BaseModel):
    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.fc = torch.nn.Linear(1, n_bins)
        edges = torch.linspace(0.0, 1.0, n_bins + 1)
        self.binner = binning_scheme.BinningScheme(edges=edges)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(name="dummy_cli", params={"n_bins": 10})


def test_train_cli_runs(tmp_path, monkeypatch):
    # run training for 1 epoch using synthetic dataset
    args = ["--model", "dummy_cli", "--dataset", "synthetic", "--epochs", "0", "--batch-size", "2"]
    monkeypatch.chdir(tmp_path)
    cli.main(args)


@pytest.mark.parametrize(
    "name",
    [n for n in MODEL_REGISTRY.keys() if n not in {"lincde", "rfcde"}],
)
def test_train_cli_registered_models(name, tmp_path, monkeypatch):
    args = ["--model", name, "--dataset", "synthetic", "--epochs", "0", "--batch-size", "2"]
    monkeypatch.chdir(tmp_path)
    cli.main(args)

