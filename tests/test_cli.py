import torch
import pytest
from outdist.cli import train, evaluate
from outdist.models import BaseModel, register_model, MODEL_REGISTRY
from outdist.configs.model import ModelConfig


@register_model("dummy_cli")
class DummyModel(BaseModel):
    def __init__(self, out_dim: int = 10):
        super().__init__()
        self.fc = torch.nn.Linear(1, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(name="dummy_cli", params={"out_dim": 10})


def test_train_cli_runs(tmp_path, monkeypatch):
    # run training for 1 epoch using dummy dataset
    args = ["--model", "dummy_cli", "--dataset", "dummy", "--epochs", "0", "--batch-size", "2"]
    monkeypatch.chdir(tmp_path)
    train.main(args)


@pytest.mark.parametrize(
    "name",
    [n for n in MODEL_REGISTRY.keys() if n not in {"lincde", "rfcde"}],
)
def test_train_cli_registered_models(name, tmp_path, monkeypatch):
    args = ["--model", name, "--dataset", "dummy", "--epochs", "0", "--batch-size", "2"]
    monkeypatch.chdir(tmp_path)
    train.main(args)


def test_evaluate_cli_runs(tmp_path, monkeypatch):
    args = ["--model", "dummy_cli", "--dataset", "dummy", "--batch-size", "2", "--metrics", "nll"]
    monkeypatch.chdir(tmp_path)
    evaluate.main(args)


@pytest.mark.parametrize(
    "name",
    [
        n
        for n in MODEL_REGISTRY.keys()
        if n not in {"ckde", "quantile_rf", "lincde", "rfcde"}
    ],
)
def test_evaluate_cli_registered_models(name, tmp_path, monkeypatch):
    args = [
        "--model",
        name,
        "--dataset",
        "dummy",
        "--batch-size",
        "2",
        "--metrics",
        "nll",
    ]
    monkeypatch.chdir(tmp_path)
    evaluate.main(args)
