import torch
from outdist.models import register_model, get_model, BaseModel
from outdist.configs.model import ModelConfig

@register_model("dummy")
class DummyModel(BaseModel):
    def __init__(self, n_bins: int = 1):
        super().__init__()
        self.fc = torch.nn.Linear(1, n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(name="dummy", params={"n_bins": 1})


def test_get_model_instantiates_registered_model():
    cfg = ModelConfig(name="dummy", params={"n_bins": 2})
    model = get_model(cfg)
    assert isinstance(model, DummyModel)
    assert model.fc.out_features == 2
