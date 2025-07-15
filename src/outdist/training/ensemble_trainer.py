import copy
import random
import numpy as np
import torch
from typing import Any, List, Tuple, Optional, Callable

from joblib import Parallel, delayed

from ..models import get_model
from ..configs.trainer import TrainerConfig
from ..configs.model import ModelConfig
from ..data.binning import BinningScheme, LearnableBinningScheme
from .trainer import Trainer
from .logger import TrainingLogger
from ..ensembles.average import AverageEnsemble
from ..ensembles.stacked import StackedEnsemble, learn_weights
from ..data import binning as binning_scheme


def _rebin_pc(
    old_probs: torch.Tensor, old_edges: torch.Tensor, new_edges: torch.Tensor
) -> torch.Tensor:
    """Redistribute probability mass from ``old_edges`` onto ``new_edges``."""

    density = old_probs / (old_edges[1:] - old_edges[:-1])
    p_new = []
    for left, right in zip(new_edges[:-1], new_edges[1:]):
        mask = (old_edges[:-1] < right) & (old_edges[1:] > left)
        if mask.any():
            overlap_left = torch.maximum(old_edges[:-1][mask], left)
            overlap_right = torch.minimum(old_edges[1:][mask], right)
            overlap = overlap_right - overlap_left
            val = (density[..., mask] * overlap).sum(dim=-1)
        else:
            val = torch.zeros(old_probs.shape[:-1], device=old_probs.device)
        p_new.append(val)
    return torch.stack(p_new, dim=-1)


class RebinAdapter(torch.nn.Module):
    """Adapt a model's logits to a common grid."""

    def __init__(self, model: Any, edges_star: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("edges_star", edges_star)
        self.binner = binning_scheme.BinningScheme(edges=edges_star.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.model(x)
        if hasattr(self.model, "binner"):
            old_edges = self.model.binner.edges.to(x)
            probs = logits.softmax(dim=-1)
            rebinned = _rebin_pc(probs, old_edges, self.edges_star.to(x))
            eps = torch.finfo(rebinned.dtype).tiny
            return (rebinned + eps).log()
        return logits

    def log_prob(self, *args: Any, **kw: Any) -> torch.Tensor:  # type: ignore[override]
        if hasattr(self.model, "log_prob"):
            return self.model.log_prob(*args, **kw)
        raise AttributeError("Underlying model has no log_prob method")

    def sample(self, *args: Any, **kw: Any) -> torch.Tensor:  # type: ignore[override]
        if hasattr(self.model, "sample"):
            return self.model.sample(*args, **kw)
        raise AttributeError("Underlying model has no sample method")


class EnsembleTrainer:
    """Bagging / deep ensemble trainer."""

    def __init__(
        self,
        model_cfgs: List[ModelConfig | str | dict],
        trainer_cfg: TrainerConfig,
        *,
        seed: int = 0,
        bootstrap: bool = True,
        n_jobs: int = 1,
        device: str | None = None,
        stack: bool = False,
        stack_cfg: dict | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        self.model_cfgs = model_cfgs
        self.trainer_cfg = trainer_cfg
        self.seed = seed
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.device = device or trainer_cfg.device
        self.stack = stack
        self.stack_cfg = stack_cfg or {}
        self.logger = logger

    # ------------------------------------------------------------------
    def _align_binning(self, models: List[Any]) -> List[Any]:
        """Ensure all models operate on the same bin edges."""

        edges_list: List[torch.Tensor] = []
        learnable = False
        for m in models:
            binner = getattr(m, "binner", None)
            if isinstance(binner, BinningScheme):
                edges_list.append(binner.edges.detach())
                if isinstance(binner, LearnableBinningScheme):
                    learnable = True
        if not edges_list:
            return models
        ref = edges_list[0]
        identical = all(torch.allclose(ref, e) for e in edges_list[1:])
        if identical and not learnable:
            return models
        edges_common = torch.stack(edges_list).mean(0)
        edges_common = torch.sort(edges_common).values
        eps = torch.finfo(edges_common.dtype).eps
        edges_common[1:] = torch.maximum(edges_common[1:], edges_common[:-1] + eps)
        adapted = []
        for m in models:
            binner = getattr(m, "binner", None)
            if isinstance(binner, BinningScheme):
                adapted.append(RebinAdapter(m, edges_common))
            else:
                adapted.append(m)
        return adapted

    # ------------------------------------------------------------------
    def _fit_one(
        self,
        cfg_idx: int,
        binning: BinningScheme | Callable[[torch.Tensor], BinningScheme],
        data: Tuple[torch.utils.data.Dataset, Optional[torch.utils.data.Dataset]],
    ):
        train_ds, val_ds = data
        cfg = copy.deepcopy(self.model_cfgs[cfg_idx])
        torch.manual_seed(self.seed + cfg_idx)
        np.random.seed(self.seed + cfg_idx)
        random.seed(self.seed + cfg_idx)

        if isinstance(cfg, str):
            model = get_model(ModelConfig(name=cfg))
        elif isinstance(cfg, dict) and "name" in cfg:
            model = get_model(ModelConfig(**cfg))
        else:
            model = get_model(cfg)  # type: ignore[arg-type]

        if self.bootstrap:
            idx = torch.randint(0, len(train_ds), (len(train_ds),))
            subset = torch.utils.data.Subset(train_ds, idx.tolist())
        else:
            subset = train_ds

        tr_cfg = copy.deepcopy(self.trainer_cfg)
        tr_cfg.device = self.device
        trainer = Trainer(tr_cfg, logger=self.logger)
        ckpt = trainer.fit(model, binning, subset, val_ds)
        return ckpt.model

    # ------------------------------------------------------------------
    def fit(
        self,
        binning: BinningScheme | Callable[[torch.Tensor], BinningScheme],
        train_ds: torch.utils.data.Dataset,
        val_ds: Optional[torch.utils.data.Dataset] = None,
    ) -> AverageEnsemble:
        data = (train_ds, val_ds)
        if self.bootstrap and isinstance(binning, LearnableBinningScheme):
            raise ValueError(
                "Bootstrapping is incompatible with LearnableBinningScheme"
            )
        if self.n_jobs == 1:
            models = [
                self._fit_one(i, binning, data) for i in range(len(self.model_cfgs))
            ]
        else:
            models = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_one)(i, binning, data)
                for i in range(len(self.model_cfgs))
            )

        models = self._align_binning(models)

        if self.stack:
            assert val_ds is not None, "stacking needs a validation split"
            # Gather validation arrays
            loader = torch.utils.data.DataLoader(val_ds, batch_size=len(val_ds))
            X_val_list: List[torch.Tensor] = []
            y_val_list: List[torch.Tensor] = []
            for xb, yb in loader:
                X_val_list.append(xb)
                y_val_list.append(yb)
            X_val = torch.cat(X_val_list)
            y_val = torch.cat(y_val_list)
            binner = getattr(models[0], "binner", None)
            if binner is not None:
                y_val = binner.to_index(y_val)
            y_val = y_val.long()
            w = learn_weights(
                models,
                X_val.numpy(),
                y_val.numpy(),
                l2=self.stack_cfg.get("l2", 1e-3),
            )
            return StackedEnsemble(models, w)
        else:
            return AverageEnsemble(models)
