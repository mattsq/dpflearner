import copy
import random
import numpy as np
import torch
from typing import Any, List, Tuple, Optional

from joblib import Parallel, delayed

from ..models import get_model
from ..configs.trainer import TrainerConfig
from ..configs.model import ModelConfig
from ..data.binning import BinningScheme
from .trainer import Trainer
from ..ensembles.average import AverageEnsemble

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
    ) -> None:
        self.model_cfgs = model_cfgs
        self.trainer_cfg = trainer_cfg
        self.seed = seed
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.device = device or trainer_cfg.device

    # ------------------------------------------------------------------
    def _fit_one(
        self,
        cfg_idx: int,
        binning: BinningScheme,
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
        trainer = Trainer(tr_cfg)
        ckpt = trainer.fit(model, binning, subset, val_ds)
        return ckpt.model

    # ------------------------------------------------------------------
    def fit(
        self,
        binning: BinningScheme,
        train_ds: torch.utils.data.Dataset,
        val_ds: Optional[torch.utils.data.Dataset] = None,
    ) -> AverageEnsemble:
        data = (train_ds, val_ds)
        if self.n_jobs == 1:
            models = [self._fit_one(i, binning, data) for i in range(len(self.model_cfgs))]
        else:
            models = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_one)(i, binning, data) for i in range(len(self.model_cfgs))
            )
        return AverageEnsemble(models)
