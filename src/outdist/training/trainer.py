from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..configs import trainer as trainer_cfg
from ..models.base import BaseModel
from ..metrics import METRICS_REGISTRY


@dataclass
class Checkpoint:
    """Simple training checkpoint returned by :meth:`Trainer.fit`."""

    model: BaseModel
    epoch: int
    metrics: Optional[Dict[str, float]] = None


class Trainer:
    """Minimal training loop implementing the design described in ``prompt.md``."""

    def __init__(self, cfg: trainer_cfg.TrainerConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        model: BaseModel,
        train_ds: Dataset,
        val_ds: Optional[Dataset] = None,
    ) -> Checkpoint:
        """Train ``model`` on ``train_ds`` and optionally ``val_ds``."""

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader = (
            DataLoader(val_ds, batch_size=self.cfg.batch_size)
            if val_ds is not None
            else None
        )

        for epoch in range(1, self.cfg.max_epochs + 1):
            model.train()
            for batch in train_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                logits = model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                self._run_validation(model, val_loader)

        return Checkpoint(model=model, epoch=self.cfg.max_epochs)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        model: BaseModel,
        test_ds: Dataset,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate ``model`` on ``test_ds`` using the requested ``metrics``."""

        model.to(self.device)
        loader = DataLoader(test_ds, batch_size=self.cfg.batch_size)
        results: Dict[str, float] = {}

        preds: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(self.device)
                logits = model(x)
                preds.append(logits.cpu())
                targets.append(y)

        y_pred = torch.cat(preds)
        y_true = torch.cat(targets)

        if not metrics:
            metrics = ["nll"]

        for name in metrics:
            metric_fn = METRICS_REGISTRY[name]
            value = metric_fn(y_pred, y_true)
            results[name] = float(value.item())

        return results

    # ------------------------------------------------------------------
    def _run_validation(self, model: BaseModel, loader: DataLoader) -> None:
        """Run one validation epoch."""

        model.eval()
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)
                _ = self.criterion(logits, y)



