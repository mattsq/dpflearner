from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..configs import trainer as trainer_cfg
from ..models.base import BaseModel
from ..metrics import METRICS_REGISTRY
from ..losses import cross_entropy, evidential_loss
from ..data import binning as binning_scheme


@dataclass
class Checkpoint:
    """Simple training checkpoint returned by :meth:`Trainer.fit`."""

    model: BaseModel
    epoch: int
    metrics: Optional[Dict[str, float]] = None


class Trainer:
    """Minimal training loop implementing the design described in ``prompt.md``."""

    def __init__(self, cfg: trainer_cfg.TrainerConfig, *, loss_fn= cross_entropy) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.loss_fn = loss_fn

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        model: BaseModel,
        binning: binning_scheme.BinningScheme,
        train_ds: Dataset,
        val_ds: Optional[Dataset] = None,
    ) -> Checkpoint:
        """Train ``model`` on ``train_ds`` and optionally ``val_ds``."""

        # Fit the binning strategy before training and handle models that
        # require a separate ``fit`` step (e.g. non-neural estimators).
        features = []
        targets = []
        for x, y in DataLoader(train_ds, batch_size=self.cfg.batch_size):
            features.append(x)
            targets.append(y)
        if targets:
            y_all = torch.cat(targets)
            binning.fit(y_all)
            if hasattr(model, "fit"):
                x_all = torch.cat(features)
                model.fit(x_all.to(self.device), y_all.to(self.device))

        model.to(self.device)
        params = list(model.parameters())
        optimizer = (
            torch.optim.Adam(params, lr=self.cfg.lr) if params else None
        )
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

                if optimizer is not None:
                    optimizer.zero_grad()
                out = model(x)
                logits = out
                if isinstance(out, dict):
                    logits = out.get("logits", out.get("probs").log())
                if self.loss_fn is evidential_loss:
                    loss = self.loss_fn(out["alpha"], y)
                else:
                    loss = self.loss_fn(logits, y)
                if optimizer is not None:
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
                out = model(x)
                logits = out
                if isinstance(out, dict):
                    logits = out.get("logits", out.get("probs").log())
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
                out = model(x)
                logits = out
                if isinstance(out, dict):
                    logits = out.get("logits", out.get("probs").log())
                if self.loss_fn is evidential_loss:
                    _ = self.loss_fn(out["alpha"], y)
                else:
                    _ = self.loss_fn(logits, y)



