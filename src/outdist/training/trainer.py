from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..configs import trainer as trainer_cfg
from ..configs.calibration import CalibratorConfig
from ..models.base import BaseModel
from ..metrics import METRICS_REGISTRY
from ..losses import cross_entropy, evidential_loss
from ..data import binning as binning_scheme
from ..calibration import get_calibrator, BaseCalibrator
from .logger import TrainingLogger


@dataclass
class Checkpoint:
    """Simple training checkpoint returned by :meth:`Trainer.fit`."""

    model: BaseModel
    epoch: int
    metrics: Optional[Dict[str, float]] = None
    calibrator: Optional[BaseCalibrator] = None


class Trainer:
    """Minimal training loop implementing the design described in ``prompt.md``."""

    def __init__(
        self,
        cfg: trainer_cfg.TrainerConfig,
        calibrator_cfg: CalibratorConfig | None = None,
        *,
        loss_fn=cross_entropy,
        logger: TrainingLogger | None = None,
        bootstrap_bins: bool = False,
        n_bin_bootstraps: int = 10,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.loss_fn = loss_fn
        self.calibrator_cfg = calibrator_cfg
        self.calibrator: Optional[BaseCalibrator] = None
        self.logger = logger
        self.bootstrap_bins = bootstrap_bins
        self.n_bin_bootstraps = n_bin_bootstraps

    # ------------------------------------------------------------------
    def _to_index(self, model: BaseModel, y: torch.Tensor) -> torch.Tensor:
        """Convert ``y`` to bin indices if ``model`` defines a binner."""

        if hasattr(model, "binner"):
            return model.binner.to_index(y)
        return y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        model: BaseModel,
        binning: binning_scheme.BinningScheme
        | Callable[[torch.Tensor], binning_scheme.BinningScheme],
        train_ds: Dataset,
        val_ds: Optional[Dataset] = None,
        *,
        conformal_cfg: dict | None = None,
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
            if self.bootstrap_bins:
                def strategy(sample: torch.Tensor) -> torch.Tensor:
                    out = binning(sample) if callable(binning) else binning.fit(sample)
                    if isinstance(out, binning_scheme.LearnableBinningScheme):
                        raise ValueError("Bootstrapping incompatible with LearnableBinningScheme")
                    if isinstance(out, binning_scheme.BinningScheme):
                        return out.edges
                    if isinstance(out, torch.Tensor):
                        return out
                    raise TypeError("Binning callable must return edges or a BinningScheme")

                binning = binning_scheme.bootstrap(
                    strategy, y_all, n_bootstrap=self.n_bin_bootstraps
                )
            else:
                if not isinstance(binning, binning_scheme.BinningScheme) and callable(binning):
                    binning = binning(y_all)
                else:
                    binning.fit(y_all)  # type: ignore[call-arg]

            if isinstance(binning, binning_scheme.LearnableBinningScheme) and not isinstance(model, nn.Module):
                raise TypeError("Learnable bins require model to be an nn.Module")
            if hasattr(model, "binner"):
                model.binner = binning
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
            if self.logger is not None:
                self.logger.start_epoch(epoch, len(train_loader))
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                if optimizer is not None:
                    optimizer.zero_grad()
                if hasattr(model, "quantile_loss"):
                    loss = model.quantile_loss(x, y.unsqueeze(1))
                elif self.loss_fn is None and hasattr(model, "imm_loss"):
                    loss = model.imm_loss(x, y)
                elif self.loss_fn is None and hasattr(model, "dsm_loss"):
                    loss = model.dsm_loss(x, y)
                elif hasattr(model, "mf_loss"):
                    loss = model.mf_loss(x, y)
                else:
                    out = model(x)
                    logits = out
                    if isinstance(out, dict):
                        logits = out.get("logits", out.get("probs").log())
                    if self.loss_fn is evidential_loss:
                        targets = self._to_index(model, y)
                        loss = self.loss_fn(out["alpha"], targets)
                    else:
                        targets = self._to_index(model, y)
                        loss = self.loss_fn(logits, targets)
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()
                if self.logger is not None:
                    self.logger.log_batch(batch_idx, float(loss.item()))

            if val_loader is not None:
                self._run_validation(model, val_loader)
            if self.logger is not None:
                self.logger.end_epoch(epoch)

        # Fit calibrator on validation data if requested
        if self.calibrator_cfg is not None and val_loader is not None:
            probs_list: List[torch.Tensor] = []
            labels_list: List[torch.Tensor] = []
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x = x.to(self.device)
                    if (
                        hasattr(model, "imm_loss") or hasattr(model, "mf_loss")
                    ) and hasattr(model, "predict_logits"):
                        logits = model.predict_logits(x)
                    else:
                        out = model(x)
                        logits = out
                        if isinstance(out, dict):
                            logits = out.get("logits", out.get("probs").log())
                    probs = logits.softmax(dim=-1).cpu()
                    probs_list.append(probs)
                    labels_list.append(self._to_index(model, y))
            val_probs = torch.cat(probs_list)
            val_labels = torch.cat(labels_list)
            self.calibrator = get_calibrator(
                self.calibrator_cfg, n_bins=val_probs.size(1)
            )
            self.calibrator.fit(val_probs, val_labels)

        # Optional conformalisation on a calibration split
        if conformal_cfg is not None:
            from outdist.conformal import CHCDSConformal
            split = conformal_cfg.get("split", 0.2)
            alpha = conformal_cfg.get("alpha", 0.1)
            tau = conformal_cfg.get("tau", 1 - alpha)
            mode = conformal_cfg.get("mode", "sub")

            if val_ds is None:
                m = int(len(train_ds) * split)
                idx = list(range(len(train_ds) - m, len(train_ds)))
                cal_loader = DataLoader(
                    torch.utils.data.Subset(train_ds, idx),
                    batch_size=self.cfg.batch_size,
                )
            else:
                cal_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size)

            X_c: List[torch.Tensor] = []
            y_c: List[torch.Tensor] = []
            for xb, yb in cal_loader:
                X_c.append(xb)
                y_c.append(yb)
            X_c = torch.cat(X_c)
            y_c = torch.cat(y_c)

            adapter = CHCDSConformal(model, tau=tau, mode=mode)
            adapter.calibrate(X_c, y_c, alpha)
            model = adapter

        return Checkpoint(
            model=model,
            epoch=self.cfg.max_epochs,
            calibrator=self.calibrator,
        )

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
                if (
                    hasattr(model, "imm_loss") or hasattr(model, "mf_loss")
                ) and hasattr(model, "predict_logits"):
                    logits = model.predict_logits(x)
                else:
                    out = model(x)
                    logits = out
                    if isinstance(out, dict):
                        logits = out.get("logits", out.get("probs").log())
                preds.append(logits.cpu())
                targets.append(self._to_index(model, y))

        y_pred = torch.cat(preds)
        y_true = torch.cat(targets)

        if self.calibrator is not None:
            probs = y_pred.softmax(dim=-1)
            probs = self.calibrator(probs)
            y_pred = probs.log()

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
                if hasattr(model, "quantile_loss"):
                    _ = model.quantile_loss(x, y.unsqueeze(1))
                elif self.loss_fn is None and hasattr(model, "imm_loss"):
                    _ = model.imm_loss(x, y)
                elif self.loss_fn is None and hasattr(model, "dsm_loss"):
                    _ = model.dsm_loss(x, y)
                elif hasattr(model, "mf_loss"):
                    _ = model.mf_loss(x, y)
                else:
                    out = model(x)
                    logits = out
                    if isinstance(out, dict):
                        logits = out.get("logits", out.get("probs").log())
                    if self.loss_fn is evidential_loss:
                        targets = self._to_index(model, y)
                        _ = self.loss_fn(out["alpha"], targets)
                    else:
                        targets = self._to_index(model, y)
                        _ = self.loss_fn(logits, targets)



