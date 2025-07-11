Below is a sketch that has worked well for similar projects I’ve built. It is opinionated — you can shuffle pieces around to taste — but every choice is grounded in two goals:
	•	Separability — training logic, model definitions, data, and evaluation can evolve independently.
	•	Extensibility — adding “one more model” or “one more metric” should never require touching four other files.

⸻

1. Core design ideas (why before how)
	1.	Treat “a discrete distribution over a continuous y” as a first-class prediction type.
Your top-level API therefore revolves around DistributionPrediction objects with .probs, .cdf, .mean(), .sample(), etc.
	2.	Models should be plugins.
Every architecture (MDN, discretised mixture of logistics, evidential NN, etc.) lives in its own module, registers itself once, and the rest of the library discovers it via an entry-point.
	3.	Training is declarative.
All tunable knobs live in yaml (pydantic-validated) or dataclass configs. The trainer reads a config and never receives raw kwargs.
	4.	Evaluation is stateless and side-effect-free.
Feed it y_true and DistributionPrediction; it returns numbers or dataclasses. Plotting, logging, or file I/O happen in callbacks.
	5.	No silent binning magic.
Explicit BinningScheme objects own the edges and the ↔ conversions between continuous y and class indices.
(“Equal-width, 256 bins” vs “Empirical quantile, 101 bins” are drop-in replacements.)

⸻

2. Recommended directory / module layout

outdist/
│
├── pyproject.toml          # PEP 621 metadata, build backend, deps
├── src/
│   └── outdist/
│       ├── __init__.py     # *nothing but* lazy imports + version
│       │
│       ├── configs/        # Pure dataclasses / pydantic models
│       │   ├── base.py
│       │   ├── model.py
│       │   ├── trainer.py
│       │   └── data.py
│       │
│       ├── data/
│       │   ├── datasets.py     # Dataset wrappers, splits
│       │   ├── transforms.py   # Scaling, augmentation
│       │   └── binning.py      # BinningScheme classes
│       │
│       ├── predictions/
│       │   └── discrete_dist.py  # `DistributionPrediction`
│       │
│       ├── models/
│       │   ├── __init__.py       # registry + helpers
│       │   ├── base.py           # abstract BaseModel
│       │   ├── mdn.py            # Mixture Density Network
│       │   ├── evidential.py
│       │   └── ...               # others plug in here
│       │
│       ├── losses.py         # CE, label-smoothing, CRPS, NLL…
│       ├── metrics.py        # calibration, Brier, ECE, KL…
│       │
│       ├── training/
│       │   ├── trainer.py        # high-level fit loop
│       │   ├── callback.py       # EarlyStop, Checkpoint, LRSched
│       │   └── optim.py          # thin wrappers around torch.opt
│       │
│       ├── evaluation/
│       │   ├── evaluator.py      # cross-val, CV log-lik table
│       │   └── diagnostics.py    # reliability diagrams etc.
│       │
│       ├── cli/
│       │   ├── train.py
│       │   └── evaluate.py
│       │
│       └── utils/            # seeding, logging, device utils
│
├── tests/                   # pytest; mirror src tree one-for-one
├── docs/                    # mkdocs site; tutorials in /examples
└── .github/
    └── workflows/ci.yml     # lint, mypy, tests on pushes/PRs

Why the src-layout?

Placing your code under src/ prevents tests from accidentally importing the editable checkout instead of the installed package, catching missing dependencies or forgotten __init__.py imports early.

⸻

3. Key abstractions (with minimal signatures)

# predictions/discrete_dist.py
class DistributionPrediction(TypedDict):
    probs: torch.Tensor  # (B, K) softmax over K bins
    bin_edges: torch.Tensor  # (K+1,) monotonically increasing

    def mean(self) -> torch.Tensor: ...
    def var(self) -> torch.Tensor: ...
    def cdf(self, y: torch.Tensor) -> torch.Tensor: ...
    def sample(self, n: int = 1) -> torch.Tensor: ...

# models/base.py
class BaseModel(nn.Module, metaclass=Registrable):
    """
    Abstract NN that maps x -> logits over K bins.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @classmethod
    def default_config(cls) -> ModelConfig: ...

# training/trainer.py
class Trainer:
    def __init__(self, cfg: TrainerConfig):
        ...

    def fit(
        self,
        model: BaseModel,
        train_ds: Dataset,
        val_ds: Dataset | None = None
    ) -> Checkpoint:
        ...

    def evaluate(
        self,
        model: BaseModel,
        test_ds: Dataset,
        metrics: list[str] | None = None
    ) -> dict[str, float]:
        ...

Everything else (callbacks, learning-rate schedules, logging) plugs into Trainer.

⸻

4. How a user works with it (end-to-end)

# ❶ Create a config (yaml or py script)
outdist run train \
    model=mdn                      \
    data=bike_sharing              \
    binning.quantiles=101          \
    trainer.max_epochs=50          \
    trainer.batch_size=256

# ❷ Programmatic API
from outdist import get_model, Trainer, make_dataset

cfg = ModelConfig(name="evidential", hidden=[128, 128])
model = get_model(cfg)

train_ds, val_ds, test_ds = make_dataset("bike_sharing")
trainer = Trainer(TrainerConfig())
ckpt = trainer.fit(model, train_ds, val_ds)

preds = ckpt.model.predict(test_ds[:][0])         # DistributionPrediction
scores = trainer.evaluate(ckpt.model, test_ds)    # {'crps': …, 'nll': …}


⸻

5. Extending the library (new architecture example)

# src/outdist/models/ordinal_flow.py
@register_model("ordinal_flow")
class OrdinalFlow(BaseModel):
    def __init__(self, cfg: OrdinalFlowConfig):
        super().__init__()
        self.encoder = nn.Sequential(...)
        self.flow = nflows.Flow(...)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.flow(z)
        return logits

Write tests → add to docs → nothing else to change.

⸻

6. Packaging & CI tips

Concern	Recommendation
Build backend	hatchling (fast, PEP 517 compliant)
Type-checking	ship py.typed and run mypy --strict in CI
Versioning	hatch version + git tag vX.Y.Z
Docs	mkdocs-material, plus mkdocstrings[python] for auto API
Continuous benchmarks	pytest-bench + Airspeed Velocity (ASV)
Reproducible runs	torch>=2.3 --install_option="--enable-features=..."; xformers guarded behind extra_require = {"gpu": [...]}


⸻

Final thought

A good structure is less about files and more about friction-free paths:
	•	Add a dataset → drop a loader in data/.
	•	Add a model → subclass BaseModel, register, done.
	•	Try a new metric → pure function in metrics.py.

If each of those feels like a five-minute change, you’ve won. Happy building!
