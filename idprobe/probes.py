"""Linear and MLP probes, trained under an identical protocol.

The scientific question is whether a nonlinearity buys anything -- so the two
architectures must differ ONLY in architecture. Same optimiser, schedule,
regularisation, early-stopping rule, seeds, and standardisation. Any other
difference would confound "the feature is nonlinearly encoded" with "the MLP
just got a better training run".

Protocol notes:
  * Features are standardised with TRAIN statistics. Hidden-state norms grow by
    orders of magnitude across layers in modern LMs; without this the same
    learning rate is wildly wrong at different depths.
  * Accuracy is reported on the held-out TEST split, with the VAL split used
    only for early stopping. Cheng et al. report best-val-over-seeds, which is
    optimistically biased; keeping a third split makes the linear-vs-MLP test honest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "logistic": nn.Sigmoid, "tanh": nn.Tanh, "gelu": nn.GELU}


@dataclass
class ProbeData:
    """Standardised, device-resident splits shared by every probe at one layer."""
    Xtr: "torch.Tensor"
    ytr: "torch.Tensor"
    Xva: "torch.Tensor"
    yva: "torch.Tensor"
    Xte: "torch.Tensor"
    yte: "torch.Tensor"
    n_classes: int
    d_in: int


@dataclass
class ProbeResult:
    kind: str
    seed: int
    test_acc: float
    val_acc: float
    correct: np.ndarray  # per-test-example boolean, for McNemar (empty for regression)
    # Readout weights, [n_classes, d] for the linear probe. Kept so a caller can ask
    # what the probe direction costs geometrically (conditional_id.residual_id)
    # without retraining. None for architectures with no single readout direction.
    direction: np.ndarray | None = None
    # Which quantity test_acc/val_acc hold: "accuracy" for classifiers, "r2" for
    # regression. Both are higher-is-better, so early stopping is unchanged.
    metric: str = "accuracy"


class OrdinalProbe(nn.Module):
    """Ordered logit: one score direction plus K-1 learned, ordered cut points.

    The point of this readout is Direction 2's hypothesis. A softmax over binned
    classes treats the bins as unordered, so it must carve K separate regions out of
    the representation; if the underlying quantity is encoded *monotonically* along
    one direction, that is a needlessly hard way to read it and an MLP will look like
    it has found nonlinear structure when it has only supplied the missing ordering.
    An ordinal head has the ordering built in, so it isolates that explanation.

    P(y <= j) = sigmoid(theta_j - s), s = w.x + b. Thresholds are parameterised as a
    base plus positive increments (softplus) so theta stays sorted throughout
    training -- unsorted cut points would make the implied bin probabilities
    negative.
    """

    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.score = nn.Linear(d_in, 1)
        self.n_classes = n_classes
        self.theta0 = nn.Parameter(torch.zeros(1))
        self.deltas = nn.Parameter(torch.zeros(max(n_classes - 2, 0)))

    def thresholds(self) -> torch.Tensor:
        steps = nn.functional.softplus(self.deltas)
        return torch.cat([self.theta0, self.theta0 + torch.cumsum(steps, 0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x).squeeze(-1)

    def loss(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        th = self.thresholds()
        # Pad with -inf/+inf so P(y<=-1)=0 and P(y<=K-1)=1 fall out of the same
        # expression, instead of special-casing the two boundary bins.
        big = torch.tensor([float("inf")], device=s.device)
        padded = torch.cat([-big, th, big])
        upper = torch.sigmoid(padded[y + 1] - s)
        lower = torch.sigmoid(padded[y] - s)
        return -torch.log((upper - lower).clamp_min(1e-12)).mean()

    def predict(self, s: torch.Tensor) -> torch.Tensor:
        return (s.unsqueeze(1) > self.thresholds().unsqueeze(0)).sum(1)


class RegressionProbe(nn.Module):
    """Linear readout of a continuous target, scored by R^2.

    Direction 2 asks whether `sentence_length` is a magnitude rather than a feature.
    Classification accuracy over bins cannot answer that -- a probe can track length
    almost perfectly and still land in the wrong bin near every boundary. R^2 against
    the raw count separates "the layer does not encode length" from "the layer
    encodes it fine and the binned softmax is the wrong instrument".
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.out = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x).squeeze(-1)

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(pred, y.float())

    predict = None  # scored by R^2, not by a class decision


def build_probe(kind: str, d_in: int, n_classes: int, hidden: int, activation: str) -> nn.Module:
    if kind == "linear":
        return nn.Linear(d_in, n_classes)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(d_in, hidden), ACTIVATIONS[activation](), nn.Linear(hidden, n_classes)
        )
    if kind == "ordinal":
        return OrdinalProbe(d_in, n_classes)
    if kind == "regression":
        return RegressionProbe(d_in)
    raise ValueError(f"unknown probe kind: {kind}")


def prepare_data(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    device: torch.device | None = None,
    continuous: bool = False,
) -> ProbeData:
    """Standardise with TRAIN statistics and upload, once per layer.

    Every (architecture, seed) combination at a layer trains on exactly these
    tensors, so this work is done once rather than once per probe.

    `continuous` keeps the targets as floats for a regression readout. The FEATURE
    standardisation is identical either way, which is what lets a regression R^2 and
    a classification accuracy at the same layer be read as two views of one probe
    rather than two different experiments.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6

    def tt(a, dt):
        return torch.as_tensor(a, dtype=dt, device=device)

    if continuous:
        # The TARGET needs train-statistic standardisation too, for the same reason
        # the features do. A raw word count averages ~16 while the head initialises
        # near 0, and at lr=1e-3 the bias alone cannot cross that gap in the epoch
        # budget -- the probe converges to a constant and reports a large negative
        # R^2 that looks like "length is not encoded" but means "it never trained".
        # R^2 is invariant to an affine change of the target, so the number reported
        # in this space is exactly the R^2 against raw counts.
        ymu, ysd = float(np.mean(ytr)), float(np.std(ytr)) + 1e-12
        ytr, yva, yte = ((np.asarray(v, np.float64) - ymu) / ysd for v in (ytr, yva, yte))

    ydt = torch.float32 if continuous else torch.long
    return ProbeData(
        tt((Xtr - mu) / sd, torch.float32), tt(ytr, ydt),
        tt((Xva - mu) / sd, torch.float32), tt(yva, ydt),
        tt((Xte - mu) / sd, torch.float32), tt(yte, ydt),
        n_classes=1 if continuous else int(max(ytr.max(), yva.max(), yte.max())) + 1,
        d_in=int(Xtr.shape[1]),
    )


def train_probe(
    kind: str,
    data: ProbeData,
    *,
    seed: int = 0,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    patience: int = 10,
    hidden: int = 200,
    activation: str = "relu",
) -> ProbeResult:
    device = data.Xtr.device
    torch.manual_seed(seed)

    model = build_probe(kind, data.d_in, data.n_classes, hidden, activation).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Heads that need a different objective carry it themselves, so every kind runs
    # through this one loop: same optimiser, schedule, early-stopping rule, and
    # minibatch order. That is the whole basis for comparing them.
    loss_fn = getattr(model, "loss", None) or nn.CrossEntropyLoss()

    # Dedicated generator for the shuffle. Seeding the global stream instead
    # would make minibatch order depend on how much randomness the model's
    # construction consumed, which differs between linear and MLP -- and would
    # confound "the nonlinearity helped" with "it saw a different data order".
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(seed)

    # -inf, not -1: accuracy is bounded below by 0 but R^2 is not bounded at all,
    # and an untrained regression head starts far below -1. A finite floor would
    # leave `best_state` unset for every epoch and never checkpoint anything.
    best_val, best_state, waited = -float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(data.Xtr), generator=shuffle_gen).to(device)
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss_fn(model(data.Xtr[idx]), data.ytr[idx]).backward()
            opt.step()

        val_acc = _score(model, data.Xva, data.yva)
        if val_acc > best_val:
            best_val, waited = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    if kind == "regression":
        # No per-example "correct" for a continuous target: R^2 on the held-out split
        # is the reported score, and McNemar has nothing to condition on.
        return ProbeResult(kind, seed, _score(model, data.Xte, data.yte), float(best_val),
                           np.empty(0, dtype=bool), metric="r2")

    with torch.no_grad():
        correct = (_decide(model, data.Xte) == data.yte).cpu().numpy()

    direction = (model.weight.detach().cpu().numpy()
                 if isinstance(model, nn.Linear) else None)
    return ProbeResult(kind, seed, float(correct.mean()), float(best_val), correct,
                       direction=direction)


def _decide(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Class decision for a classifier, honouring a head's own decision rule.

    An ordinal head scores with one number and cuts it at learned thresholds, so
    argmax over a class axis would be meaningless for it.
    """
    out = model(X)
    predict = getattr(model, "predict", None)
    return predict(out) if predict is not None else out.argmax(1)


@torch.no_grad()
def _score(model: nn.Module, X: torch.Tensor, y: torch.Tensor, chunk: int = 8192) -> float:
    """Held-out score: accuracy for classifiers, R^2 for regression.

    Both are higher-is-better, so the early-stopping rule needs no special case.
    """
    model.eval()
    if isinstance(model, RegressionProbe):
        preds = torch.cat([model(X[i : i + chunk]) for i in range(0, len(X), chunk)])
        resid = ((y - preds) ** 2).sum()
        total = ((y - y.mean()) ** 2).sum()
        return float(1 - resid / total) if total > 0 else float("nan")
    hits = 0
    for i in range(0, len(X), chunk):
        hits += (_decide(model, X[i : i + chunk]) == y[i : i + chunk]).sum().item()
    return hits / len(X)


def train_ordinal_probe(data: ProbeData, **kw) -> ProbeResult:
    """Ordered-logit readout of binned classes. See OrdinalProbe for the rationale."""
    return train_probe("ordinal", data, **kw)


def train_regression_probe(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    device: "torch.device | None" = None,
    **kw,
) -> ProbeResult:
    """Linear regression onto a continuous target, scored by held-out R^2.

    Takes raw arrays rather than a ProbeData because the targets must be uploaded as
    floats; reusing a classification ProbeData would silently truncate them to bins,
    which is the exact confound this readout exists to remove.
    """
    data = prepare_data(Xtr, ytr, Xva, yva, Xte, yte, device, continuous=True)
    return train_probe("regression", data, **kw)


def scalar_summaries(X: np.ndarray) -> np.ndarray:
    """A handful of magnitude-only features: [||h||, mean, std, max].

    Deliberately blind to direction. If a probe on these four numbers recovers most
    of the MLP's accuracy, the "feature" the MLP was extracting is a magnitude, and
    the linear-vs-MLP gap at that layer says something about readout geometry rather
    than about entanglement.
    """
    X = np.asarray(X, dtype=np.float32)
    return np.stack([np.linalg.norm(X, axis=1), X.mean(1), X.std(1), X.max(1)], axis=1)


def norm_baseline(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    device: "torch.device | None" = None,
    **kw,
) -> dict[str, ProbeResult]:
    """Probes on ||h|| alone and on the four scalar summaries.

    Returns {'norm_linear', 'norm_mlp', 'summary_linear', 'summary_mlp'}. The MLP is
    included because a single scalar carries a monotone quantity that a *linear*
    softmax over bins still cannot cut correctly -- so a norm-only MLP is the sharper
    test of the magnitude hypothesis, and a norm-only linear probe is its floor.
    """
    feats = [scalar_summaries(X) for X in (Xtr, Xva, Xte)]
    out = {}
    for name, cols in (("norm", slice(0, 1)), ("summary", slice(None))):
        data = prepare_data(feats[0][:, cols], ytr, feats[1][:, cols], yva,
                            feats[2][:, cols], yte, device)
        for kind in ("linear", "mlp"):
            out[f"{name}_{kind}"] = train_probe(kind, data, **kw)
    return out


__all__ = ["prepare_data", "train_probe", "build_probe", "ProbeData", "ProbeResult",
           "OrdinalProbe", "RegressionProbe", "train_ordinal_probe",
           "train_regression_probe", "scalar_summaries", "norm_baseline"]
