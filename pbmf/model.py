"""PBMF neural biomarker model with a differentiable log-rank contrastive loss.

The loss formulation follows the predictive-biomarker objective: for a learned
score p(x) ∈ [0, 1] treated as soft B+/B- group membership, compute the
weighted log-rank Z-statistic separately within each treatment arm. We want:

- Z(atezo) very negative — B+ patients on atezo survive longer than B-
- Z(docetaxel) ≈ 0      — the same split has no effect under standard-of-care

The weighted log-rank uses soft at-risk counts n_B+(t) = Σ p_i · 𝟙(T_i ≥ t)
and soft event counts d_B+(t) = Σ p_i · 𝟙(T_i = t, δ_i = 1), which preserves
differentiability w.r.t. p.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def differentiable_logrank_z(
    times: torch.Tensor,
    events: torch.Tensor,
    score: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Signed weighted log-rank Z-statistic for soft group membership.

    ``score`` is in [0, 1] and is interpreted as soft membership in group B+;
    (1 - score) is the soft membership in B-. Returns (O - E) / sqrt(V), where
    O and E are the observed and expected events in B+ across event times. A
    negative value means B+ has fewer events than expected (better survival).
    """
    # Unique event times, in ascending order (use all distinct times so
    # at-risk sets stay well-defined even if all deaths tie).
    t_unique, _ = torch.unique(times, return_counts=True)
    t_unique, _ = torch.sort(t_unique)

    # at_risk[k, i] = 1 if subject i is at risk at time t_unique[k]
    at_risk = (times.unsqueeze(0) >= t_unique.unsqueeze(1)).float()
    # event_at[k, i] = 1 if subject i has an event exactly at t_unique[k]
    event_at = ((times.unsqueeze(0) == t_unique.unsqueeze(1))
                & (events.unsqueeze(0) == 1)).float()

    w1 = score          # weight in B+
    w2 = 1.0 - score    # weight in B-

    n1 = at_risk @ w1
    n2 = at_risk @ w2
    d1 = event_at @ w1
    d2 = event_at @ w2
    d  = d1 + d2
    n  = n1 + n2

    # Hypergeometric expected and variance at each event time.
    # Only contribute when there's at least one event and at least 2 at risk.
    has_event = (d > 0) & (n > 1)

    e1 = torch.where(has_event, d * n1 / (n + eps), torch.zeros_like(d))
    var = torch.where(
        has_event,
        d * n1 * n2 * (n - d) / (n.pow(2) * (n - 1) + eps),
        torch.zeros_like(d),
    )

    O_minus_E = (d1 - e1).sum()
    V = var.sum()
    return O_minus_E / torch.sqrt(V + eps)


class PBMFNet(nn.Module):
    """Small MLP: `in_dim` → 64 → 2, softmaxed. Output[:, 1] is the B+ score."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, 1]


def pbmf_contrastive_loss(
    score: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    arm: torch.Tensor,
    lam_control: float = 1.0,
    treatment_arm_value: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the PBMF contrastive loss.

    ``arm`` is 0/1: 1 = treatment (atezo), 0 = control (docetaxel).
    Loss = Z(treatment) + λ · |Z(control)|.
      - Negative Z(treatment) is good (B+ lives longer on atezo)
      - |Z(control)| ≈ 0 keeps the biomarker predictive rather than prognostic.
    """
    tx_mask = arm == treatment_arm_value
    ct_mask = ~tx_mask

    z_tx = differentiable_logrank_z(times[tx_mask], events[tx_mask], score[tx_mask])
    z_ct = differentiable_logrank_z(times[ct_mask], events[ct_mask], score[ct_mask])
    loss = z_tx + lam_control * z_ct.abs()
    return loss, {"z_treatment": z_tx.detach(), "z_control": z_ct.detach()}


def train_pbmf(
    X: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    arm: torch.Tensor,
    hidden: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 500,
    lam_control: float = 1.0,
    dropout: float = 0.0,
    verbose: int = 0,
    seed: int | None = None,
) -> tuple[PBMFNet, list[dict]]:
    """Single-model training loop, full-batch. Returns (model, history)."""
    if seed is not None:
        torch.manual_seed(seed)
    model = PBMFNet(in_dim=X.shape[1], hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        score = model(X)[:, 1]
        loss, diag = pbmf_contrastive_loss(score, times, events, arm, lam_control=lam_control)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        if verbose and (epoch % max(1, epochs // verbose) == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d} | loss {loss.item():+.4f} | "
                  f"Z_tx {diag['z_treatment'].item():+.3f} | "
                  f"Z_ct {diag['z_control'].item():+.3f}")
        history.append({
            "epoch": epoch,
            "loss": loss.item(),
            "z_treatment": diag["z_treatment"].item(),
            "z_control": diag["z_control"].item(),
        })
    return model, history


def train_one_bagged(
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    arm: np.ndarray,
    patient_idx: np.ndarray,
    feat_idx: np.ndarray,
    steps: int = 500,
    ifrac: float = 0.1,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    hidden: int = 64,
    lam_control: float = 0.0,
    seed: int = 0,
) -> "PBMFNet":
    """Train one bagged model with per-step dropout.

    - ``patient_idx`` selects the bag (paper uses 80% of patients).
    - ``feat_idx`` selects features (paper uses 28 of 29 in the POPLAR/OAK config).
    - At each gradient step we further resample a ``(1 - ifrac)`` fraction of the bag.
    - ``lam_control = 0`` by default: for POPLAR/OAK the paper leaves out the control-arm
      penalty and relies on ensemble diversity + pruning to enforce the predictive signal.
    """
    torch.manual_seed(seed)
    Xb = X[np.ix_(patient_idx, feat_idx)].astype(np.float32)
    times_b  = times[patient_idx].astype(np.float32)
    events_b = events[patient_idx].astype(np.float32)
    arm_b    = arm[patient_idx].astype(np.int64)

    Xb_t     = torch.from_numpy(Xb)
    times_t  = torch.from_numpy(times_b)
    events_t = torch.from_numpy(events_b)
    arm_t    = torch.from_numpy(arm_b)

    n_bag  = len(patient_idx)
    n_keep = max(10, int(round(n_bag * (1.0 - ifrac))))

    model = PBMFNet(in_dim=len(feat_idx), hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    rng = np.random.RandomState(seed)
    for _ in range(steps):
        sub = rng.choice(n_bag, size=n_keep, replace=False)
        sub_t = torch.from_numpy(sub).long()
        opt.zero_grad()
        score = model(Xb_t[sub_t])[:, 1]
        loss, _ = pbmf_contrastive_loss(
            score,
            times_t[sub_t], events_t[sub_t], arm_t[sub_t],
            lam_control=lam_control,
        )
        # Degenerate soft-assignment → variance is 0 → nan loss: skip this step.
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
    return model


def _infer_sign_flip(
    model: "PBMFNet",
    X: np.ndarray,
    feat_idx: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    arm: np.ndarray,
    treatment_arm_value: int = 1,
) -> bool:
    """Sign-correct a trained model. Channel 1 should be B+ (better atezo survival).

    If Z(arm=treatment) on channel-1 scores is > 0 (B+ actually does worse on atezo),
    flip the channel assignment.
    """
    with torch.no_grad():
        X_t = torch.from_numpy(X[:, feat_idx].astype(np.float32))
        score = model(X_t)[:, 1]
    tx = arm == treatment_arm_value
    z = differentiable_logrank_z(
        torch.from_numpy(times[tx].astype(np.float32)),
        torch.from_numpy(events[tx].astype(np.float32)),
        score[torch.from_numpy(tx)],
    )
    return bool(z.item() > 0)


def train_pbmf_ensemble(
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    arm: np.ndarray,
    M: int = 100,
    steps: int = 500,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    patient_bag: float = 0.8,
    n_features: int | None = None,
    ifrac: float = 0.1,
    lam_control: float = 0.0,
    hidden: int = 64,
    seed: int = 0,
    verbose: int = 0,
) -> list[dict]:
    """Train a bagged ensemble. Returns a list of {'model','feat_idx','flip'}."""
    N, F = X.shape
    n_pat  = int(round(N * patient_bag))
    n_feat = F if n_features is None else min(F, n_features)

    rng = np.random.RandomState(seed)
    ensemble: list[dict] = []
    for m in range(M):
        seed_m = int(rng.randint(10**9))
        p_idx = rng.choice(N, size=n_pat, replace=False)
        f_idx = (rng.choice(F, size=n_feat, replace=False) if n_feat < F
                 else np.arange(F))
        model = train_one_bagged(
            X, times, events, arm, p_idx, f_idx,
            steps=steps, ifrac=ifrac, lr=lr, weight_decay=weight_decay,
            hidden=hidden, lam_control=lam_control, seed=seed_m,
        )
        flip = _infer_sign_flip(model, X, f_idx, times, events, arm)
        ensemble.append({"model": model, "feat_idx": f_idx, "flip": flip})
        if verbose and (m + 1) % verbose == 0:
            n_flipped = sum(e["flip"] for e in ensemble)
            print(f"  trained {m+1}/{M}  ({n_flipped} sign-flipped)")
    return ensemble


def score_ensemble(ensemble: list[dict], X: np.ndarray) -> np.ndarray:
    """Per-model B+ scores after sign correction. Returns (N, M)."""
    X = X.astype(np.float32)
    N = X.shape[0]
    M = len(ensemble)
    out = np.zeros((N, M), dtype=np.float32)
    for m, info in enumerate(ensemble):
        with torch.no_grad():
            probs = info["model"](torch.from_numpy(X[:, info["feat_idx"]])).numpy()
        s = probs[:, 1]
        out[:, m] = (1.0 - s) if info["flip"] else s
    return out


def prune_ensemble(
    scores: np.ndarray,
    percentile: float = 95.0,
    threshold: float = 0.5,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """Consensus-based pruning from the PBMF paper.

    R (N×M): binary B+ assignments.
    A (N×N): co-assignment fraction A[i,j] = mean_m 1{R[i,m] == R[j,m]}.
    C (N×M): C[i,m] = Pearson(A[:,i], R[:,m]).
    Threshold C at the ``percentile``-th percentile, count strong-agreement patients per
    model, keep models whose count exceeds the ``percentile``-th percentile of counts.
    """
    R = (scores >= threshold).astype(np.float32)
    N, M = R.shape

    # Co-assignment: A = (R R^T + (1-R)(1-R)^T) / M
    A = (R @ R.T + (1.0 - R) @ (1.0 - R).T) / M

    A_c = A - A.mean(axis=0, keepdims=True)
    R_c = R - R.mean(axis=0, keepdims=True)
    norm_A = np.sqrt((A_c ** 2).sum(axis=0)) + eps   # (N,)
    norm_R = np.sqrt((R_c ** 2).sum(axis=0)) + eps   # (M,)
    C = (A_c.T @ R_c) / (norm_A[:, None] * norm_R[None, :])

    c_thresh = np.percentile(C, percentile)
    strong = C >= c_thresh
    counts = strong.sum(axis=0)
    count_thresh = np.percentile(counts, percentile)
    kept = np.where(counts > count_thresh)[0]

    return kept, {
        "C": C, "A": A, "counts": counts,
        "c_thresh": float(c_thresh),
        "count_thresh": float(count_thresh),
    }


def distill_tree(
    scores: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    max_depth: int = 4,
    epsilon: float = 0.1,
    min_samples_leaf: int = 10,
    ccp_alpha: float = 0.0,
    random_state: int = 0,
):
    """Knowledge-distill an ensemble consensus score into a decision tree.

    Trains on "high confidence" patients only — those with |score - 0.5| > ``epsilon`` —
    so the tree learns from cases where the ensemble has taken a clear side.
    ``ccp_alpha`` enables cost-complexity pruning, which collapses splits whose
    children agree on the same class (useful for readable rules).

    Returns ``(tree, conf_mask, summary)`` where ``summary`` is a small dict of
    fit statistics (n kept, n pruned, train accuracy).
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    if len(scores) != X.shape[0]:
        raise ValueError("scores and X must have the same length")

    conf_mask = np.abs(scores - 0.5) > epsilon
    X_conf = X[conf_mask]
    y_conf = (scores[conf_mask] > 0.5).astype(int)

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=random_state,
    )
    tree.fit(X_conf, y_conf)

    summary = {
        "n_total": len(scores),
        "n_confident": int(conf_mask.sum()),
        "n_ambiguous": int((~conf_mask).sum()),
        "n_bplus_in_train": int(y_conf.sum()),
        "train_accuracy": float(accuracy_score(y_conf, tree.predict(X_conf))),
        "epsilon": epsilon,
        "max_depth": max_depth,
        "ccp_alpha": ccp_alpha,
    }
    return tree, conf_mask, summary


__all__ = [
    "differentiable_logrank_z",
    "PBMFNet",
    "pbmf_contrastive_loss",
    "train_pbmf",
    "train_one_bagged",
    "train_pbmf_ensemble",
    "score_ensemble",
    "prune_ensemble",
    "distill_tree",
]
