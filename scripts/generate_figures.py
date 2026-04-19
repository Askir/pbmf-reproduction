"""Regenerate the README figures from scratch.

Runs a trimmed-down PBMF pipeline (M=300 for speed) and saves:
  figures/oak_km.png       — held-out OAK KM curves, B+ vs B- by arm
  figures/distilled_tree.png — the depth-4 distilled decision tree
  figures/training_curves.png — single-model loss / Z-stats over epochs

Usage:  uv run python scripts/generate_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

from pbmf import (
    build_features,
    evaluate_biomarker,
    plot_km_strata,
    prune_ensemble,
    score_ensemble,
    train_pbmf_ensemble,
    distill_tree,
)
from pbmf.model import train_pbmf

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def load_data():
    feat, _ = build_features()
    feat = feat.dropna(subset=["OS", "OS_event", "blSLD"]).reset_index(drop=True)
    clinical = ["age", "sex_M", "ecog", "blSLD", "metsites",
                "hist_squamous", "ever_smoker", "msaf", "btmb"]
    mut = [c for c in feat.columns if c.startswith("mut_")]
    poplar_mask = feat["Trial"].values == "POPLAR"
    oak_mask = feat["Trial"].values == "OAK"

    scaler = StandardScaler().fit(feat.loc[poplar_mask, clinical])
    X_scaled = np.hstack(
        [scaler.transform(feat[clinical]), feat[mut].values]
    ).astype(np.float32)
    X_raw = np.hstack([feat[clinical].values, feat[mut].values]).astype(np.float32)

    return feat, clinical + mut, X_scaled, X_raw, poplar_mask, oak_mask


def training_curves(feat, X, poplar_mask):
    times = feat.loc[poplar_mask, "OS"].values.astype(np.float32)
    events = feat.loc[poplar_mask, "OS_event"].values.astype(np.float32)
    arm = (feat.loc[poplar_mask, "TRT01P"] == "Atezolizumab").astype(int).values

    torch.manual_seed(42); np.random.seed(42)
    _, hist = train_pbmf(
        X=torch.tensor(X[poplar_mask]),
        times=torch.tensor(times),
        events=torch.tensor(events),
        arm=torch.tensor(arm, dtype=torch.long),
        hidden=64, lr=1e-3, weight_decay=1e-3,
        epochs=200, lam_control=1.0, seed=42,
    )
    import pandas as pd
    h = pd.DataFrame(hist)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(h["epoch"], h["loss"], color="k")
    axes[0].set_title("training loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Z(atezo) + |Z(doce)|")
    axes[1].plot(h["epoch"], h["z_treatment"], label="Z(atezo) — want ↓↓", color="C0")
    axes[1].plot(h["epoch"], h["z_control"], label="Z(docetaxel) — want ≈ 0", color="C1")
    axes[1].axhline(0, color="gray", linestyle=":", linewidth=0.8)
    axes[1].set_title("log-rank Z-statistics during training")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Z")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'training_curves.png'}")


def ensemble_and_tree(feat, feature_names, X_scaled, X_raw, poplar_mask, oak_mask):
    times = feat["OS"].values.astype(np.float32)
    events = feat["OS_event"].values.astype(np.int64)
    arm = (feat["TRT01P"] == "Atezolizumab").astype(int).values

    print("training ensemble (M=300)…")
    ens = train_pbmf_ensemble(
        X_scaled[poplar_mask], times[poplar_mask], events[poplar_mask], arm[poplar_mask],
        M=300, steps=500, lr=1e-2, patient_bag=0.8,
        n_features=X_scaled.shape[1] - 1, ifrac=0.1, lam_control=0.0,
        seed=0, verbose=0,
    )
    p_scores = score_ensemble(ens, X_scaled[poplar_mask])
    o_scores = score_ensemble(ens, X_scaled[oak_mask])
    kept, _ = prune_ensemble(p_scores, percentile=95)
    print(f"kept {len(kept)}/{len(ens)} after pruning")
    p_cons = p_scores[:, kept].mean(axis=1)
    o_cons = o_scores[:, kept].mean(axis=1)

    # KM plot on OAK using ensemble consensus
    oak_df = feat[oak_mask].assign(
        score=o_cons, bplus=(o_cons >= 0.5).astype(int)
    )
    summary = evaluate_biomarker(oak_df, label_col="bplus").as_series()
    print(f"Ensemble OAK: HR+/HR- = {summary['HR+/HR-']}, int p = {summary['p(interaction)']}")

    fig, _ = plot_km_strata(
        oak_df,
        label_col="bplus",
        title="OAK (held out) — OS by treatment, PBMF ensemble B+ vs B-",
    )
    fig.savefig(OUT / "oak_km.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'oak_km.png'}")

    # Distilled tree
    tree, _, tsum = distill_tree(
        scores=p_cons, X=X_raw[poplar_mask], feature_names=feature_names,
        max_depth=4, epsilon=0.1, min_samples_leaf=10,
        ccp_alpha=0.005, random_state=0,
    )
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(
        tree, feature_names=feature_names, class_names=["B-", "B+"],
        filled=True, rounded=True, precision=2, impurity=False, ax=ax,
    )
    ax.set_title(
        f"Distilled tree (max_depth=4, ε=0.1, ccp_alpha=0.005) — "
        f"trained on {tsum['n_confident']} confident POPLAR patients"
    )
    plt.tight_layout()
    fig.savefig(OUT / "distilled_tree.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'distilled_tree.png'}")


def main():
    feat, feat_names, X_scaled, X_raw, poplar_mask, oak_mask = load_data()
    training_curves(feat, X_scaled, poplar_mask)
    ensemble_and_tree(feat, feat_names, X_scaled, X_raw, poplar_mask, oak_mask)


if __name__ == "__main__":
    main()
