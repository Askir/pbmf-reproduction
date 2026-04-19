"""Biomarker evaluation: HRs, log-rank tests, and stratified KM plots."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


@dataclass
class BiomarkerResult:
    n_pos: int
    n_neg: int
    treatment_hr_pos: float    # treatment vs control within B+
    treatment_hr_pos_ci: tuple[float, float]
    treatment_hr_pos_p: float
    treatment_hr_neg: float    # treatment vs control within B-
    treatment_hr_neg_ci: tuple[float, float]
    treatment_hr_neg_p: float
    predictive_ratio: float    # HR_pos / HR_neg; <1 means the biomarker selects treatment responders
    interaction_p: float        # p-value of arm × biomarker interaction in Cox model

    def as_series(self) -> pd.Series:
        return pd.Series({
            "n_B+":      self.n_pos,
            "n_B-":      self.n_neg,
            "HR(Tx|B+)": self.treatment_hr_pos,
            "CI(Tx|B+)": f"[{self.treatment_hr_pos_ci[0]:.2f}, {self.treatment_hr_pos_ci[1]:.2f}]",
            "p(Tx|B+)":  self.treatment_hr_pos_p,
            "HR(Tx|B-)": self.treatment_hr_neg,
            "CI(Tx|B-)": f"[{self.treatment_hr_neg_ci[0]:.2f}, {self.treatment_hr_neg_ci[1]:.2f}]",
            "p(Tx|B-)":  self.treatment_hr_neg_p,
            "HR+/HR-":   self.predictive_ratio,
            "p(interaction)": self.interaction_p,
        })


def _cox_hr(df: pd.DataFrame, time_col: str, event_col: str, covariate: str) -> tuple[float, tuple[float, float], float]:
    """Fit a univariate Cox model on a single covariate, return HR, 95% CI, p."""
    cph = CoxPHFitter()
    cph.fit(df[[time_col, event_col, covariate]].dropna(),
            duration_col=time_col, event_col=event_col)
    row = cph.summary.loc[covariate]
    return (float(row["exp(coef)"]),
            (float(row["exp(coef) lower 95%"]), float(row["exp(coef) upper 95%"])),
            float(row["p"]))


def evaluate_biomarker(
    df: pd.DataFrame,
    label_col: str,
    time_col: str = "OS",
    event_col: str = "OS_event",
    arm_col: str = "TRT01P",
    treatment_arm: str = "Atezolizumab",
    control_arm: str = "Docetaxel",
) -> BiomarkerResult:
    """Evaluate a binary biomarker on a dataframe.

    ``label_col`` should be 0/1 (B- / B+). Computes treatment-vs-control HR
    within each biomarker stratum and the interaction p-value from a Cox
    model with arm, biomarker, and their product.
    """
    d = df[[label_col, arm_col, time_col, event_col]].dropna().copy()
    d = d[d[arm_col].isin([treatment_arm, control_arm])]
    d["_tx"]   = (d[arm_col] == treatment_arm).astype(int)
    d["_lab"]  = d[label_col].astype(int)
    d["_tx_x_lab"] = d["_tx"] * d["_lab"]

    pos = d[d["_lab"] == 1]
    neg = d[d["_lab"] == 0]

    hr_pos, ci_pos, p_pos = _cox_hr(pos, time_col, event_col, "_tx")
    hr_neg, ci_neg, p_neg = _cox_hr(neg, time_col, event_col, "_tx")

    cph = CoxPHFitter()
    cph.fit(d[[time_col, event_col, "_tx", "_lab", "_tx_x_lab"]],
            duration_col=time_col, event_col=event_col)
    interaction_p = float(cph.summary.loc["_tx_x_lab", "p"])

    return BiomarkerResult(
        n_pos=len(pos), n_neg=len(neg),
        treatment_hr_pos=hr_pos, treatment_hr_pos_ci=ci_pos, treatment_hr_pos_p=p_pos,
        treatment_hr_neg=hr_neg, treatment_hr_neg_ci=ci_neg, treatment_hr_neg_p=p_neg,
        predictive_ratio=hr_pos / hr_neg if hr_neg > 0 else np.nan,
        interaction_p=interaction_p,
    )


def plot_km_strata(
    df: pd.DataFrame,
    label_col: str,
    time_col: str = "OS",
    event_col: str = "OS_event",
    arm_col: str = "TRT01P",
    treatment_arm: str = "Atezolizumab",
    control_arm: str = "Docetaxel",
    title: str | None = None,
    ax_pair: tuple[plt.Axes, plt.Axes] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Two-panel KM: B+ on the left, B- on the right, each with both arms overlaid."""
    if ax_pair is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    else:
        axes = ax_pair
        fig = axes[0].figure

    d = df[[label_col, arm_col, time_col, event_col]].dropna()
    d = d[d[arm_col].isin([treatment_arm, control_arm])]

    for ax, (lab_val, lab_name) in zip(axes, [(1, "B+"), (0, "B-")]):
        sub = d[d[label_col] == lab_val]
        for arm, color in [(treatment_arm, "C0"), (control_arm, "C1")]:
            s = sub[sub[arm_col] == arm]
            if len(s) == 0:
                continue
            med = s[time_col].median()
            (KaplanMeierFitter()
                .fit(s[time_col], s[event_col],
                     label=f"{arm} n={len(s)} med={med:.1f}")
                .plot_survival_function(ax=ax, ci_show=False, color=color))
        atz = sub[sub[arm_col] == treatment_arm]
        doc = sub[sub[arm_col] == control_arm]
        if len(atz) and len(doc):
            lr = logrank_test(atz[time_col], doc[time_col], atz[event_col], doc[event_col])
            ax.set_title(f"{lab_name} (n={len(sub)}, log-rank p={lr.p_value:.3g})")
        else:
            ax.set_title(f"{lab_name} (n={len(sub)})")
        ax.set_xlabel("Months")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Survival probability")
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    return fig, tuple(axes)
