"""Data loading and feature engineering for the Gandara 2018 POPLAR/OAK data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_XLSX = Path(__file__).resolve().parent.parent / "41591_2018_134_MOESM3_ESM.xlsx"

_CLINICAL_SHEETS = {"POPLAR": "POPLAR_Clinical_Data", "OAK": "OAK_Clinical_Data"}
_VARIANT_SHEET = "OAK_POPLAR_btmb_variants"

_NUMERIC_CLINICAL = [
    "cfDNA_Input_ng", "MSAF", "Median_exon_coverage", "btmb", "BAGE",
    "ECOGGR", "blSLD", "METSITES", "PFS", "PFS.CNSR", "OS", "OS.CNSR",
]


def load_clinical(trial: str, xlsx_path: str | Path = DEFAULT_XLSX) -> pd.DataFrame:
    sheet = _CLINICAL_SHEETS[trial]
    df = pd.read_excel(xlsx_path, sheet_name=sheet).replace({".": np.nan})
    df = df.rename(columns={
        "cfDNA_input_ng": "cfDNA_Input_ng",
        "btmb_QC": "QC_Status",
        "PRIORTXC": "PRIORTX",
        "EMLAMUT": "EML4MUT",
    })
    df["Trial"] = trial
    df["TRT01P"] = df["TRT01P"].replace({"MPDL3280A": "Atezolizumab"})
    for col in _NUMERIC_CLINICAL:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["OS_event"] = (df["OS.CNSR"] == 0).astype(int)
    df["PFS_event"] = (df["PFS.CNSR"] == 0).astype(int)
    return df


def load_variants(xlsx_path: str | Path = DEFAULT_XLSX) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, sheet_name=_VARIANT_SHEET)


def _gene_prevalence(variants: pd.DataFrame, patient_ids: set[int],
                     effect_filter: str) -> pd.Series:
    v = variants[variants["PtID"].isin(patient_ids)].copy()
    if effect_filter == "non-synonymous":
        v = v[v["effect"] != "synonymous"]
    elif effect_filter == "non-syn-non-driver":
        v = v[(v["effect"] != "synonymous") & (v["omitted_driver_mutation"] != "yes")]
    elif effect_filter != "all":
        raise ValueError(f"Unknown effect_filter: {effect_filter}")
    per_gene = v.groupby("gene_name")["PtID"].nunique()
    return (per_gene / len(patient_ids)).sort_values(ascending=False)


def build_features(
    xlsx_path: str | Path = DEFAULT_XLSX,
    top_n_genes: int = 20,
    min_prevalence: float | None = None,
    effect_filter: str = "non-synonymous",
    gene_selection_cohort: str = "both",
) -> tuple[pd.DataFrame, list[str]]:
    """Build the PBMF-style feature matrix.

    Clinical features (from the paper): BAGE, SEX (binary M), ECOGGR, blSLD,
    METSITES, histology (binary squamous), ever-smoker, MSAF, btmb.

    Mutation features: top-N genes by patient-level prevalence (binary 0/1,
    any variant counts) — gene list is selected on the BEP patients of
    ``gene_selection_cohort`` ("POPLAR", "OAK", or "both"), then applied to
    both trials so a single model trains on one feature definition.

    Returns the per-patient feature dataframe (BEP only) and the list of
    selected gene symbols.
    """
    poplar = load_clinical("POPLAR", xlsx_path)
    oak    = load_clinical("OAK",    xlsx_path)
    variants = load_variants(xlsx_path)

    shared_cols = [c for c in poplar.columns if c in oak.columns]
    clinical = pd.concat([poplar[shared_cols], oak[shared_cols]], ignore_index=True)
    clinical = clinical[clinical["BEP"] == "Y"].copy()

    # Gene selection on the chosen cohort
    if gene_selection_cohort == "both":
        sel_ids = set(clinical["PtID"])
    else:
        sel_ids = set(clinical.loc[clinical["Trial"] == gene_selection_cohort, "PtID"])
    prev = _gene_prevalence(variants, sel_ids, effect_filter=effect_filter)
    if min_prevalence is not None:
        prev = prev[prev >= min_prevalence]
    genes = list(prev.head(top_n_genes).index)

    # Patient × gene binary matrix (any variant of the chosen class)
    v = variants.copy()
    if effect_filter == "non-synonymous":
        v = v[v["effect"] != "synonymous"]
    elif effect_filter == "non-syn-non-driver":
        v = v[(v["effect"] != "synonymous") & (v["omitted_driver_mutation"] != "yes")]
    v = v[v["gene_name"].isin(genes)]
    mut = (v.assign(_one=1)
             .pivot_table(index="PtID", columns="gene_name",
                          values="_one", aggfunc="max", fill_value=0)
             .reindex(columns=genes, fill_value=0)
             .astype(int)
             .add_prefix("mut_"))

    # Clinical feature encoding
    feat = pd.DataFrame(index=clinical["PtID"].values)
    feat["Trial"]   = clinical["Trial"].values
    feat["TRT01P"]  = clinical["TRT01P"].values
    feat["OS"]      = clinical["OS"].values
    feat["OS_event"] = clinical["OS_event"].values
    feat["PFS"]     = clinical["PFS"].values
    feat["PFS_event"] = clinical["PFS_event"].values
    feat["age"]     = clinical["BAGE"].values
    feat["sex_M"]   = (clinical["SEX"].values == "M").astype(int)
    feat["ecog"]    = clinical["ECOGGR"].values
    feat["blSLD"]   = clinical["blSLD"].values
    feat["metsites"] = clinical["METSITES"].values
    feat["hist_squamous"] = (clinical["HIST"].values == "SQUAMOUS").astype(int)
    # TOBHX values are CURRENT / PREVIOUS / NEVER — PBMF paper uses smoking history
    # as a single covariate; collapse to ever-smoker (CURRENT or PREVIOUS).
    feat["ever_smoker"] = clinical["TOBHX"].isin(["CURRENT", "PREVIOUS"]).astype(int).values
    feat["msaf"]    = clinical["MSAF"].values
    feat["btmb"]    = clinical["btmb"].values
    feat.index.name = "PtID"

    feat = feat.join(mut, how="left").fillna({c: 0 for c in mut.columns})
    for c in mut.columns:
        feat[c] = feat[c].astype(int)

    return feat.reset_index(), genes


CLINICAL_FEATURE_COLS = [
    "age", "sex_M", "ecog", "blSLD", "metsites",
    "hist_squamous", "ever_smoker", "msaf", "btmb",
]
