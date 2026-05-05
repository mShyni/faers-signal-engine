# src/06_rq3_random_forest.py
#
# Step 6: Research Question 3 - Random Forest classifier vs OMOP reference set.
#
# Hypothesis (H3a): Random Forest achieves 5-fold stratified CV AUC >= 0.85,
#                   exceeding the best single disproportionality method by an
#                   absolute AUC difference >= 0.03 (bootstrap p < 0.05).
#                   Permutation feature importance ranks EBGM and IC as dominant.
#

# About the OMOP reference file:
#   Need a CSV at data/reference/omop_reference.csv with columns:
#     ingredient (uppercase RxNorm), pt (MedDRA Preferred Term, Title Case), label (0 or 1)


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score
)
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
REF_DIR = Path("data/reference")
FIG_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
N_FOLDS = 5


def load_inputs():
    cand_path = PROCESSED_DIR / "candidate_set.parquet"
    omop_path = REF_DIR / "omop_reference.csv"
    if not cand_path.exists():
        raise FileNotFoundError(f"{cand_path} not found.")
    if not omop_path.exists():
        raise FileNotFoundError(
            f"{omop_path} not found. Create the OMOP reference CSV at this path "
            "(columns: ingredient, pt, label)."
        )
    candidates = pd.read_parquet(cand_path)
    omop = pd.read_csv(omop_path)

    # Normalize join keys
    omop["ingredient"] = omop["ingredient"].str.strip().str.upper()
    omop["pt"] = omop["pt"].str.strip().str.title()
    candidates["ingredient"] = candidates["ingredient"].str.strip().str.upper()
    candidates["pt"] = candidates["pt"].str.strip().str.title()
    return candidates, omop


def merge_with_omop(candidates, omop):
    """
    Inner-join the candidate set with the OMOP labels. Pairs in OMOP that are
    NOT in the candidate set (because a < 3 in FAERS) are reported as a separate
    'reduction in cases' figure - this is the testable-subset constraint
    explicitly framed as a feature of RQ3 in the interim report.
    """
    n_pos_full = (omop["label"] == 1).sum()
    n_neg_full = (omop["label"] == 0).sum()
    print(f"OMOP reference (full): {n_pos_full} positives + {n_neg_full} negatives")

    merged = candidates.merge(omop, on=["ingredient", "pt"], how="inner")
    n_pos_test = (merged["label"] == 1).sum()
    n_neg_test = (merged["label"] == 0).sum()
    print(f"OMOP testable subset (after inner join with FAERS candidates):")
    print(f"  positives = {n_pos_test} ({n_pos_test/max(n_pos_full,1)*100:.1f}% of full set)")
    print(f"  negatives = {n_neg_test} ({n_neg_full and n_neg_test/n_neg_full*100:.1f}% of full set)")

    # Save the testable subset summary - this is the 'reduction in cases'
    summary = pd.DataFrame([
        {"set": "full", "positives": n_pos_full, "negatives": n_neg_full},
        {"set": "testable", "positives": n_pos_test, "negatives": n_neg_test},
    ])
    summary.to_csv(TABLE_DIR / "rq3_omop_testable_subset.csv", index=False)
    return merged


def build_feature_matrix(merged):
    """
    Six features per the interim report:
      log_ror, log_prr, ic, log_ebgm, log(observed), log(expected)
    """
    merged = merged.copy()
    merged["log_a"] = np.log(merged["a"].clip(lower=1))
    merged["log_expected"] = np.log(merged["expected"].clip(lower=1e-9))
    feature_cols = ["log_ror", "log_prr", "ic", "log_ebgm", "log_a", "log_expected"]
    X = merged[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    y = merged.loc[X.index, "label"].astype(int)
    return X, y, feature_cols, merged.loc[X.index]


def cv_auc(model, X, y, n_folds=N_FOLDS):
    """
    Stratified K-fold cross-validated AUC. Uses cross_val_predict with
    method='predict_proba' to get out-of-fold probabilities, then computes
    a single AUC over all out-of-fold predictions.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return roc_auc_score(y, proba), proba


def single_method_auc(scores, y):
    """AUC when using a single disproportionality score directly as the classifier."""
    return roc_auc_score(y, scores)


def bootstrap_auc_diff(y, p_rf, p_baseline, n=N_BOOTSTRAP):
    """
    Paired bootstrap p-value for the AUC difference (RF - baseline).
    Resample observation indices with replacement, recompute AUCs each time.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_obs = len(y)
    diffs = np.empty(n)
    y_arr = y.values
    for i in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        # Need both classes in the resample, otherwise AUC is undefined
        if len(np.unique(y_arr[idx])) < 2:
            diffs[i] = np.nan
            continue
        diffs[i] = roc_auc_score(y_arr[idx], p_rf[idx]) - roc_auc_score(y_arr[idx], p_baseline[idx])
    diffs = diffs[~np.isnan(diffs)]
    # Two-sided p-value: fraction of bootstrap samples where the difference
    # is on the opposite side of zero from the observed difference
    observed_diff = roc_auc_score(y, p_rf) - roc_auc_score(y, p_baseline)
    if observed_diff > 0:
        p_value = (diffs <= 0).mean()
    else:
        p_value = (diffs >= 0).mean()
    return observed_diff, 2 * min(p_value, 1 - p_value), diffs


def metrics_at_threshold(y, p, thresh=0.5):
    """Sensitivity, specificity, PPV at a probability cutoff."""
    pred = (p >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    return sensitivity, specificity, ppv


def main():
    print("Loading inputs...")
    candidates, omop = load_inputs()
    print(f"Candidate set: {len(candidates):,} pairs")
    print(f"OMOP reference: {len(omop):,} labelled pairs")

    print("\nMerging with OMOP reference...")
    merged = merge_with_omop(candidates, omop)
    if len(merged) < 30:
        print("WARNING: very small testable subset (< 30 pairs). Results unreliable.")

    X, y, feature_cols, merged_features = build_feature_matrix(merged)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"  Class distribution: {dict(y.value_counts())}")

    # ---------- Random Forest ----------
    print("\nFitting Random Forest with 5-fold CV...")
    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_auc, rf_proba = cv_auc(rf, X, y)
    rf_sens, rf_spec, rf_ppv = metrics_at_threshold(y, rf_proba)
    print(f"  RF cross-validated AUC: {rf_auc:.4f}")
    print(f"  RF sensitivity: {rf_sens:.3f}, specificity: {rf_spec:.3f}, PPV: {rf_ppv:.3f}")

    # ---------- Logistic regression baseline ----------
    print("\nFitting logistic regression baseline...")
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced",
                                      random_state=RANDOM_STATE)),
    ])
    lr_auc, lr_proba = cv_auc(logreg, X, y)
    lr_sens, lr_spec, lr_ppv = metrics_at_threshold(y, lr_proba)
    print(f"  Logistic regression CV AUC: {lr_auc:.4f}")

    # ---------- Single-method baselines ----------
    print("\nSingle-method AUCs (each disproportionality score alone):")
    single_aucs = {}
    for col, label in [("log_ror", "ROR"), ("log_prr", "PRR"),
                        ("ic", "IC"), ("log_ebgm", "EBGM")]:
        single_aucs[label] = single_method_auc(X[col].values, y)
        print(f"  {label}: AUC = {single_aucs[label]:.4f}")

    best_single_method = max(single_aucs, key=single_aucs.get)
    best_single_auc = single_aucs[best_single_method]
    print(f"  Best single method: {best_single_method} (AUC = {best_single_auc:.4f})")

    # ---------- Bootstrap test: RF vs best single method ----------
    print(f"\nBootstrap AUC difference test: RF vs {best_single_method}...")
    best_method_col = {"ROR": "log_ror", "PRR": "log_prr",
                       "IC": "ic", "EBGM": "log_ebgm"}[best_single_method]
    best_proba = X[best_method_col].values
    # Normalize to [0,1] so AUC is computed on a comparable score scale
    # (AUC is rank-invariant, so this is just to keep the array tidy)
    best_proba_norm = (best_proba - best_proba.min()) / (best_proba.max() - best_proba.min() + 1e-12)

    obs_diff, p_diff, _ = bootstrap_auc_diff(y, rf_proba, best_proba_norm)
    print(f"  Observed AUC difference (RF - {best_single_method}): {obs_diff:+.4f}")
    print(f"  Bootstrap two-sided p-value: {p_diff:.4f}")

    # ---------- Decision ----------
    auc_threshold_met = rf_auc >= 0.85
    margin_met = obs_diff >= 0.03 and p_diff < 0.05
    h3a_accepted = auc_threshold_met and margin_met
    print(f"\n[H3 decision]")
    print(f"  RF AUC >= 0.85: {auc_threshold_met} (AUC = {rf_auc:.4f})")
    print(f"  AUC margin >= 0.03 with p < 0.05: {margin_met}")
    print(f"  H3a accepted: {h3a_accepted}")

    # ---------- Permutation importance (on a single train) ----------
    print("\nComputing permutation feature importance...")
    rf.fit(X, y)
    importance = permutation_importance(
        rf, X, y, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    print(importance_df)
    importance_df.to_csv(TABLE_DIR / "rq3_permutation_importance.csv", index=False)

    # ---------- Save results ----------
    results = pd.DataFrame([{
        "rf_cv_auc": round(rf_auc, 4),
        "rf_sensitivity": round(rf_sens, 3),
        "rf_specificity": round(rf_spec, 3),
        "rf_ppv": round(rf_ppv, 3),
        "logreg_cv_auc": round(lr_auc, 4),
        **{f"single_auc_{k.lower()}": round(v, 4) for k, v in single_aucs.items()},
        "best_single_method": best_single_method,
        "best_single_auc": round(best_single_auc, 4),
        "auc_diff_rf_vs_best_single": round(obs_diff, 4),
        "bootstrap_p_value": round(p_diff, 4),
        "rf_auc_geq_0.85": bool(auc_threshold_met),
        "auc_margin_geq_0.03_p_lt_0.05": bool(margin_met),
        "h3a_accepted": bool(h3a_accepted),
        "n_testable": len(y),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
    }])
    results.to_csv(TABLE_DIR / "rq3_random_forest_results.csv", index=False)

    # ---------- ROC curves ----------
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba_arr, color in [
        ("Random Forest", rf_proba, "#4472C4"),
        ("Logistic Reg.", lr_proba, "#ED7D31"),
        (f"{best_single_method} (single)", best_proba_norm, "#70AD47"),
    ]:
        fpr, tpr, _ = roc_curve(y, proba_arr)
        auc = roc_auc_score(y, proba_arr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("RQ3: ROC curves on OMOP testable subset")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq3_roc_curves.png")
    plt.close()
    print("Saved ROC plot to outputs/figures/rq3_roc_curves.png")

    # ---------- Permutation importance bar plot ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    imp_sorted = importance_df.sort_values("importance_mean")
    ax.barh(imp_sorted["feature"], imp_sorted["importance_mean"],
            xerr=imp_sorted["importance_std"], color="#7030A0")
    ax.set_xlabel("Permutation importance (mean decrease in accuracy)")
    ax.set_title("RQ3: Permutation feature importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq3_permutation_importance.png")
    plt.close()
    print("Saved importance plot to outputs/figures/rq3_permutation_importance.png")

    print("\nRQ3 complete.")


if __name__ == "__main__":
    main()
