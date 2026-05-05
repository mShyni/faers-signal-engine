# src/04_rq1_isolation_forest.py
#
# Step 4: Research Question 1 - Isolation Forest on the four disproportionality scores.
#
# Hypothesis (H1a): The unsupervised anomaly score recovers the same robust consensus
#                   signal subset that conventional disproportionality flags identify.
#
# Decision criteria (from interim report):
#   - H1a accepted if Spearman rho >= 0.60 between anomaly score and conventional
#     flag count (p < 0.001)
#   - AND Cohen's kappa >= 0.60 between top decile of anomaly score and consensus flag
#
# What this script does:
#   1. Load the candidate set from step 3
#   2. Take the four disproportionality scores: log(ROR), log(PRR), IC, log(EBGM)
#   3. Standardize them
#   4. Fit an Isolation Forest (no labels needed - this is unsupervised)
#   5. Get the anomaly score for every pair (higher = more anomalous = stronger signal)
#   6. Compare to the conventional flag count via Spearman correlation
#   7. Compare top-decile anomalies to consensus flag via Cohen's kappa
#   8. Save scored output and a small results table

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
FIG_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
RANDOM_STATE = 42


def load_candidates():
    path = PROCESSED_DIR / "candidate_set.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 03_feature_engineering.py first."
        )
    return pd.read_parquet(path)


def main():
    print("Loading candidate set...")
    df = load_candidates()
    print(f"Loaded {len(df):,} pairs")

    # ---------- Build feature matrix ----------
    # The four disproportionality scores. Log-transform skewed measures so
    # that the Gaussian assumption behind StandardScaler is more reasonable.
    feature_cols = ["log_ror", "log_prr", "ic", "log_ebgm"]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Pairs after dropping non-finite scores: {len(X):,}")

    # Track which rows of the original df correspond to X
    valid_idx = X.index

    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # ---------- Fit Isolation Forest ----------
    # contamination='auto' lets sklearn pick a default; we are interested in
    # the continuous anomaly score, not the binary classification.
    print("Fitting Isolation Forest (this can take a minute on large candidate sets)...")
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    # In sklearn, score_samples returns higher values for "normal" points.
    # We want higher = more anomalous, so we negate.
    anomaly_score = -iso.score_samples(X_scaled)

    # Store back in the original dataframe
    df["anomaly_score"] = np.nan
    df.loc[valid_idx, "anomaly_score"] = anomaly_score

    # ---------- Hypothesis test 1: Spearman correlation ----------
    sub = df.loc[valid_idx, ["anomaly_score", "flag_count", "flag_consensus"]].copy()
    rho, p_value = spearmanr(sub["anomaly_score"], sub["flag_count"])
    print(f"\n[H1 test 1] Spearman correlation between anomaly score and flag count")
    print(f"  rho = {rho:.4f}, p = {p_value:.2e}")

    # ---------- Hypothesis test 2: Cohen's kappa ----------
    # Top decile of anomaly score -> "high anomaly" binary indicator
    threshold = sub["anomaly_score"].quantile(0.90)
    sub["top_decile"] = (sub["anomaly_score"] >= threshold).astype(int)
    kappa = cohen_kappa_score(sub["top_decile"], sub["flag_consensus"])
    print(f"\n[H1 test 2] Cohen's kappa between anomaly top decile and consensus flag")
    print(f"  kappa = {kappa:.4f}")

    # ---------- Decision ----------
    decision_rho = (rho >= 0.60) and (p_value < 0.001)
    decision_kappa = kappa >= 0.60
    h1a_accepted = decision_rho and decision_kappa
    print(f"\n[H1 decision]")
    print(f"  Spearman criterion (rho >= 0.60, p < 0.001): {decision_rho}")
    print(f"  Cohen's kappa criterion (kappa >= 0.60):     {decision_kappa}")
    print(f"  H1a accepted: {h1a_accepted}")

    # ---------- Save outputs ----------
    df.to_parquet(PROCESSED_DIR / "candidate_set_with_anomaly.parquet", index=False)

    results = pd.DataFrame([{
        "n_pairs": len(sub),
        "spearman_rho": round(rho, 4),
        "spearman_p": p_value,
        "cohens_kappa": round(kappa, 4),
        "rho_criterion_met": decision_rho,
        "kappa_criterion_met": decision_kappa,
        "h1a_accepted": h1a_accepted,
        "top_decile_threshold": round(float(threshold), 4),
    }])
    results.to_csv(TABLE_DIR / "rq1_isolation_forest_results.csv", index=False)

    # Top-N anomalies for inspection (sanity check)
    top_n = (
        df.loc[valid_idx]
        .sort_values("anomaly_score", ascending=False)
        .head(50)[["ingredient", "pt", "a", "ror", "prr", "ic", "ebgm",
                   "flag_count", "anomaly_score"]]
    )
    top_n.to_csv(TABLE_DIR / "rq1_top50_anomalies.csv", index=False)

    # ---------- Visualization 1: anomaly score vs flag count ----------
    fig, ax = plt.subplots(figsize=(8, 6))
    # Add some jitter to flag_count for visibility
    jitter = np.random.uniform(-0.15, 0.15, size=len(sub))
    ax.scatter(sub["flag_count"] + jitter, sub["anomaly_score"],
               alpha=0.05, s=5, color="#4472C4")
    ax.set_xlabel("Number of conventional methods flagging the pair (0-4)")
    ax.set_ylabel("Isolation Forest anomaly score")
    ax.set_title(f"RQ1: Anomaly score vs conventional flag count\n"
                 f"Spearman rho = {rho:.3f}, p = {p_value:.2e}")
    ax.axhline(threshold, color="red", linestyle="--",
               label=f"Top-decile threshold ({threshold:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq1_anomaly_vs_flagcount.png")
    plt.close()
    print(f"\nSaved scatter plot to outputs/figures/rq1_anomaly_vs_flagcount.png")

    # ---------- Visualization 2: anomaly score distribution ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sub["anomaly_score"], bins=80, color="#70AD47", edgecolor="white")
    ax.axvline(threshold, color="red", linestyle="--",
               label=f"Top decile cutoff ({threshold:.3f})")
    ax.set_xlabel("Isolation Forest anomaly score")
    ax.set_ylabel("Number of pairs")
    ax.set_title("RQ1: Anomaly score distribution across all candidate pairs")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq1_anomaly_score_distribution.png")
    plt.close()
    print("Saved histogram to outputs/figures/rq1_anomaly_score_distribution.png")

    print("\nRQ1 complete.")


if __name__ == "__main__":
    main()
