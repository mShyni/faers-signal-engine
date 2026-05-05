# src/05_rq2_gmm.py
#
# Step 5: Research Question 2 - Gaussian Mixture Model on the four-method score space.
#
# Hypothesis (H2a): A BIC-selected GMM identifies a separable high-signal component
#                   whose ranking by posterior probability agrees with the EBGM rank
#                   (Spearman rho > 0.85), supporting robust prioritization.
#
# Decision criteria (from interim report):
#   - Component count chosen by minimum BIC across k = 1..6
#   - High-signal component is the one with the highest joint mean across all
#     four standardized features
#   - That component must be small (<= 10% of pairs) and have within-component
#     means exceeding the population by >= 1.5 SD on each method
#   - Silhouette score >= 0.30
#   - Ranking by posterior probability of high-signal membership must reach
#     Spearman rho > 0.85 with EBGM rank
#
# What this script does:
#   1. Load the candidate set
#   2. Standardize the four log-transformed disproportionality scores
#   3. Fit GMM for k = 1..6 components, pick best by BIC
#   4. Compute silhouette on a sample (silhouette is O(n^2) - sample for speed)
#   5. Identify the high-signal component
#   6. Compute Spearman correlation between posterior probability and EBGM rank
#   7. Compute the cross-method concordance correlation matrix (Table for Table 11)
#   8. Save outputs and figures

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
FIG_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
K_RANGE = range(1, 7)            # Try 1 to 6 components
SILHOUETTE_SAMPLE_SIZE = 20000   # silhouette is slow on large data


def load_candidates():
    path = PROCESSED_DIR / "candidate_set.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 03_feature_engineering.py first.")
    return pd.read_parquet(path)


def main():
    print("Loading candidate set...")
    df = load_candidates()
    print(f"Loaded {len(df):,} pairs")

    feature_cols = ["log_ror", "log_prr", "ic", "log_ebgm"]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    valid_idx = X.index
    print(f"Pairs with finite scores: {len(X):,}")

    # ---------- Cross-method concordance (RQ2 motivation) ----------
    # Spearman rho between every pair of methods. Confirms / quantifies the high
    # frequentist concordance and lower frequentist-vs-Bayesian concordance noted
    # in the interim report.
    print("\nComputing cross-method Spearman correlations...")
    methods = ["ror", "prr", "ic", "ebgm"]
    corr = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for m1 in methods:
        for m2 in methods:
            r, _ = spearmanr(df[m1], df[m2], nan_policy="omit")
            corr.loc[m1, m2] = round(r, 3)
    print(corr)
    corr.to_csv(TABLE_DIR / "rq2_method_concordance.csv")

    # Heatmap of the concordance matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values.astype(float), cmap="Blues", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_yticklabels([m.upper() for m in methods])
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title("RQ2: Cross-method Spearman concordance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq2_concordance_heatmap.png")
    plt.close()
    print("Saved concordance heatmap to outputs/figures/rq2_concordance_heatmap.png")

    # ---------- Standardize features for GMM ----------
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # ---------- Fit GMM for each k, select by BIC ----------
    print("\nFitting GMM for each k and selecting by BIC...")
    bic_scores = {}
    fitted_models = {}
    for k in K_RANGE:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=RANDOM_STATE,
            n_init=3,
            max_iter=200,
        )
        gmm.fit(X_scaled)
        bic_scores[k] = gmm.bic(X_scaled)
        fitted_models[k] = gmm
        print(f"  k = {k}, BIC = {bic_scores[k]:,.0f}")

    best_k = min(bic_scores, key=bic_scores.get)
    print(f"\nBest k by BIC: {best_k}")
    best_model = fitted_models[best_k]

    # BIC plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(bic_scores.keys()), list(bic_scores.values()),
            marker="o", color="#4472C4")
    ax.axvline(best_k, color="red", linestyle="--", label=f"Best k = {best_k}")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title("RQ2: BIC across GMM component counts")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq2_bic_curve.png")
    plt.close()
    print("Saved BIC curve to outputs/figures/rq2_bic_curve.png")

    # ---------- Predict cluster labels and posterior probabilities ----------
    labels = best_model.predict(X_scaled)
    posteriors = best_model.predict_proba(X_scaled)

    # ---------- Identify the high-signal component ----------
    # The component with the highest sum of standardized means is the
    # high-signal cluster.
    component_means = best_model.means_
    high_signal_comp = int(np.argmax(component_means.sum(axis=1)))
    print(f"\nHigh-signal component: {high_signal_comp}")
    print(f"  Component means (standardized): {component_means[high_signal_comp].round(2)}")

    # Check: each method mean exceeds population (which is 0 in standardized space) by 1.5 SD
    methods_pass = (component_means[high_signal_comp] >= 1.5).all()
    print(f"  All four methods >= 1.5 SD above population: {methods_pass}")

    # Component size as a fraction of all pairs
    high_signal_size = (labels == high_signal_comp).sum() / len(labels)
    size_pass = high_signal_size <= 0.10
    print(f"  High-signal component size: {high_signal_size:.2%} ({size_pass=})")

    # ---------- Silhouette ----------
    n_for_sil = min(SILHOUETTE_SAMPLE_SIZE, len(X_scaled))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_scaled), size=n_for_sil, replace=False)
    if best_k > 1:
        sil = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
    else:
        sil = float("nan")  # silhouette undefined for k=1
    print(f"  Silhouette (sample of {n_for_sil:,}): {sil:.3f}")
    sil_pass = (sil >= 0.30) if not np.isnan(sil) else False

    # ---------- Posterior-probability rank vs EBGM rank ----------
    # Probability of belonging to the high-signal component
    p_high_signal = posteriors[:, high_signal_comp]

    df["gmm_label"] = -1
    df.loc[valid_idx, "gmm_label"] = labels
    df["p_high_signal"] = np.nan
    df.loc[valid_idx, "p_high_signal"] = p_high_signal

    sub = df.loc[valid_idx, ["p_high_signal", "ebgm"]].copy()
    rho_pe, p_pe = spearmanr(sub["p_high_signal"], sub["ebgm"])
    print(f"\n[H2 test] Spearman rho between posterior probability and EBGM:")
    print(f"  rho = {rho_pe:.4f}, p = {p_pe:.2e}")
    rank_pass = rho_pe > 0.85

    # ---------- Decision ----------
    h2a_accepted = methods_pass and size_pass and sil_pass and rank_pass
    print(f"\n[H2 decision]")
    print(f"  All means >= 1.5 SD above pop:   {methods_pass}")
    print(f"  Component size <= 10%:           {size_pass}")
    print(f"  Silhouette >= 0.30:              {sil_pass}")
    print(f"  Rank correlation > 0.85:         {rank_pass}")
    print(f"  H2a accepted: {h2a_accepted}")

    # ---------- Save outputs ----------
    df.to_parquet(PROCESSED_DIR / "candidate_set_with_gmm.parquet", index=False)

    results = pd.DataFrame([{
        "best_k": best_k,
        "best_bic": bic_scores[best_k],
        "high_signal_component": high_signal_comp,
        "high_signal_component_size_pct": round(high_signal_size * 100, 2),
        "silhouette_score": round(sil, 3) if not np.isnan(sil) else None,
        "spearman_phigh_vs_ebgm": round(rho_pe, 4),
        "spearman_phigh_vs_ebgm_p": p_pe,
        "all_means_geq_1.5sd": bool(methods_pass),
        "component_size_leq_10pct": bool(size_pass),
        "silhouette_geq_0.30": bool(sil_pass),
        "rank_correlation_geq_0.85": bool(rank_pass),
        "h2a_accepted": bool(h2a_accepted),
    }])
    results.to_csv(TABLE_DIR / "rq2_gmm_results.csv", index=False)

    # Save component means
    component_means_df = pd.DataFrame(
        best_model.means_, columns=feature_cols
    )
    component_means_df["weight"] = best_model.weights_
    component_means_df.index.name = "component"
    component_means_df.to_csv(TABLE_DIR / "rq2_component_means.csv")

    # Top 50 high-signal pairs by posterior probability
    top50 = (
        df.loc[valid_idx]
        .sort_values("p_high_signal", ascending=False)
        .head(50)[["ingredient", "pt", "a", "ror", "prr", "ic", "ebgm",
                   "p_high_signal", "gmm_label"]]
    )
    top50.to_csv(TABLE_DIR / "rq2_top50_high_signal.csv", index=False)

    print("\nRQ2 complete.")


if __name__ == "__main__":
    main()
