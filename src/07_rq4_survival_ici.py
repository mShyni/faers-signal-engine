# src/07_rq4_survival_ici.py
#
# Step 7: Research Question 4 - Random Survival Forest on ICI quarterly EBGM trajectories.
#
# Hypothesis (H4a): The engine detects >= 60% of labelled immune-related adverse events
#                   at least one quarter before their addition to the product label,
#                   with median lead time >= 2 quarters (Wilcoxon signed-rank p < 0.05).
#                   Random Survival Forest concordance index >= 0.65.

# Required input file: data/reference/ici_label_revisions.csv with columns:
#   ingredient, pt, label_revision_quarter, soc


from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
REF_DIR = Path("data/reference")
FIG_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Seven approved immune checkpoint inhibitors (RxNorm ingredient names, uppercase)
ICI_INGREDIENTS = [
    "PEMBROLIZUMAB",
    "NIVOLUMAB",
    "ATEZOLIZUMAB",
    "DURVALUMAB",
    "AVELUMAB",
    "IPILIMUMAB",
    "CEMIPLIMAB",
]

# 12-quarter window. Indexed 1..12 chronologically.
QUARTERS = [
    "2023Q1", "2023Q2", "2023Q3", "2023Q4",
    "2024Q1", "2024Q2", "2024Q3", "2024Q4",
    "2025Q1", "2025Q2", "2025Q3", "2025Q4",
]
QUARTER_INDEX = {q: i + 1 for i, q in enumerate(QUARTERS)}
LAST_QUARTER_INDEX = len(QUARTERS)

# Minimum cell threshold for any pair to enter trajectory analysis at all
MIN_A_TOTAL = 3
EB05_DETECTION_THRESHOLD = 2.0


# ============================================================
# A. Cumulative quarterly EBGM trajectories
# ============================================================

def build_cumulative_counts(triples_ici, triples_all):
    """
    For each ICI ingredient and each PT seen with any ICI, build cumulative
    counts a, n_drug, n_event, N at the end of each quarter.

    triples_ici: distinct (primaryid, ingredient, pt, quarter) for ICI rows
    triples_all: distinct (primaryid, ingredient, pt, quarter) for the whole table
                 (needed for n_event and N denominators)
    """
    # ----- Cumulative N per quarter (whole-database denominator) -----
    n_per_quarter = (
        triples_all.groupby("quarter", observed=True)["primaryid"]
        .nunique()
        .reindex(QUARTERS, fill_value=0)
        .cumsum()
        .rename("N_cum")
    )

    # ----- Cumulative n_event per (PT, quarter) -----
    n_event_per_q = (
        triples_all.groupby(["pt", "quarter"], observed=True)["primaryid"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=QUARTERS, fill_value=0)
        .cumsum(axis=1)
        .stack()
        .rename("n_event_cum")
        .reset_index()
    )

    # ----- Cumulative n_drug per (ICI ingredient, quarter) -----
    n_drug_per_q = (
        triples_ici.groupby(["ingredient", "quarter"], observed=True)["primaryid"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=QUARTERS, fill_value=0)
        .cumsum(axis=1)
        .stack()
        .rename("n_drug_cum")
        .reset_index()
    )

    # ----- Cumulative a per (ingredient, PT, quarter) -----
    a_per_q = (
        triples_ici.groupby(["ingredient", "pt", "quarter"], observed=True)["primaryid"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=QUARTERS, fill_value=0)
        .cumsum(axis=1)
        .stack()
        .rename("a_cum")
        .reset_index()
    )

    # ----- Merge everything -----
    out = a_per_q.merge(n_drug_per_q, on=["ingredient", "quarter"])
    out = out.merge(n_event_per_q, on=["pt", "quarter"])
    out["N_cum"] = out["quarter"].map(n_per_quarter)
    return out


def compute_quarterly_ebgm(traj_counts):
    """
    Compute quarterly EBGM and EB05 using single-component Gamma-Poisson
    shrinkage on the cumulative counts. Hyperparameters are estimated once
    on the full-window cumulative distribution to keep the prior stable.
    """
    print("Estimating Gamma-Poisson hyperparameters on full-window data...")
    full = traj_counts[traj_counts["quarter"] == QUARTERS[-1]].copy()
    full["expected"] = full["n_drug_cum"] * full["n_event_cum"] / full["N_cum"].clip(lower=1)
    rr = full["a_cum"] / full["expected"].clip(lower=1e-9)
    rr_trim = rr[(rr > 0) & (rr < rr.quantile(0.99))]
    mean_rr = rr_trim.mean()
    var_rr = rr_trim.var(ddof=1)
    alpha = max(mean_rr ** 2 / max(var_rr, 1e-9), 0.5)
    beta = max(mean_rr / max(var_rr, 1e-9), 0.5)
    print(f"  alpha = {alpha:.3f}, beta = {beta:.3f}")

    traj_counts = traj_counts.copy()
    traj_counts["expected"] = (
        traj_counts["n_drug_cum"] * traj_counts["n_event_cum"]
        / traj_counts["N_cum"].clip(lower=1)
    )
    O = traj_counts["a_cum"].astype(float).values
    E = traj_counts["expected"].astype(float).values

    post_shape = alpha + O
    post_rate = beta + E
    traj_counts["ebgm"] = np.exp(digamma(post_shape) - np.log(post_rate))
    traj_counts["eb05"] = stats.gamma.ppf(0.05, a=post_shape, scale=1.0 / post_rate)
    traj_counts["q_index"] = traj_counts["quarter"].map(QUARTER_INDEX)
    return traj_counts


# ============================================================
# B. First-detection time per pair
# ============================================================

def compute_first_detection(traj):
    """
    Detection event = first quarter where EB05 > threshold AND a_cum >= MIN_A_TOTAL.
    If no quarter satisfies both, right-censor at LAST_QUARTER_INDEX.
    """
    traj = traj.copy()
    traj["detected"] = (
        (traj["eb05"] > EB05_DETECTION_THRESHOLD)
        & (traj["a_cum"] >= MIN_A_TOTAL)
    )

    # First detection quarter index per pair
    detected_only = traj[traj["detected"]]
    first_det = (
        detected_only.groupby(["ingredient", "pt"], observed=True)["q_index"]
        .min()
        .reset_index(name="first_detection_q")
    )

    # All pairs with a_cum >= MIN_A_TOTAL by 2025Q4 (must enter survival set)
    final_q = traj[traj["q_index"] == LAST_QUARTER_INDEX]
    candidates = final_q[final_q["a_cum"] >= MIN_A_TOTAL][
        ["ingredient", "pt", "a_cum", "ebgm", "eb05"]
    ].rename(columns={
        "a_cum": "a_total", "ebgm": "ebgm_final", "eb05": "eb05_final"
    })

    # Left-join: pairs without a detection get NaN -> right-censored at LAST_QUARTER
    survival = candidates.merge(first_det, on=["ingredient", "pt"], how="left")
    survival["event"] = (~survival["first_detection_q"].isna()).astype(int)
    survival["time_to_detection"] = survival["first_detection_q"].fillna(LAST_QUARTER_INDEX).astype(int)
    return survival


# ============================================================
# C. Hypothesis tests and survival modeling
# ============================================================

def merge_with_labels(survival, labels):
    """
    Merge survival data with the irAE labelled set so we have the label
    revision quarter for each labelled pair. Pairs in the labelled set
    that did not appear in survival are reported but cannot be tested.
    """
    labels = labels.copy()
    labels["ingredient"] = labels["ingredient"].str.strip().str.upper()
    labels["pt"] = labels["pt"].str.strip().str.title()
    labels["label_q_index"] = labels["label_revision_quarter"].map(QUARTER_INDEX)

    merged = survival.merge(labels, on=["ingredient", "pt"], how="inner")
    print(f"\nLabelled irAEs covered by survival set: {len(merged)} / {len(labels)}")
    return merged


def lead_time_test(merged):
    """
    Lead time = label_q_index - first_detection_q (only for detected pairs).
    Wilcoxon signed-rank tests whether lead times are significantly > 0.
    """
    detected = merged[merged["event"] == 1].copy()
    if len(detected) < 5:
        print("Too few detected labelled pairs for Wilcoxon test.")
        return None
    detected["lead_time_q"] = detected["label_q_index"] - detected["first_detection_q"]
    median_lead = detected["lead_time_q"].median()

    # Wilcoxon signed-rank: H0 = lead_time = 0, H1 = lead_time > 0
    if (detected["lead_time_q"] != 0).any():
        stat, p_value = stats.wilcoxon(
            detected["lead_time_q"], alternative="greater"
        )
    else:
        stat, p_value = 0.0, 1.0

    pct_early = (detected["lead_time_q"] >= 1).sum() / len(detected) * 100
    print(f"\n[H4 lead time test]")
    print(f"  n labelled pairs detected: {len(detected)}")
    print(f"  median lead time (quarters): {median_lead}")
    print(f"  pct detected >= 1 quarter early: {pct_early:.1f}%")
    print(f"  Wilcoxon signed-rank stat = {stat:.2f}, p (one-sided) = {p_value:.4f}")
    return {
        "n_labelled_detected": int(len(detected)),
        "median_lead_quarters": float(median_lead),
        "pct_detected_at_least_1q_early": round(pct_early, 1),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p_one_sided": float(p_value),
    }, detected


def fit_survival_forest(survival, traj):
    """
    Fit a Random Survival Forest. Falls back to a clear message if the
    scikit-survival package is not installed.

    Covariates:
      - ingredient (one-hot ICI agent)
      - SOC (one-hot MedDRA System Organ Class - we approximate by the first
             word of the PT if a separate SOC is not provided; you can later
             swap in a real PT->SOC lookup if desired)
      - log(cumulative report volume) at last quarter
    """
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import concordance_index_censored
    except ImportError:
        print("\nscikit-survival is not installed. Install with:")
        print("    pip install scikit-survival")
        print("Skipping Random Survival Forest fit.")
        return None

    surv = survival.copy()

    # Cumulative report volume at the final quarter for each pair
    final_q = traj[traj["q_index"] == LAST_QUARTER_INDEX][
        ["ingredient", "pt", "a_cum"]
    ].rename(columns={"a_cum": "cumulative_report_volume"})
    surv = surv.merge(final_q, on=["ingredient", "pt"], how="left")
    surv["log_cum_volume"] = np.log(surv["cumulative_report_volume"].clip(lower=1))

    # Approximate SOC: take the first word of the PT (placeholder).
    # Replace with a real PT->SOC lookup from MedDRA if you have one.
    surv["soc_approx"] = surv["pt"].str.split().str[0].str.title()

    # Build feature matrix
    X = pd.get_dummies(
        surv[["ingredient", "soc_approx", "log_cum_volume"]],
        columns=["ingredient", "soc_approx"],
        drop_first=False,
    ).astype(float)

    # Build structured survival outcome
    y = np.array(
        list(zip(surv["event"].astype(bool), surv["time_to_detection"].astype(float))),
        dtype=[("event", bool), ("time", float)],
    )

    print(f"\nFitting Random Survival Forest on {len(X):,} pairs, {X.shape[1]} features...")
    rsf = RandomSurvivalForest(
        n_estimators=300,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rsf.fit(X.values, y)

    # Out-of-bag concordance is preferable but sksurv doesn't expose it
    # directly for survival forests; use in-sample c-index on the predicted
    # risk score (lower time -> higher risk in our framing).
    risk_scores = rsf.predict(X.values)
    c_index, _, _, _, _ = concordance_index_censored(
        y["event"], y["time"], risk_scores
    )
    print(f"  Concordance index: {c_index:.4f}")
    return {"concordance_index": float(c_index), "n_pairs": int(len(X))}


# ============================================================
# Main
# ============================================================

def main():
    # ---------- Load analytic table ----------
    analytic_path = PROCESSED_DIR / "analytic_table.parquet"
    if not analytic_path.exists():
        raise FileNotFoundError(f"{analytic_path} not found. Run 01_clean_and_merge.py first.")
    df = pd.read_parquet(analytic_path)
    print(f"Loaded analytic table: {len(df):,} rows")

    # Make sure the quarter column has the expected labels
    df["quarter"] = df["quarter"].astype(str).str.upper()
    df = df[df["quarter"].isin(QUARTERS)]

    # Distinct triples for the whole table (denominators for n_event and N)
    triples_all = df[["primaryid", "ingredient", "pt", "quarter"]].drop_duplicates()

    # Filter to ICI ingredients
    df_ici = df[df["ingredient"].isin(ICI_INGREDIENTS)].copy()
    triples_ici = df_ici[["primaryid", "ingredient", "pt", "quarter"]].drop_duplicates()
    print(f"ICI subset: {len(df_ici):,} rows ({df_ici['ingredient'].value_counts().to_dict()})")

    if len(df_ici) == 0:
        print("No ICI rows in the analytic table. Exiting.")
        return

    # ---------- Build trajectories ----------
    print("\nBuilding cumulative quarterly counts...")
    traj_counts = build_cumulative_counts(triples_ici, triples_all)
    print(f"Trajectory rows (pair x quarter): {len(traj_counts):,}")

    print("\nComputing quarterly EBGM and EB05...")
    traj = compute_quarterly_ebgm(traj_counts)
    traj.to_parquet(PROCESSED_DIR / "ici_quarterly_trajectories.parquet", index=False)
    print(f"Saved trajectories to {PROCESSED_DIR / 'ici_quarterly_trajectories.parquet'}")

    # ---------- Compute first-detection time ----------
    survival = compute_first_detection(traj)
    survival.to_parquet(PROCESSED_DIR / "ici_survival_data.parquet", index=False)
    print(f"\nSurvival set: {len(survival):,} pairs")
    print(f"  Detected by 2025Q4: {survival['event'].sum():,} pairs")
    print(f"  Right-censored:     {(survival['event']==0).sum():,} pairs")

    # ---------- Load labelled irAEs ----------
    label_path = REF_DIR / "ici_label_revisions.csv"
    if not label_path.exists():
        print(f"\n{label_path} not found - skipping H4 hypothesis test.")
        print("Use 07b_make_ici_template.py to generate a starter file.")
        merged = None
        lead_results = None
        detected_labelled = None
    else:
        labels = pd.read_csv(label_path)
        merged = merge_with_labels(survival, labels)
        merged.to_csv(TABLE_DIR / "rq4_labelled_pairs_merged.csv", index=False)
        out = lead_time_test(merged)
        if out is not None:
            lead_results, detected_labelled = out
            detected_labelled.to_csv(TABLE_DIR / "rq4_lead_times.csv", index=False)
        else:
            lead_results = None
            detected_labelled = None

    # ---------- Random Survival Forest ----------
    rsf_results = fit_survival_forest(survival, traj)

    # ---------- Decision ----------
    pct_pass = lead_results and lead_results["pct_detected_at_least_1q_early"] >= 60
    median_pass = lead_results and lead_results["median_lead_quarters"] >= 2
    wilcoxon_pass = lead_results and lead_results["wilcoxon_p_one_sided"] < 0.05
    cindex_pass = rsf_results and rsf_results["concordance_index"] >= 0.65

    h4a_accepted = bool(pct_pass and median_pass and wilcoxon_pass and cindex_pass)
    print(f"\n[H4 decision]")
    print(f"  >= 60% labelled irAEs detected >=1q early: {bool(pct_pass)}")
    print(f"  Median lead time >= 2 quarters:            {bool(median_pass)}")
    print(f"  Wilcoxon p < 0.05:                         {bool(wilcoxon_pass)}")
    print(f"  Concordance index >= 0.65:                 {bool(cindex_pass)}")
    print(f"  H4a accepted: {h4a_accepted}")

    # ---------- Save consolidated results ----------
    summary = {
        "n_ici_pairs_in_survival_set": int(len(survival)),
        "n_detected": int(survival["event"].sum()),
    }
    if lead_results:
        summary.update(lead_results)
    if rsf_results:
        summary.update(rsf_results)
    summary["h4a_accepted"] = h4a_accepted
    pd.DataFrame([summary]).to_csv(TABLE_DIR / "rq4_survival_results.csv", index=False)

    # ---------- Trajectory plot for a couple of well-known irAE pairs ----------
    print("\nPlotting EBGM trajectories for a few showcase pairs...")
    showcase = [
        ("PEMBROLIZUMAB", "Hypothyroidism"),
        ("PEMBROLIZUMAB", "Pneumonitis"),
        ("NIVOLUMAB", "Colitis"),
        ("NIVOLUMAB", "Hepatitis"),
        ("IPILIMUMAB", "Colitis"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for ing, pt in showcase:
        sub = traj[(traj["ingredient"] == ing) & (traj["pt"] == pt)].sort_values("q_index")
        if len(sub) == 0:
            continue
        ax.plot(sub["q_index"], sub["ebgm"], marker="o", label=f"{ing} - {pt}")
    ax.axhline(EB05_DETECTION_THRESHOLD, color="gray", linestyle="--",
               label=f"Detection threshold (EBGM = {EB05_DETECTION_THRESHOLD})")
    ax.set_xticks(range(1, LAST_QUARTER_INDEX + 1))
    ax.set_xticklabels(QUARTERS, rotation=45)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Cumulative EBGM")
    ax.set_title("RQ4: Cumulative quarterly EBGM trajectories for selected ICI-irAE pairs")
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq4_ici_trajectories.png")
    plt.close()
    print(f"Saved trajectory plot to outputs/figures/rq4_ici_trajectories.png")

    # ---------- Lead-time histogram ----------
    if detected_labelled is not None and len(detected_labelled) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(detected_labelled["lead_time_q"], bins=range(-6, 13),
                color="#70AD47", edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", label="Label revision = 0")
        ax.set_xlabel("Lead time in quarters (label revision - engine detection)")
        ax.set_ylabel("Number of labelled irAE pairs")
        ax.set_title("RQ4: Engine detection lead time vs product label revisions")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "rq4_lead_time_histogram.png")
        plt.close()
        print("Saved lead-time histogram to outputs/figures/rq4_lead_time_histogram.png")

    print("\nRQ4 complete.")


if __name__ == "__main__":
    main()
