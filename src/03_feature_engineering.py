# src/03_feature_engineering.py
#
# Step 3: Build the (ingredient, PT) candidate set and compute four disproportionality scores.
#

# About the EBGM implementation:
#   The R package openEBGM uses a 2-component Gamma mixture prior (DuMouchel 1999).
#   This script uses a SIMPLIFIED single-component Gamma prior with hyperparameters
#   estimated by method of moments on the empirical relative reporting ratio (RR).


from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma

PROCESSED_DIR = Path("data/processed")
TABLE_DIR = Path("outputs/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Minimum-cell threshold (Evans et al., 2001)
MIN_A = 3


def load_analytic():
    path = PROCESSED_DIR / "analytic_table.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 01_clean_and_merge.py first.")
    return pd.read_parquet(path)


# ============================================================
# Step A: Build contingency table counts for every (drug, PT) pair
# ============================================================

def build_contingency(df):
    """
    Build the 2x2 contingency table values (a, b, c, d) for every (ingredient, pt) pair.

    Definitions (per Mouffak et al. 2023, standard pharmacovigilance):
        a = number of distinct reports with this drug AND this PT
        b = number of distinct reports with this drug AND any other PT
        c = number of distinct reports with any other drug AND this PT
        d = number of distinct reports with neither
        N = total number of distinct reports in the analytic table
        E (expected) = (a + b) * (a + c) / N

    Returns a DataFrame with one row per pair plus the count columns.
    """
    print("Building distinct (report, drug, PT) triples...")
    triples = df[["primaryid", "ingredient", "pt"]].drop_duplicates()
    print(f"  Distinct triples: {len(triples):,}")

    N = triples["primaryid"].nunique()
    print(f"  Total distinct reports (N): {N:,}")

    # a: reports per (drug, pt) pair
    print("Counting reports per (drug, PT) pair...")
    pair_counts = (
        triples.groupby(["ingredient", "pt"], observed=True)["primaryid"]
        .nunique()
        .reset_index(name="a")
    )
    print(f"  Total candidate pairs before threshold: {len(pair_counts):,}")

    pair_counts = pair_counts[pair_counts["a"] >= MIN_A].copy()
    print(f"  Pairs after a >= {MIN_A} threshold: {len(pair_counts):,}")

    # n_drug: reports per ingredient
    print("Counting reports per ingredient and per PT...")
    n_drug = (
        triples.groupby("ingredient", observed=True)["primaryid"]
        .nunique()
        .reset_index(name="n_drug")
    )
    n_event = (
        triples.groupby("pt", observed=True)["primaryid"]
        .nunique()
        .reset_index(name="n_event")
    )

    # Merge
    out = pair_counts.merge(n_drug, on="ingredient").merge(n_event, on="pt")
    out["N"] = N
    out["b"] = out["n_drug"] - out["a"]
    out["c"] = out["n_event"] - out["a"]
    out["d"] = out["N"] - out["n_drug"] - out["n_event"] + out["a"]
    out["expected"] = out["n_drug"] * out["n_event"] / out["N"]

    return out


# ============================================================
# Step B: Compute the four disproportionality measures
# ============================================================

def compute_ror(df):
    """
    Reporting Odds Ratio with 95% confidence interval.
        ROR  = (a * d) / (b * c)
        SE(log ROR) = sqrt(1/a + 1/b + 1/c + 1/d)
        95% CI = exp(log ROR +/- 1.96 * SE)
    Signal threshold: ROR >= 2 AND lower 95% CI > 1.
    """
    a, b, c, d = df["a"], df["b"], df["c"], df["d"]
    # Avoid divide-by-zero by clipping; cells with b=0 or c=0 mean perfect signals
    b_safe = b.clip(lower=0.5)
    c_safe = c.clip(lower=0.5)
    d_safe = d.clip(lower=0.5)
    a_safe = a.clip(lower=0.5)

    df["ror"] = (a_safe * d_safe) / (b_safe * c_safe)
    df["log_ror"] = np.log(df["ror"])
    se_log_ror = np.sqrt(1.0 / a_safe + 1.0 / b_safe + 1.0 / c_safe + 1.0 / d_safe)
    df["ror_ci_lower"] = np.exp(df["log_ror"] - 1.96 * se_log_ror)
    df["ror_ci_upper"] = np.exp(df["log_ror"] + 1.96 * se_log_ror)
    df["flag_ror"] = ((df["ror"] >= 2.0) & (df["ror_ci_lower"] > 1.0)).astype(int)
    return df


def compute_prr(df):
    """
    Proportional Reporting Ratio with chi-square statistic.
        PRR = (a / (a + b)) / (c / (c + d))
        chi-square: standard 2x2 Pearson statistic on the contingency table
    Signal threshold (Evans et al. 2001): PRR >= 2 AND chi-square >= 4 AND a >= 3.
    """
    a, b, c, d = df["a"], df["b"], df["c"], df["d"]

    p_drug = a / (a + b).clip(lower=1)
    p_other = c / (c + d).clip(lower=1)
    df["prr"] = p_drug / p_other.clip(lower=1e-12)
    df["log_prr"] = np.log(df["prr"].clip(lower=1e-12))

    # 2x2 chi-square (without continuity correction, standard for PV signal detection)
    n = a + b + c + d
    numerator = (a * d - b * c) ** 2 * n
    denominator = ((a + b) * (c + d) * (a + c) * (b + d)).clip(lower=1)
    df["chi_square"] = numerator / denominator

    df["flag_prr"] = (
        (df["prr"] >= 2.0) & (df["chi_square"] >= 4.0) & (df["a"] >= 3)
    ).astype(int)
    return df


def compute_ic(df):
    """
    Information Component using BCPNN (Bate et al., 1998).
    Standard formulation with shrinkage parameter 0.5 on each cell:
        IC = log2((a + 0.5) * (N + 1) / ((n_drug + 0.5) * (n_event + 0.5)))
        V(IC) ~ (1/ln(2))^2 * (1/(a+0.5) + 1/(n_drug+0.5) + 1/(n_event+0.5) - 1/(N+1))
        IC025 = IC - 1.96 * sqrt(V(IC))
    Signal threshold: IC025 > 0.
    """
    a = df["a"]
    n_drug = df["n_drug"]
    n_event = df["n_event"]
    N = df["N"]

    numerator = (a + 0.5) * (N + 1)
    denominator = (n_drug + 0.5) * (n_event + 0.5)
    df["ic"] = np.log2(numerator / denominator)

    var_ic = (1.0 / np.log(2)) ** 2 * (
        1.0 / (a + 0.5) + 1.0 / (n_drug + 0.5) + 1.0 / (n_event + 0.5) - 1.0 / (N + 1)
    )
    var_ic = var_ic.clip(lower=0)
    df["ic_025"] = df["ic"] - 1.96 * np.sqrt(var_ic)
    df["flag_ic"] = (df["ic_025"] > 0).astype(int)
    return df


def compute_ebgm(df):
    """
    EBGM via simplified Gamma-Poisson shrinkage (DuMouchel, 1999).

    Model:
        Observed count O ~ Poisson(lambda * E), where E is the expected count
        Prior: lambda ~ Gamma(alpha, beta)  [single component, simplified]
        Posterior: lambda | O ~ Gamma(alpha + O, beta + E)

    Hyperparameters alpha, beta are estimated by method of moments on the
    empirical relative reporting ratio (RR = O / E):
        mean(RR) ~ alpha / beta
        var(RR)  ~ alpha / beta^2  (approximate, ignoring sampling variability)

    EBGM is the posterior geometric mean:
        EBGM = exp(E[log(lambda) | O])
             = exp(digamma(alpha + O) - log(beta + E))

    EB05 is the 5th percentile of the posterior Gamma:
        EB05 = qgamma(0.05, shape = alpha + O, scale = 1 / (beta + E))

    Signal threshold: EB05 > 2 (DuMouchel 1999, FDA internal screening).

    NOTE: openEBGM (R) uses a 2-component Gamma mixture which is more
    flexible. This single-component approximation is faster and adequate
    for hypothesis-generating signal detection.
    """
    O = df["a"].astype(float).values
    E = df["expected"].astype(float).values

    # Empirical RR for hyperparameter estimation, with light trimming to avoid
    # extreme outliers dominating the moment estimates
    rr = O / np.clip(E, 1e-9, None)
    rr_trimmed = rr[(rr > 0) & (rr < np.quantile(rr, 0.99))]
    mean_rr = rr_trimmed.mean()
    var_rr = rr_trimmed.var(ddof=1)

    # Method of moments: alpha = mean^2 / var, beta = mean / var
    alpha = max(mean_rr ** 2 / max(var_rr, 1e-9), 0.5)
    beta = max(mean_rr / max(var_rr, 1e-9), 0.5)

    print(f"  EBGM prior estimated: alpha={alpha:.3f}, beta={beta:.3f}")
    print(f"  (interpretation: prior mean of lambda ~ {alpha/beta:.3f})")

    # Posterior parameters
    post_shape = alpha + O
    post_rate = beta + E
    post_scale = 1.0 / post_rate

    # Posterior geometric mean (EBGM)
    df["ebgm"] = np.exp(digamma(post_shape) - np.log(post_rate))
    df["log_ebgm"] = np.log(df["ebgm"].clip(lower=1e-12))

    # EB05 = 5th percentile of posterior Gamma
    df["eb05"] = stats.gamma.ppf(0.05, a=post_shape, scale=post_scale)
    df["flag_ebgm"] = (df["eb05"] > 2.0).astype(int)
    return df


# ============================================================
# Step C: Consensus flag and summary
# ============================================================

def add_consensus(df):
    """A pair is in the 'consensus' set if at least 3 of 4 methods flag it."""
    df["flag_count"] = df[["flag_ror", "flag_prr", "flag_ic", "flag_ebgm"]].sum(axis=1)
    df["flag_consensus"] = (df["flag_count"] >= 3).astype(int)
    return df


def print_flag_summary(df):
    """Reproduces the layout of Table 10 in the interim report."""
    n_total = len(df)
    rows = [
        ("ROR",       "ROR >= 2; lower 95% CI > 1",         df["flag_ror"].sum()),
        ("PRR",       "PRR >= 2; chi^2 >= 4; a >= 3",       df["flag_prr"].sum()),
        ("IC",        "IC025 > 0",                          df["flag_ic"].sum()),
        ("EBGM",      "EB05 > 2",                           df["flag_ebgm"].sum()),
        ("Consensus", ">= 3 of 4 methods agreeing",         df["flag_consensus"].sum()),
    ]
    print("\n=== Disproportionality Flag Counts ===")
    print(f"{'Method':<11} {'Threshold':<35} {'Pairs flagged':>14} {'Pct of set':>12}")
    print("-" * 75)
    for method, threshold, count in rows:
        pct = count / n_total * 100 if n_total else 0
        print(f"{method:<11} {threshold:<35} {count:>14,} {pct:>11.2f}%")

    summary = pd.DataFrame(
        [
            {"method": m, "threshold": t, "pairs_flagged": c,
             "pct_of_candidate_set": round(c / n_total * 100, 2)}
            for m, t, c in rows
        ]
    )
    summary.to_csv(TABLE_DIR / "flag_counts.csv", index=False)
    print(f"\nSaved to outputs/tables/flag_counts.csv")


def main():
    print("Loading analytic table...")
    df = load_analytic()
    print(f"Loaded {len(df):,} rows")

    print("\nBuilding contingency tables...")
    candidates = build_contingency(df)

    print("\nComputing ROR...")
    candidates = compute_ror(candidates)

    print("Computing PRR...")
    candidates = compute_prr(candidates)

    print("Computing IC (BCPNN)...")
    candidates = compute_ic(candidates)

    print("Computing EBGM (Gamma-Poisson shrinkage)...")
    candidates = compute_ebgm(candidates)

    print("\nAdding consensus flag...")
    candidates = add_consensus(candidates)

    print_flag_summary(candidates)

    # Save the full candidate set with all features
    out_path = PROCESSED_DIR / "candidate_set.parquet"
    candidates.to_parquet(out_path, index=False)
    print(f"\nSaved candidate set to {out_path}")
    print(f"  Rows: {len(candidates):,}")
    print(f"  Columns: {list(candidates.columns)}")


# ------------------------------------------------------------
# Optional: full openEBGM via rpy2 (only if you want the 2-component mixture)
# ------------------------------------------------------------
def ebgm_via_openebgm(df):
    """
    Stub showing how to compute EBGM via the R openEBGM package.
    Requires R, rpy2, and the openEBGM package installed.

    In R:
        install.packages("openEBGM")

    In Python:
        pip install rpy2

    Usage (replace compute_ebgm() above with this if you want the full mixture):
        from rpy2.robjects import pandas2ri, r
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        openebgm = importr("openEBGM")
        # ... see openEBGM vignette for the exact processRaw + autoHyper + ebgm flow
    """
    raise NotImplementedError("Stub - implement when ready to use full mixture model")


if __name__ == "__main__":
    main()
