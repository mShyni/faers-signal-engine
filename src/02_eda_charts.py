# src/02_eda_charts.py
#
# Step 2: Exploratory Data Analysis charts.
#
# Generates the charts referenced in the interim report (Figures 1-3 and a few extras).
# All figures are saved to outputs/figures/ as PNG files at 300 dpi.
#
# Charts produced:
#   - pipeline_stage_counts.png  (Figure 1: row counts at each cleaning stage)
#   - sex_distribution.png       (Figure 2: sex distribution in analytic table)
#   - top15_reactions.png        (Figure 3: top 15 MedDRA Preferred Terms)
#   - top15_drugs.png            (top 15 ingredients by report volume)
#   - quarterly_volume.png       (reports per quarter line chart)
#   - age_distribution.png       (age histogram)
#   - reports_by_country.png     (top 10 reporting countries)


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

PROCESSED_DIR = Path("data/processed")
FIG_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Use a clean, readable matplotlib style throughout
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def load_data():
    """Load the analytic table produced by 01_clean_and_merge.py."""
    path = PROCESSED_DIR / "analytic_table.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 01_clean_and_merge.py first."
        )
    return pd.read_parquet(path)


def chart_pipeline_stages():
    """Figure 1 in the interim report. Reads the summary CSV produced in step 1."""
    src = TABLE_DIR / "pipeline_stage_counts.csv"
    if not src.exists():
        print(f"Skipping pipeline stage chart, {src} not found.")
        return
    df = pd.read_csv(src).dropna(subset=["rows"])
    df["rows_millions"] = df["rows"] / 1_000_000

    fig, ax = plt.subplots()
    ax.barh(df["stage"], df["rows_millions"], color="#4472C4")
    ax.set_xlabel("Records (millions)")
    ax.set_title("Figure 1. Record volume across major pipeline stages")
    ax.invert_yaxis()
    for i, val in enumerate(df["rows_millions"]):
        ax.text(val, i, f" {val:.2f}M", va="center")
    plt.tight_layout()
    out = FIG_DIR / "pipeline_stage_counts.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def chart_sex_distribution(df):
    """Figure 2: sex distribution in cleaned FAERS analytic table."""
    counts = df["sex"].value_counts(dropna=False)
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots()
    bars = ax.bar(counts.index, pct.values,
                  color=["#E15759", "#4472C4", "#A6A6A6"])
    ax.set_ylabel("Percent of analytic rows")
    ax.set_title("Figure 2. Sex distribution in the cleaned FAERS analytic table")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    for bar, p in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{p:.1f}%", ha="center")

    plt.tight_layout()
    out = FIG_DIR / "sex_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def chart_top_reactions(df, n=15):
    """Figure 3: most frequent MedDRA Preferred Terms after cleaning."""
    top = df["pt"].value_counts().head(n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index[::-1], top.values[::-1], color="#70AD47")
    ax.set_xlabel("Number of analytic rows")
    ax.set_title(f"Figure 3. Top {n} MedDRA Preferred Terms after cleaning\n"
                 f"(High frequency alone does not imply higher drug-specific risk)")
    for i, v in enumerate(top.values[::-1]):
        ax.text(v, i, f" {v:,}", va="center")
    plt.tight_layout()
    out = FIG_DIR / "top15_reactions.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def chart_top_drugs(df, n=15):
    """Top ingredients by report volume."""
    top = df["ingredient"].value_counts().head(n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index[::-1], top.values[::-1], color="#ED7D31")
    ax.set_xlabel("Number of analytic rows")
    ax.set_title(f"Top {n} drug ingredients by report volume")
    for i, v in enumerate(top.values[::-1]):
        ax.text(v, i, f" {v:,}", va="center")
    plt.tight_layout()
    out = FIG_DIR / "top15_drugs.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def chart_quarterly_volume(df):
    """Reports per quarter (line chart). Helps spot structural breaks."""
    # quarter labels look like 2023Q1, 2023Q2, ...; sort them properly
    q_counts = df.groupby("quarter")["primaryid"].nunique().reset_index()
    q_counts = q_counts.sort_values("quarter")

    fig, ax = plt.subplots()
    ax.plot(q_counts["quarter"], q_counts["primaryid"], marker="o", color="#4472C4")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Unique reports (primaryid)")
    ax.set_title("Quarterly report volume in the analytic table")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = FIG_DIR / "quarterly_volume.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")

    # Save the underlying counts as a CSV (useful for the final report)
    q_counts.to_csv(TABLE_DIR / "quarterly_volume.csv", index=False)


def chart_age_distribution(df):
    """Histogram of patient age, capped at 100 to ignore obvious data-entry errors."""
    ages = df["age"].dropna()
    ages = ages[(ages >= 0) & (ages <= 100)]

    fig, ax = plt.subplots()
    ax.hist(ages, bins=20, color="#4472C4", edgecolor="white")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of analytic rows")
    ax.set_title("Patient age distribution (analytic rows, ages 0-100)")
    plt.tight_layout()
    out = FIG_DIR / "age_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def chart_top_countries(df, n=10):
    """Top n reporting countries."""
    if "country" not in df.columns:
        return
    top = df["country"].fillna("Unknown").value_counts().head(n)

    fig, ax = plt.subplots()
    ax.bar(top.index, top.values, color="#7030A0")
    ax.set_ylabel("Number of analytic rows")
    ax.set_title(f"Top {n} reporting countries")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = FIG_DIR / "top_countries.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def main():
    print("Loading analytic table...")
    df = load_data()
    print(f"Loaded {len(df):,} rows")

    print("\nGenerating EDA charts...")
    chart_pipeline_stages()
    chart_sex_distribution(df)
    chart_top_reactions(df, n=15)
    chart_top_drugs(df, n=15)
    chart_quarterly_volume(df)
    chart_age_distribution(df)
    chart_top_countries(df, n=10)

    # Save a small EDA summary table for the final write-up
    sex_summary = df["sex"].value_counts(normalize=True).reset_index()
    sex_summary.columns = ["sex", "proportion"]
    sex_summary.to_csv(TABLE_DIR / "sex_distribution.csv", index=False)

    top_pts = df["pt"].value_counts().head(20).reset_index()
    top_pts.columns = ["preferred_term", "rows"]
    top_pts.to_csv(TABLE_DIR / "top20_reactions.csv", index=False)

    top_drugs = df["ingredient"].value_counts().head(20).reset_index()
    top_drugs.columns = ["ingredient", "rows"]
    top_drugs.to_csv(TABLE_DIR / "top20_drugs.csv", index=False)

    print("\nDone. All charts saved to outputs/figures/")


if __name__ == "__main__":
    main()
