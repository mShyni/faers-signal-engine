# src/01_clean_and_merge.py
#
# Step 1: Clean the raw FAERS quarter files and merge them into one analytic table

from pathlib import Path
import pandas as pd
import numpy as np

# ---------- Paths ----------
EXTRACT_DIR = Path("data/extracted")
PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs/tables")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Brand to ingredient mapping ----------
# This is a starter dictionary of common brand-to-ingredient mappings.
# Add more as you encounter them. Keys MUST be uppercase.
# For drugs not in this dict, we fall back to whatever the prod_ai field gives us.
BRAND_TO_INGREDIENT = {
    # Biologics and biosimilars (collapsing biosimilars to the originator ingredient
    # is essential, otherwise signals get split across many brand names)
    "HUMIRA": "ADALIMUMAB",
    "AMJEVITA": "ADALIMUMAB",
    "CYLTEZO": "ADALIMUMAB",
    "HYRIMOZ": "ADALIMUMAB",
    "HADLIMA": "ADALIMUMAB",
    "REMICADE": "INFLIXIMAB",
    "INFLECTRA": "INFLIXIMAB",
    "RENFLEXIS": "INFLIXIMAB",
    "AVSOLA": "INFLIXIMAB",
    "ENBREL": "ETANERCEPT",
    "ERELZI": "ETANERCEPT",
    "RITUXAN": "RITUXIMAB",
    "TRUXIMA": "RITUXIMAB",
    "RUXIENCE": "RITUXIMAB",
    "HERCEPTIN": "TRASTUZUMAB",
    "OGIVRI": "TRASTUZUMAB",
    "KANJINTI": "TRASTUZUMAB",

    # GLP-1 / GIP agonists (most-reported drug class in recent quarters)
    "OZEMPIC": "SEMAGLUTIDE",
    "WEGOVY": "SEMAGLUTIDE",
    "RYBELSUS": "SEMAGLUTIDE",
    "MOUNJARO": "TIRZEPATIDE",
    "ZEPBOUND": "TIRZEPATIDE",
    "TRULICITY": "DULAGLUTIDE",
    "VICTOZA": "LIRAGLUTIDE",
    "SAXENDA": "LIRAGLUTIDE",

    # Immune checkpoint inhibitors (used in RQ4)
    "KEYTRUDA": "PEMBROLIZUMAB",
    "OPDIVO": "NIVOLUMAB",
    "TECENTRIQ": "ATEZOLIZUMAB",
    "IMFINZI": "DURVALUMAB",
    "BAVENCIO": "AVELUMAB",
    "YERVOY": "IPILIMUMAB",
    "LIBTAYO": "CEMIPLIMAB",

    # Common anticoagulants and other high-volume drugs
    "ELIQUIS": "APIXABAN",
    "XARELTO": "RIVAROXABAN",
    "PRADAXA": "DABIGATRAN",
    "LIPITOR": "ATORVASTATIN",
    "PLAVIX": "CLOPIDOGREL",
    "NEXIUM": "ESOMEPRAZOLE",
    "PROTONIX": "PANTOPRAZOLE",
    "PRILOSEC": "OMEPRAZOLE",
    "LYRICA": "PREGABALIN",
    "ZOLOFT": "SERTRALINE",
    "PROZAC": "FLUOXETINE",
    "CYMBALTA": "DULOXETINE",
    "EFFEXOR": "VENLAFAXINE",
    "ABILIFY": "ARIPIPRAZOLE",
    "SEROQUEL": "QUETIAPINE",
    "RISPERDAL": "RISPERIDONE",
    "ADDERALL": "AMPHETAMINE/DEXTROAMPHETAMINE",
    "RITALIN": "METHYLPHENIDATE",
    "CONCERTA": "METHYLPHENIDATE",
    "SYNTHROID": "LEVOTHYROXINE",
    "GLUCOPHAGE": "METFORMIN",
    "JANUVIA": "SITAGLIPTIN",
    "JARDIANCE": "EMPAGLIFLOZIN",
    "FARXIGA": "DAPAGLIFLOZIN",
    "INVOKANA": "CANAGLIFLOZIN",
}


def normalize_drug_name(drugname, prod_ai):
    """
    Pick the best ingredient string for a drug row.
    Priority:
      1. If the brand name (uppercased) is in our mapping dict, use the mapped ingredient.
      2. Otherwise, if prod_ai is filled in, use that (uppercased).
      3. Otherwise, fall back to the cleaned drugname.
    """
    if pd.isna(drugname):
        drugname = ""
    drug_upper = str(drugname).strip().upper()

    # Strip dosage info that often follows a comma (e.g. "HUMIRA, 40MG/0.4ML")
    drug_upper_stem = drug_upper.split(",")[0].strip()

    if drug_upper_stem in BRAND_TO_INGREDIENT:
        return BRAND_TO_INGREDIENT[drug_upper_stem]

    if pd.notna(prod_ai) and str(prod_ai).strip():
        return str(prod_ai).strip().upper()

    return drug_upper_stem if drug_upper_stem else np.nan


def parse_faers_date(value):
    """
    FAERS dates can come as YYYYMMDD, YYYYMM, or YYYY (integers or strings).
    Returns a pandas Timestamp or NaT.
    """
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if not s or s == "nan":
        return pd.NaT
    if len(s) == 8:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    if len(s) == 6:
        return pd.to_datetime(s + "01", format="%Y%m%d", errors="coerce")
    if len(s) == 4:
        return pd.to_datetime(s + "0101", format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def normalize_sex(value):
    """Group sex values into Female / Male / Unknown."""
    if pd.isna(value):
        return "Unknown"
    v = str(value).strip().upper()
    if v in ("F", "FEMALE"):
        return "Female"
    if v in ("M", "MALE"):
        return "Male"
    return "Unknown"


def find_table_file(quarter_dir, table_name):
    """
    Each FAERS zip extracts to either an ascii/ subfolder or directly into the quarter folder.
    Look for files like DEMO23Q1.txt, DRUG23Q1.txt, REAC23Q1.txt anywhere under the quarter_dir.
    """
    candidates = list(quarter_dir.rglob(f"{table_name.upper()}*.txt"))
    candidates += list(quarter_dir.rglob(f"{table_name.lower()}*.txt"))
    if not candidates:
        return None
    return candidates[0]


def read_faers_table(filepath, columns_to_keep):
    """
    Read a FAERS pipe-delimited file. FAERS uses '$' as delimiter (not '|').
    Encoding is latin-1 because some reports contain non-UTF-8 characters.
    """
    df = pd.read_csv(
        filepath,
        sep="$",
        encoding="latin-1",
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
    )
    df.columns = [c.strip().lower() for c in df.columns]
    # keep only the columns we actually need
    available = [c for c in columns_to_keep if c in df.columns]
    return df[available].copy()


def clean_demo(filepath, quarter_label):
    """Clean DEMO table for one quarter.

    FAERS uses different column names for country across quarters:
    'reporter_country' is the most recent, 'occr_country' is also used,
    and very old quarters used 'occp_country'. We read all of them,
    then collapse into a single 'country' column. If none are present
    we create an empty country column so downstream code still works.
    """
    df = read_faers_table(filepath, [
        "primaryid", "caseid", "age", "age_cod", "sex",
        "reporter_country", "occr_country", "occp_country",
        "event_dt", "fda_dt"
    ])

    # Collapse country variants into a single 'country' column.
    # Preference order: reporter_country, occr_country, occp_country.
    country_candidates = [c for c in
                          ["reporter_country", "occr_country", "occp_country"]
                          if c in df.columns]
    if country_candidates:
        df["country"] = df[country_candidates[0]]
        for c in country_candidates[1:]:
            df["country"] = df["country"].fillna(df[c])
        # drop the originals so we don't carry duplicates downstream
        df = df.drop(columns=country_candidates)
    else:
        df["country"] = pd.NA

    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    df["caseid"] = pd.to_numeric(df["caseid"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = df["sex"].apply(normalize_sex)
    if "event_dt" in df.columns:
        df["event_dt"] = df["event_dt"].apply(parse_faers_date)
    if "fda_dt" in df.columns:
        df["fda_dt"] = df["fda_dt"].apply(parse_faers_date)
    df["quarter"] = quarter_label
    df = df.dropna(subset=["primaryid", "caseid"])
    return df


def clean_drug(filepath):
    """Clean DRUG table for one quarter and apply brand-to-ingredient mapping."""
    df = read_faers_table(filepath, [
        "primaryid", "drug_seq", "drugname", "prod_ai", "role_cod"
    ])
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    df = df.dropna(subset=["primaryid"])

    # Keep only Primary Suspect and Secondary Suspect drug roles
    df["role_cod"] = df["role_cod"].fillna("").str.strip().str.upper()
    df = df[df["role_cod"].isin(["PS", "SS"])].copy()

    # Build the ingredient column
    df["ingredient"] = df.apply(
        lambda row: normalize_drug_name(row.get("drugname"), row.get("prod_ai")),
        axis=1,
    )
    df = df.dropna(subset=["ingredient"])
    df = df[df["ingredient"].str.len() > 0]
    return df[["primaryid", "drug_seq", "ingredient", "role_cod"]]


def clean_reac(filepath):
    """Clean REAC table for one quarter."""
    df = read_faers_table(filepath, ["primaryid", "pt"])
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    df = df.dropna(subset=["primaryid"])
    df["pt"] = df["pt"].fillna("").str.strip()
    df = df[df["pt"].str.len() > 0]
    df["pt"] = df["pt"].str.title()  # standardize case for MedDRA PT
    return df


def process_one_quarter(quarter_dir):
    """Read DEMO, DRUG, REAC for one quarter and return cleaned dataframes."""
    quarter_label = quarter_dir.name.replace("faers_ascii_", "").upper()
    print(f"\n--- Processing {quarter_label} ---")

    demo_file = find_table_file(quarter_dir, "DEMO")
    drug_file = find_table_file(quarter_dir, "DRUG")
    reac_file = find_table_file(quarter_dir, "REAC")

    if not all([demo_file, drug_file, reac_file]):
        print(f"  Missing files in {quarter_dir.name}, skipping")
        return None, None, None

    demo = clean_demo(demo_file, quarter_label)
    drug = clean_drug(drug_file)
    reac = clean_reac(reac_file)

    print(f"  DEMO rows: {len(demo):,}")
    print(f"  DRUG rows (PS+SS only): {len(drug):,}")
    print(f"  REAC rows: {len(reac):,}")

    return demo, drug, reac


def main():
    quarter_dirs = sorted([p for p in EXTRACT_DIR.iterdir() if p.is_dir()])
    if not quarter_dirs:
        print(f"No quarter folders found under {EXTRACT_DIR}")
        print("Run download_extract_faers.py first.")
        return

    print(f"Found {len(quarter_dirs)} quarters to process")

    all_demo = []
    all_drug = []
    all_reac = []

    for qdir in quarter_dirs:
        demo, drug, reac = process_one_quarter(qdir)
        if demo is None:
            continue
        all_demo.append(demo)
        all_drug.append(drug)
        all_reac.append(reac)

    if not all_demo:
        print("No data was processed. Exiting.")
        return

    print("\n--- Combining all quarters ---")
    demo_all = pd.concat(all_demo, ignore_index=True)
    drug_all = pd.concat(all_drug, ignore_index=True)
    reac_all = pd.concat(all_reac, ignore_index=True)

    print(f"Combined DEMO rows (raw): {len(demo_all):,}")
    print(f"Combined DRUG rows (raw): {len(drug_all):,}")
    print(f"Combined REAC rows (raw): {len(reac_all):,}")

    # ---------- Deduplication ----------
    # Each FAERS report version has its own primaryid, but the underlying case is caseid.
    # Following AEOLUS, we keep the most recent primaryid per caseid.
    print("\n--- De-duplicating DEMO by caseid (keep latest primaryid) ---")
    demo_dedup = (
        demo_all.sort_values("primaryid", ascending=False)
        .drop_duplicates(subset=["caseid"], keep="first")
        .reset_index(drop=True)
    )
    print(f"DEMO after dedup: {len(demo_dedup):,}")

    # Restrict DRUG and REAC to primaryids that survived dedup
    kept_pids = set(demo_dedup["primaryid"].tolist())
    drug_kept = drug_all[drug_all["primaryid"].isin(kept_pids)].copy()
    reac_kept = reac_all[reac_all["primaryid"].isin(kept_pids)].copy()
    print(f"DRUG after restriction to deduped cases: {len(drug_kept):,}")
    print(f"REAC after restriction to deduped cases: {len(reac_kept):,}")

    # ---------- Merge ----------
    # Final analytic table: one row per (case, suspect drug, reaction) combination.
    print("\n--- Merging DEMO + DRUG + REAC ---")
    # Defensive: if any expected demo column is missing across all quarters,
    # add it as empty so the selector below cannot raise KeyError.
    desired_demo_cols = ["primaryid", "caseid", "age", "sex",
                         "country", "event_dt", "quarter"]
    for col in desired_demo_cols:
        if col not in demo_dedup.columns:
            demo_dedup[col] = pd.NA

    drug_demo = drug_kept.merge(
        demo_dedup[desired_demo_cols],
        on="primaryid",
        how="inner",
    )
    analytic = drug_demo.merge(reac_kept, on="primaryid", how="inner")
    print(f"Analytic table rows: {len(analytic):,}")
    print(f"Unique caseids: {analytic['caseid'].nunique():,}")
    print(f"Unique ingredients: {analytic['ingredient'].nunique():,}")
    print(f"Unique PTs: {analytic['pt'].nunique():,}")

    # ---------- Save ----------
    out_path = PROCESSED_DIR / "analytic_table.parquet"
    analytic.to_parquet(out_path, index=False)
    print(f"\nSaved analytic table to: {out_path}")

    # Also save the dedup DEMO so EDA can use case-level demographics directly
    demo_dedup.to_parquet(PROCESSED_DIR / "demo_dedup.parquet", index=False)
    print(f"Saved deduped DEMO to: {PROCESSED_DIR / 'demo_dedup.parquet'}")

    # Save a small pipeline-stage summary that mirrors Table 8 in the interim report
    summary = pd.DataFrame([
        {"stage": "Raw DEMO ingest", "rows": len(demo_all)},
        {"stage": "Raw DRUG ingest (all roles)", "rows": None},  # not retained, see note
        {"stage": "Raw REAC ingest", "rows": len(reac_all)},
        {"stage": "DEMO after deduplication", "rows": len(demo_dedup)},
        {"stage": "DRUG after PS+SS role restriction", "rows": len(drug_all)},
        {"stage": "DRUG kept after dedup join", "rows": len(drug_kept)},
        {"stage": "Merged analytic table", "rows": len(analytic)},
    ])
    summary.to_csv(OUTPUTS_DIR / "pipeline_stage_counts.csv", index=False)
    print(f"Saved pipeline stage summary to {OUTPUTS_DIR / 'pipeline_stage_counts.csv'}")


if __name__ == "__main__":
    main()