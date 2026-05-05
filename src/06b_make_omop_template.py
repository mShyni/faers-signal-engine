# src/06b_make_omop_template.py
#
# Helper script to generate a starter OMOP reference CSV. Fills in the
# four-outcome positive/negative pairs from Ryan et al. (2013) and Harpaz
# et al. (2014). Edit and extend before running 06_rq3_random_forest.py.
#
# Reference sources:
#   Ryan, P. B. et al. (2013). Defining a reference set to support
#       methodological research in drug safety. Drug Safety, 36, S33-S47.
#   Harpaz, R. et al. (2014). A time-indexed reference standard of adverse
#       drug reactions. Scientific Data, 1, 140043.
#
# The four outcomes are:
#   - Acute liver injury
#   - Acute kidney injury
#   - Acute myocardial infarction
#   - Gastrointestinal bleeding
#
# Each outcome maps to one or more MedDRA Preferred Terms in our analytic
# table. The Ryan et al. reference set has 165 positives and 234 negatives
# total across the four outcomes. Below is a CURATED STARTER set you can
# build on. Add rows from the source papers' supplements as you confirm
# them.

from pathlib import Path
import pandas as pd

REF_DIR = Path("data/reference")
REF_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Outcome -> MedDRA PT mapping ----------
# These are the most commonly used PT mappings for the four OMOP outcomes
# in the FAERS literature. Multiple PTs map to one outcome - we will expand
# the reference list accordingly.
OUTCOME_PTS = {
    "Acute Liver Injury": [
        "Hepatic Failure", "Hepatitis Acute", "Liver Injury",
        "Hepatocellular Injury", "Hepatic Necrosis",
    ],
    "Acute Kidney Injury": [
        "Acute Kidney Injury", "Renal Failure Acute",
        "Renal Tubular Necrosis",
    ],
    "Acute Myocardial Infarction": [
        "Myocardial Infarction", "Acute Myocardial Infarction",
    ],
    "Gastrointestinal Bleeding": [
        "Gastrointestinal Haemorrhage", "Upper Gastrointestinal Haemorrhage",
        "Lower Gastrointestinal Haemorrhage", "Melaena", "Haematemesis",
    ],
}

# ---------- Curated positive controls (drug, outcome) ----------
# Drugs are RxNorm ingredient names in uppercase. These are well-established
# associations from product labels and pharmacology references. This is a
# STARTER list - extend by consulting Ryan et al. supplement Table S2.
POSITIVES = [
    # Acute Liver Injury
    ("ACETAMINOPHEN", "Acute Liver Injury"),
    ("ISONIAZID", "Acute Liver Injury"),
    ("VALPROIC ACID", "Acute Liver Injury"),
    ("AMOXICILLIN", "Acute Liver Injury"),
    ("KETOCONAZOLE", "Acute Liver Injury"),
    ("METHOTREXATE", "Acute Liver Injury"),
    ("DICLOFENAC", "Acute Liver Injury"),
    ("LEFLUNOMIDE", "Acute Liver Injury"),
    ("NEVIRAPINE", "Acute Liver Injury"),
    ("PROPYLTHIOURACIL", "Acute Liver Injury"),

    # Acute Kidney Injury
    ("VANCOMYCIN", "Acute Kidney Injury"),
    ("GENTAMICIN", "Acute Kidney Injury"),
    ("IBUPROFEN", "Acute Kidney Injury"),
    ("CISPLATIN", "Acute Kidney Injury"),
    ("AMPHOTERICIN B", "Acute Kidney Injury"),
    ("LITHIUM", "Acute Kidney Injury"),
    ("TACROLIMUS", "Acute Kidney Injury"),
    ("CYCLOSPORINE", "Acute Kidney Injury"),

    # Acute Myocardial Infarction
    ("ROFECOXIB", "Acute Myocardial Infarction"),
    ("CELECOXIB", "Acute Myocardial Infarction"),
    ("SIBUTRAMINE", "Acute Myocardial Infarction"),
    ("ROSIGLITAZONE", "Acute Myocardial Infarction"),

    # GI Bleeding
    ("ASPIRIN", "Gastrointestinal Bleeding"),
    ("WARFARIN", "Gastrointestinal Bleeding"),
    ("IBUPROFEN", "Gastrointestinal Bleeding"),
    ("NAPROXEN", "Gastrointestinal Bleeding"),
    ("DICLOFENAC", "Gastrointestinal Bleeding"),
    ("CLOPIDOGREL", "Gastrointestinal Bleeding"),
    ("APIXABAN", "Gastrointestinal Bleeding"),
    ("RIVAROXABAN", "Gastrointestinal Bleeding"),
    ("DABIGATRAN", "Gastrointestinal Bleeding"),
]

# ---------- Curated negative controls (drug, outcome) ----------
# Negative controls are drug-outcome pairs with no known association. These
# follow Ryan et al.'s approach of sampling drugs not labelled for the
# outcome and not biologically plausible.
NEGATIVES = [
    ("LEVOTHYROXINE", "Acute Liver Injury"),
    ("LEVOTHYROXINE", "Acute Kidney Injury"),
    ("LEVOTHYROXINE", "Acute Myocardial Infarction"),
    ("METFORMIN", "Gastrointestinal Bleeding"),
    ("METFORMIN", "Acute Liver Injury"),
    ("AMLODIPINE", "Acute Liver Injury"),
    ("AMLODIPINE", "Gastrointestinal Bleeding"),
    ("LISINOPRIL", "Acute Liver Injury"),
    ("LISINOPRIL", "Acute Myocardial Infarction"),
    ("OMEPRAZOLE", "Acute Myocardial Infarction"),
    ("LORATADINE", "Acute Liver Injury"),
    ("LORATADINE", "Acute Kidney Injury"),
    ("LORATADINE", "Acute Myocardial Infarction"),
    ("LORATADINE", "Gastrointestinal Bleeding"),
    ("CETIRIZINE", "Acute Liver Injury"),
    ("CETIRIZINE", "Acute Myocardial Infarction"),
    ("CETIRIZINE", "Gastrointestinal Bleeding"),
    ("FAMOTIDINE", "Acute Liver Injury"),
    ("FAMOTIDINE", "Acute Myocardial Infarction"),
    ("MELATONIN", "Acute Liver Injury"),
    ("MELATONIN", "Acute Kidney Injury"),
    ("MELATONIN", "Acute Myocardial Infarction"),
    ("MELATONIN", "Gastrointestinal Bleeding"),
]


def expand(entries, label):
    """Expand a list of (drug, outcome) into one row per (drug, PT)."""
    rows = []
    for drug, outcome in entries:
        for pt in OUTCOME_PTS[outcome]:
            rows.append({
                "ingredient": drug.upper(),
                "pt": pt,
                "outcome": outcome,
                "label": label,
            })
    return rows


def main():
    pos_rows = expand(POSITIVES, label=1)
    neg_rows = expand(NEGATIVES, label=0)
    df = pd.DataFrame(pos_rows + neg_rows)

    # De-duplicate (a drug-PT pair could appear under multiple outcomes;
    # if it does, treat it as a positive)
    df = df.sort_values("label", ascending=False).drop_duplicates(
        subset=["ingredient", "pt"], keep="first"
    ).reset_index(drop=True)

    out = REF_DIR / "omop_reference.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print(f"  Positives: {(df['label'] == 1).sum()}")
    print(f"  Negatives: {(df['label'] == 0).sum()}")
    print("\nExtend this file by adding rows from the Ryan et al. (2013) supplement.")


if __name__ == "__main__":
    main()
