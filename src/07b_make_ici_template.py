# src/07b_make_ici_template.py
#
# Helper script to generate a starter CSV of immune-related adverse events
# (irAEs) labelled in the product labels of the seven approved immune
# checkpoint inhibitors. The label_revision_quarter is the FAERS quarter
# in which the irAE was added to the product label (Adverse Reactions or
# Warnings/Precautions section).
#
# IMPORTANT: The dates below are PLACEHOLDERS based on the original product
# label dates and well-known label revisions. You should verify and update
# them from the actual FDA labels at https://accessdata.fda.gov/scripts/cder/daf/
# before final analysis. For irAEs that were already on the label before the
# 2023Q1 study window starts, set label_revision_quarter to 2023Q1 - this means
# any engine detection during the study period is "after the fact" and will
# show as a non-positive lead time, which is the conservative interpretation.
#
# Required columns: ingredient, pt, label_revision_quarter, soc

from pathlib import Path
import pandas as pd

REF_DIR = Path("data/reference")
REF_DIR.mkdir(parents=True, exist_ok=True)

# Format: (ingredient, PT, label_revision_quarter, SOC)
# All seven approved ICIs already have most class-effect irAEs on their
# labels prior to 2023Q1. We mark these as "2023Q1" to indicate the irAE
# was already labelled at the start of the study window. New label additions
# during the study window should be entered with their actual quarter.
ROWS = [
    # Pembrolizumab
    ("PEMBROLIZUMAB", "Hypothyroidism",      "2023Q1", "Endocrine"),
    ("PEMBROLIZUMAB", "Hyperthyroidism",     "2023Q1", "Endocrine"),
    ("PEMBROLIZUMAB", "Pneumonitis",         "2023Q1", "Respiratory"),
    ("PEMBROLIZUMAB", "Colitis",             "2023Q1", "Gastrointestinal"),
    ("PEMBROLIZUMAB", "Hepatitis",           "2023Q1", "Hepatobiliary"),
    ("PEMBROLIZUMAB", "Adrenal Insufficiency", "2023Q1", "Endocrine"),
    ("PEMBROLIZUMAB", "Hypophysitis",        "2023Q1", "Endocrine"),
    ("PEMBROLIZUMAB", "Type 1 Diabetes Mellitus", "2023Q1", "Endocrine"),
    ("PEMBROLIZUMAB", "Nephritis",           "2023Q1", "Renal"),
    ("PEMBROLIZUMAB", "Myocarditis",         "2023Q1", "Cardiac"),

    # Nivolumab
    ("NIVOLUMAB", "Hypothyroidism",          "2023Q1", "Endocrine"),
    ("NIVOLUMAB", "Hyperthyroidism",         "2023Q1", "Endocrine"),
    ("NIVOLUMAB", "Pneumonitis",             "2023Q1", "Respiratory"),
    ("NIVOLUMAB", "Colitis",                 "2023Q1", "Gastrointestinal"),
    ("NIVOLUMAB", "Hepatitis",               "2023Q1", "Hepatobiliary"),
    ("NIVOLUMAB", "Adrenal Insufficiency",   "2023Q1", "Endocrine"),
    ("NIVOLUMAB", "Hypophysitis",            "2023Q1", "Endocrine"),
    ("NIVOLUMAB", "Nephritis",               "2023Q1", "Renal"),
    ("NIVOLUMAB", "Myocarditis",             "2023Q1", "Cardiac"),

    # Atezolizumab
    ("ATEZOLIZUMAB", "Hypothyroidism",       "2023Q1", "Endocrine"),
    ("ATEZOLIZUMAB", "Pneumonitis",          "2023Q1", "Respiratory"),
    ("ATEZOLIZUMAB", "Colitis",              "2023Q1", "Gastrointestinal"),
    ("ATEZOLIZUMAB", "Hepatitis",            "2023Q1", "Hepatobiliary"),
    ("ATEZOLIZUMAB", "Adrenal Insufficiency", "2023Q1", "Endocrine"),

    # Durvalumab
    ("DURVALUMAB", "Hypothyroidism",         "2023Q1", "Endocrine"),
    ("DURVALUMAB", "Pneumonitis",            "2023Q1", "Respiratory"),
    ("DURVALUMAB", "Colitis",                "2023Q1", "Gastrointestinal"),
    ("DURVALUMAB", "Hepatitis",              "2023Q1", "Hepatobiliary"),

    # Avelumab
    ("AVELUMAB", "Pneumonitis",              "2023Q1", "Respiratory"),
    ("AVELUMAB", "Colitis",                  "2023Q1", "Gastrointestinal"),
    ("AVELUMAB", "Hepatitis",                "2023Q1", "Hepatobiliary"),
    ("AVELUMAB", "Hypothyroidism",           "2023Q1", "Endocrine"),

    # Ipilimumab
    ("IPILIMUMAB", "Colitis",                "2023Q1", "Gastrointestinal"),
    ("IPILIMUMAB", "Hepatitis",              "2023Q1", "Hepatobiliary"),
    ("IPILIMUMAB", "Hypophysitis",           "2023Q1", "Endocrine"),
    ("IPILIMUMAB", "Pneumonitis",            "2023Q1", "Respiratory"),
    ("IPILIMUMAB", "Adrenal Insufficiency",  "2023Q1", "Endocrine"),

    # Cemiplimab
    ("CEMIPLIMAB", "Pneumonitis",            "2023Q1", "Respiratory"),
    ("CEMIPLIMAB", "Colitis",                "2023Q1", "Gastrointestinal"),
    ("CEMIPLIMAB", "Hepatitis",              "2023Q1", "Hepatobiliary"),
    ("CEMIPLIMAB", "Hypothyroidism",         "2023Q1", "Endocrine"),
]


def main():
    df = pd.DataFrame(ROWS, columns=["ingredient", "pt", "label_revision_quarter", "soc"])
    out = REF_DIR / "ici_label_revisions.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print("\nIMPORTANT: review and update label_revision_quarter values from")
    print("the actual FDA labels (https://accessdata.fda.gov/scripts/cder/daf/)")
    print("before running 07_rq4_survival_ici.py for final results.")


if __name__ == "__main__":
    main()
