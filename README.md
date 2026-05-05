# FAERS Drug Safety Surveillance Signal-Detection Engine

QM640 Capstone Project. Builds a disproportionality and machine-learning pipeline on twelve quarters of FDA FAERS data (2023Q1 through 2025Q4) to address four research questions.

The raw data files are not uploaded to this GitHub repository because they are large.  
Instead, the data can be downloaded directly from the FDA FAERS website (https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html).

## Project layout

```
faers-signal-engine/
├── requirements.txt
├── README.md
├── src/
│   ├── download_extract_faers.py   (your existing script)
│   ├── 01_clean_and_merge.py
│   ├── 02_eda_charts.py
│   ├── 03_feature_engineering.py
│   ├── 04_rq1_isolation_forest.py
│   ├── 05_rq2_gmm.py
│   ├── 06b_make_omop_template.py
│   ├── 06_rq3_random_forest.py
│   ├── 07b_make_ici_template.py
│   ├── 07_rq4_survival_ici.py
│   └── 08_preliminary_results.py
├── data/
│   ├── raw/         (zips land here)
│   ├── extracted/   (DEMO/DRUG/REAC text files)
│   ├── processed/   (parquet outputs from steps 01 and 03)
│   └── reference/   (your OMOP and ICI label CSVs)
└── outputs/
    ├── figures/     (PNG charts at 300 dpi)
    └── tables/      (CSV summaries)
```

## Setup

From a terminal inside the project folder:

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (Mac or Linux)

pip install -r requirements.txt
```

If `scikit-survival` fails to install, the RQ4 script will still run the lead-time test and just skip the Random Survival Forest piece. You can install the rest and come back to it.

## Run order

Run each script from the project root, one at a time. Each one prints progress to the console and writes outputs to `outputs/figures/` and `outputs/tables/`.

| Step | Script | What it does | Roughly how long |
|------|--------|--------------|-------------------|
| 0 | `download_extract_faers.py` | Downloads and extracts all 12 quarter zips | 30 to 60 min depending on network |
| 1 | `01_clean_and_merge.py` | Cleans DEMO, DRUG, REAC files. Maps brand names to ingredients. Deduplicates cases. Saves `analytic_table.parquet` and `pipeline_stage_counts.csv` (Table 8 in the report). | 5 to 10 min |
| 2 | `02_eda_charts.py` | Generates the seven EDA charts referenced in the interim report (pipeline counts, sex distribution, top reactions, top drugs, quarterly volume, age, country). | 1 to 2 min |
| 3 | `03_feature_engineering.py` | Builds drug-event contingency tables and computes ROR, PRR, IC, EBGM with confidence bounds. Saves `candidate_set.parquet` and `flag_counts.csv` (Table 10). | 10 to 20 min |
| 4 | `06b_make_omop_template.py` | Creates a starter `data/reference/omop_reference.csv` with positive and negative drug-event controls. **Edit this file before running step 6.** | seconds |
| 5 | `04_rq1_isolation_forest.py` | RQ1: unsupervised anomaly detection on the four-method score vector. | 2 to 5 min |
| 6 | `05_rq2_gmm.py` | RQ2: cross-method concordance plus a Gaussian Mixture Model to find a high-signal cluster. | 2 to 5 min |
| 7 | `06_rq3_random_forest.py` | RQ3: supervised Random Forest on the OMOP-testable subset, compared against logistic regression and each single method. | 2 to 5 min |
| 8 | `07b_make_ici_template.py` | Creates a starter `data/reference/ici_label_revisions.csv` with the seven immune checkpoint inhibitors and their classic immune-related adverse events. **Edit this file before running step 9.** | seconds |
| 9 | `07_rq4_survival_ici.py` | RQ4: builds quarterly cumulative trajectories for the seven ICIs, finds the first quarter each drug-event pair crosses the EBGM threshold, computes lead time against your label dates, and fits a Random Survival Forest. | 5 to 10 min |
| 10 | `08_preliminary_results.py` | Reads every CSV produced above and writes one consolidated `preliminary_results_summary.txt` plus a `hypothesis_decisions.csv`. This is what you cite in the final report. | seconds |

## A few things worth knowing

**Brand to ingredient mapping.** Step 1 contains a dictionary of about 70 common brand names mapped to RxNorm ingredients. It covers the seven ICIs, common biologics, GLP-1s, anticoagulants, and frequently reported psychiatric and diabetes drugs. If you find an important brand that is not mapped, just add it to the `BRAND_TO_INGREDIENT` dict near the top of `01_clean_and_merge.py` and rerun.

**EBGM implementation.** The interim report planned to use `openEBGM` in R via rpy2. To keep the install simple, step 3 ships with a single-component Gamma-Poisson shrinkage model that uses method-of-moments to estimate the prior. This is the same family of model and gives very similar EB05 rankings. If you want the full two-component DuMouchel (1999) version, uninstall rpy2 from `requirements.txt` and replace `compute_ebgm()` with the `ebgm_via_openebgm()` stub already in the file.

**Reference files you must edit.**
- `data/reference/omop_reference.csv` (after step 4): around 50 starter rows. The interim report mentions a curated testable subset, so expand this from the Ryan et al. (2013) supplement before step 7.
- `data/reference/ici_label_revisions.csv` (after step 8): around 40 starter rows for the seven ICIs. Verify each `label_revision_quarter` against the FDA label history at accessdata.fda.gov before step 9.

**Reproducibility.** Every script that involves randomness uses `RANDOM_STATE = 42`. Rerunning gives identical numbers.
