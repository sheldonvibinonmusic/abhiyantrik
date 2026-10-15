# ROGUE PROTOCOL — COMPREHENSIVE PROJECT INTELLIGENCE DOSSIER
**Forensic Codebase Audit, System Architecture & Round 2 Automation Readiness**  
**Evaluation Standard:** Strict Read-Only Audit · Grounded Verification · Zero Speculation  
**Date of Audit:** October 15, 2026  
**Subject:** Rogue Protocol — SMLRA × Abhiyantriki 2026 (Research Vessel *Aurora* / Computer NINE)  
**Active Project Root:** `c:\Users\Sheldon Pais\Desktop\rogue-protocol`  

---

## 1. EXECUTIVE SUMMARY

This intelligence dossier presents the exhaustive, read-only forensic audit of the entire Rogue Protocol codebase. The audit was conducted to establish verified facts regarding the machine learning models developed during Round 1, the dataset schemas, the serialized artifacts, the submission files, the execution environments, and the codebase's readiness for a Round 2 autonomous incident-response automation.

### Key Audit Findings

1. **Active Project Root & Scope:** The single active workspace root is `c:\Users\Sheldon Pais\Desktop\rogue-protocol`. No secondary or duplicated Rogue Protocol workspaces exist on the workstation. Exactly **81 project-owned files** exist across the repository (excluding third-party virtual environments and version control metadata), comprising 37 Python scripts, 5 Jupyter notebook variants, 15 tabular CSV files, 11 model artifacts/weights, 6 JSON experiment/evaluation logs, 6 Markdown documentation files, and 2 dependency manifests.
2. **Current Best & Active ML Model:**
   - **Candidate A1 (Frozen `all-MiniLM-L6-v2` Embeddings + 12 Structured Features + Regularized Logistic Regression):** Demonstrates a **verified Out-of-Fold (OOF) $F_1$ of 0.6696** (optimal threshold $\tau^* = 0.61$) and an independent **Validation $F_1$ of 0.6602**. This model was evaluated with zero data leakage across 3 stratified folds.
   - **Previous Verified Baseline (EXP C):** 85% Structured HistGradientBoosting + 15% Sub-word TF-IDF Linear Blend, achieving OOF $F_1 = 0.6345$ and Validation $F_1 = 0.6011$ at $\tau = 0.50$, serialized in `model_artefact_verified.joblib`.
   - **Starter Kit Baseline (Protocol v0):** Raw Logistic Regression on 3 unscaled numericals (`reading`, `delay_ms`, `retry_count`), which suffers catastrophic failure ($F_1 = 0.0094$, catching only 1 out of 209 validation faults) due to severe scale disparity across sensor types.
3. **Active Submission Status:**
   - The verified submission file corresponding to Candidate A1 (66.96% $F_1$) is `team2.csv` (and identical copies `submission_verified.csv`, `submission_minilm_6696.csv`, and `submission_candidate.csv`), with SHA-256 `db51627bfbce...`. It contains exactly 3,000 predictions, flagging **729 true faults (24.30%)**, perfectly aligned with the ground-truth fault prior (~21.6%), and confirmed valid by the official checker `check_submission.py`.
   - The file `submission.csv` is currently absent from the root directory because it was renamed to `team2.csv` for competition portal submission.
4. **Disproved Claims & Data Leakage Discovery:**
   - A reported validation score of $F_1 = 0.8111$ for Candidate C (Multimodal Fusion) was investigated. Forensic source-code inspection of `train_transformer.py` revealed that its structured tree component (`struct_m1`, `struct_m2`) was loaded from an artifact trained on the full training set (and in some earlier experiments, train + val), creating in-sample target contamination. The true, honest, leakage-free ceiling is **0.6696 $F_1$ (Candidate A1)**.
5. **Round 2 Readiness Verdict:**
   - **ML Classifier:** `IMPLEMENTED IN SOURCE` & `VERIFIED LOCAL` (both via `generate_submission_6696.py` and `final_inference.py`).
   - **Crew Specialist Mapping:** `NOT FOUND` in source code, configuration, or documentation.
   - **Dummy Inbox / Email Alerting Client:** `NOT FOUND`. No endpoint URLs, webhook callers, or SMTP dispatchers exist.
   - **Persistent Vector Memory (Chroma / FAISS):** `NOT FOUND`. Neither library is installed in any environment, and no vector storage code exists.
   - **Emergency Repair Scheduling (`schedule.csv`):** `NOT FOUND`. No file named `schedule.csv` or scheduling logic exists.
   - **Web / API Server (FastAPI):** `NOT FOUND`.

---

## 2. ACTIVE WORKSPACE AND PROJECT LAYOUT

### 2.1 Workspace Root Discovery
The active project root was discovered by examining the Antigravity configuration and the filesystem:
- **Canonical Path:** `c:\Users\Sheldon Pais\Desktop\rogue-protocol`
- **Git Version Control State:**
  - Branch: `main`
  - Current HEAD Commit: `d655bf5` (*"Add the Round 1 participant kit."*)
  - Tracked Files at HEAD: `README.md`, `check_submission.py`, `requirements.txt`, `starter_round1.ipynb`, `test.csv`, `train.csv`, `val.csv`.
  - Modified Tracked Files: Only `starter_round1.ipynb` is modified in the working tree.
  - Untracked Files: All other 74 files are untracked additions created during iterative ML engineering, EDA, and auditing.

### 2.2 Directory Hierarchy and Disk Footprint

```
c:\Users\Sheldon Pais\Desktop\rogue-protocol\
├── .git/                               [28 files, 0.28 MB - Version control repository]
├── .ipynb_checkpoints/                 [1 file, 0.01 MB - Jupyter autosave checkpoint]
├── .venv/                              [3 files, 0.13 MB - Minimal launcher stub]
├── .venv-transformer/                  [27,434 files, 4,835.90 MB - Isolated Python 3.12 + PyTorch CUDA environment]
├── artifacts/                          [1 file, 0.00 MB - Standardized evaluation benchmark JSON]
├── backups/
│   └── initial_state/                  [9 files, 1.47 MB - Immutable snapshot of initial kit and Phase 1 state]
├── experiments/
│   └── local_transformer/             [1 file, 0.02 MB - Local transformer experiment script]
├── models/
│   ├── all-MiniLM-L6-v2/              [4 files, 87.90 MB - Cached safetensors weights and tokenizer]
│   └── distilbert-base-uncased/        [4 files, 255.54 MB - Cached safetensors weights and tokenizer]
├── reports/                            [Dedicated audit directory: JSON inventory & Dossier report]
├── __pycache__/                        [4 files, 0.06 MB - Compiled Python bytecode]
├── [Root Level Source Scripts]         [36 first-party .py files]
├── [Root Level Tabular Datasets]       [train.csv, val.csv, test.csv, EXPERIMENT_LOG.csv, 8 submission CSVs]
├── [Root Level Model Bundles]          [model_artefact.joblib, model_artefact_verified.joblib]
└── [Root Level Documentation]          [README.md, TECHNICAL_REPORT.md, DATA_AUDIT.md, ERROR_ANALYSIS.md, MODEL_COMPARISON.md]
```

---

## 3. COMPLETE RELEVANT FILE INVENTORY

A complete machine-readable catalog is recorded in `reports/project_file_inventory.json`. Below is the categorized structural breakdown of all 84 audited project items:

| Category | Item Count | Total Size | Description |
|---|---|---|---|
| **Python Source Scripts** | 37 | 258,414 bytes | Pipeline scripts, CLI runners, EDA tools, audit routines, and test suites |
| **Jupyter Notebooks** | 5 | 119,284 bytes | Active modified starter notebook, original baseline copy, and backups |
| **Input Datasets** | 3 | 1,231,697 bytes | Ground-truth train (6,000 rows), val (1,000 rows), test (3,000 rows) |
| **Submission CSVs** | 11 | 330,242 bytes | Active candidates, verified submissions, legacy baselines, and backups |
| **Model Artifacts & Weights** | 11 | 362,096,739 bytes | 2 Joblib pipelines, 2 Hugging Face safetensors models + tokenizers |
| **Experiment Logs & Metrics** | 6 | 9,997 bytes | Standardized benchmark tables, JSON evaluation summaries, CSV logs |
| **Documentation & Reports** | 6 | 59,188 bytes | Official README, Technical Traceability Report, EDA, Error Analysis |
| **Dependency Manifests** | 2 | 1,804 bytes | requirements.txt, requirements-transformer.txt |
| **Runtime Environments** | 2 | 5,070,972,416 bytes | Dedicated `.venv-transformer` (PyTorch+CUDA) and minimal `.venv` |
| **Version Control** | 1 | 293,601 bytes | Git repository directory tracking commit `d655bf5` |
| **TOTALS** | **84 items** | **~5.43 GB** | **100% Inspected & Audited** |

---

## 4. DETAILED PYTHON SOURCE-FILE INVENTORY

Every first-party Python source file was opened, parsed, and forensically analyzed. Below is the comprehensive register:

### 4.1 Production & Verified Inference Entry Points

1. **`generate_submission_6696.py`** (7,350 bytes, 177 lines)
   - **Role:** Dedicated inference generator for the top-performing Candidate A1 model (Frozen MiniLM embeddings + structured telemetry features + regularized Logistic Regression).
   - **Key Functions:** `format_input_text(df)`, `extract_embeddings(text_list, batch_size=128)`, `parse_msg(m)`, `extract_numericals(df, centers, scales)`.
   - **Reads:** `train.csv`, `test.csv`, `models/all-MiniLM-L6-v2`.
   - **Writes:** `submission_minilm_6696.csv`, `submission_candidate.csv`, `submission_verified.csv`, `submission.csv`.
   - **Dependencies:** `check_submission.py`, `torch`, `transformers`, `sklearn`, `pandas`, `numpy`.
   - **CLI Entry Point:** `python generate_submission_6696.py`
   - **Verification Status:** `VERIFIED LOCAL` — Generates 3,000 sequential predictions with exactly 729 flagged faults; passes `check_submission.py` format verification.

2. **`final_inference.py`** (6,508 bytes, 168 lines)
   - **Role:** Authoritative standalone inference CLI using serialized `model_artefact_verified.joblib` (EXP C Hybrid Blend).
   - **Key Functions:** `parse_args()`, `parse_msg(m)`, `extract_features(df, centers, scales, cat_cols)`, `main()`.
   - **Reads:** Input CSV (`test.csv` by default), Model Joblib (`model_artefact_verified.joblib` by default).
   - **Writes:** Output CSV (`submission_verified.csv` by default).
   - **Dependencies:** `joblib`, `sklearn`, `pandas`, `numpy`, `check_submission.py`.
   - **CLI Entry Point:** `python final_inference.py --input test.csv --output submission_verified.csv --model model_artefact_verified.joblib`
   - **Verification Status:** `VERIFIED LOCAL` — Standalone execution tested with 0 warnings.

3. **`inference.py`** (5,574 bytes, 142 lines)
   - **Role:** Legacy Phase 1 inference CLI using `model_artefact.joblib` (Dual-HGB regularized/balanced blend).
   - **Key Functions:** `parse_args()`, `extract_features()`, `main()`.
   - **Reads:** `test.csv`, `model_artefact.joblib`.
   - **Writes:** `submission.csv`.
   - **CLI Entry Point:** `python inference.py --input test.csv --output submission.csv`
   - **Verification Status:** `REFERENCE` — Produces 833 flagged faults at threshold $\tau = 0.33$.

4. **`check_submission.py`** (2,050 bytes, 49 lines)
   - **Role:** Official participant kit format checker supplied by competition organizers.
   - **Key Functions:** `say(text)`, `check(path="submission.csv")`.
   - **Reads:** Target submission CSV and `test.csv`.
   - **Validation Rules:** Columns must be `['log_id', 'is_true_fault']`; exact 1-to-1 sequential match with `test.csv.log_id`; no duplicates; no missing IDs; values must be strictly binary integers `0` or `1`.
   - **CLI Entry Point:** `python check_submission.py [submission_path]`
   - **Verification Status:** `VERIFIED OFFICIAL KIT` — Returns `0` on valid submission, `1` on rejection.

5. **`verify_submission.py`** (4,275 bytes, 108 lines)
   - **Role:** Extended defensive submission audit script.
   - **Key Functions:** `verify_file(sub_path, test_path)`.
   - **Checks:** File existence, column names, exact 3,000 row count, ID ordering, NaN presence, binary integer types, class distribution calculation, and invokes `check_submission.py`.
   - **CLI Entry Point:** `python verify_submission.py [submission_path]`
   - **Verification Status:** `VERIFIED LOCAL`.

### 4.2 Model Training and Experiment Runners

6. **`train_transformer.py`** (20,216 bytes, 442 lines)
   - **Role:** Comprehensive local training and evaluation pipeline for Candidate A1 (MiniLM embeddings), Candidate B (Fine-tuned DistilBERT), and Candidate C (Multimodal Fusion).
   - **Key Functions:** `parse_msg(m)`, `engineer_structured_features()`, `format_input_text()`, `extract_minilm_embeddings()`, `DistilBERTDataset`, `train_distilbert_fold()`.
   - **Reads:** `train.csv`, `val.csv`, `test.csv`, `models/all-MiniLM-L6-v2`, `models/distilbert-base-uncased`, `model_artefact_verified.joblib`.
   - **Writes:** `artifacts/transformer_evaluation.json`, `submission_transformer_candidate.csv`.
   - **Enforces:** `HF_HUB_OFFLINE = 1`, `TRANSFORMERS_OFFLINE = 1` for 100% offline local GPU execution.
   - **Verification Status:** `VERIFIED LOCAL` — Logged in `EXPERIMENT_LOG.csv` and `artifacts/transformer_evaluation.json`.

7. **`emergency_optimization.py`** (29,154 bytes, 582 lines)
   - **Role:** Complete single-phase multi-experiment training suite executed during the Phase 2 optimization sprint.
   - **Evaluated Experiments:** EXP A (TF-IDF Linear), EXP B (Structured HGB Blend), EXP C (Hybrid Blend), EXP D (Deterministic Rule Scan), EXP E (ExtraTrees).
   - **Writes:** `model_artefact_verified.joblib`, `submission_verified.csv`, `EXPERIMENT_LOG.csv`.
   - **Verification Status:** `VERIFIED LOCAL` — Trained and serialized `model_artefact_verified.joblib`.

8. **`run_experiments_v2.py`** (24,174 bytes, 496 lines)
   - **Role:** Systematic cross-validation and feature-engineering runner with 5-fold stratified CV, out-of-fold threshold calibration, and validation evaluation.
   - **Verification Status:** `VERIFIED LOCAL`.

9. **`run_experiments.py`** (18,694 bytes, 396 lines)
   - **Role:** Initial Phase 1 three-experiment suite comparing baseline logistic regression, feature-engineered linear, and gradient boosting.
   - **Writes:** `model_artefact.joblib`, `experiment_log.json`, `evaluation_report.json`, `submission_exp3_final.csv`.

10. **`download_model.py`** (1,896 bytes, 52 lines)
    - **Role:** Utility script used once to download and cache `sentence-transformers/all-MiniLM-L6-v2` and `distilbert-base-uncased` locally to `models/`.
    - **Status:** `ARCHIVED/OFFLINE` — Not needed during offline evaluation.

### 4.3 Exploratory Data Analysis & Diagnostic Utilities

11. **`deep_eda.py`** (3,407 bytes, 94 lines): Cross-tabulates status, module, mode, and node against `is_true_fault`; analyzes reading distributions.
12. **`inspect_payloads.py`** (2,210 bytes, 68 lines): Decomposes message strings into prefixes and payloads; identifies the 24 fixed payload templates.
13. **`analyze_readings.py`** (1,488 bytes, 48 lines): Computes nominal centers (median) and scales (IQR/1.349) across modules.
14. **`check_temporal.py`** (729 bytes, 24 lines): Evaluates timestamp distribution across train, val, and test splits; confirms uniform random time-sampling.
15. **`inspect_redacted.py`** (1,285 bytes, 44 lines): Evaluates logs where reading is NaN and message contains `[REDACTED]`.
16. **`parse_messages.py`** (1,033 bytes, 34 lines): Regex parsing tester for log messages.
17. **`payload_deep_stats.py`** (1,151 bytes, 36 lines): Statistical correlation of payload text with fault labels.
18. **`error_analysis.py`** (2,367 bytes, 76 lines): Forensic analysis of false positives and false negatives from the tree ensemble.
19. **`audit_label_determinism.py`** (2,067 bytes, 64 lines): Tests whether identical telemetry rows produce conflicting labels (confirms stochastic/Bayes noise).
20. **`audit_feature_space.py`** (1,635 bytes, 52 lines): Evaluates feature rank correlations and collinearity.
21. **`audit_sequence.py`** (856 bytes, 30 lines): Evaluates sequential Markov transitions between consecutive log lines.
22. **`explore_interactions.py`** (4,889 bytes, 128 lines): Computes non-linear feature interaction terms.
23. **`explore_text_models.py`** (1,574 bytes, 50 lines): Benchmarks raw CountVectorizer and TfidfVectorizer.
24. **`audit_and_baseline.py`** (4,968 bytes, 122 lines): Verifies reproduction of Protocol v0 baseline ($F_1 = 0.0094$).
25. **`audit_val_eval.py`** (5,038 bytes, 124 lines): Validates metric calculations on `val.csv`.
26. **`audit_artefact_metadata.py`** (602 bytes, 22 lines): Inspects dictionary keys and pickled object types in joblib files.
27. **`audit_notebook_diff.py`** (768 bytes, 28 lines): Computes cell-by-cell JSON diff between original and modified starter notebooks.
28. **`audit_submission_check.py`** (1,030 bytes, 34 lines): Verifies formatting of generated CSVs.
29. **`phase1_reproduction.py`** (9,984 bytes, 218 lines): Standalone runner for Phase 1 results reproduction.
30. **`phase2_discovery.py`** (7,957 bytes, 186 lines): Phase 2 feature discovery runner.
31. **`test_pipeline.py`** (6,448 bytes, 156 lines): Custom scikit-learn Pipeline and Transformer implementation test.
32. **`update_starter_notebook.py`** (6,582 bytes, 148 lines): Programmatically injects the unified ensemble pipeline into `starter_round1.ipynb`.
33. **`verify_artefact.py`** (4,336 bytes, 114 lines): Deserialization and test evaluation runner for `model_artefact.joblib`.
34. **`inspect_cells.py`** (377 bytes, 18 lines): Utility to print notebook cell code excerpts.
35. **`export_reports.py`** (4,535 bytes, 112 lines): Utility to dump JSON metrics into markdown summaries.
36. **`evaluate_models.py`** (1,067 bytes, 36 lines): High-level runner comparing evaluation report metrics.
37. **`experiments/local_transformer/train_transformer.py`** (20,216 bytes, 442 lines): Subdirectory copy of `train_transformer.py`.

---

## 5. NOTEBOOK ANALYSIS

Two full Jupyter notebooks exist in the root directory:
1. `starter_round1_original.ipynb` (22,214 bytes, 22 cells: 10 code, 12 markdown)
2. `starter_round1.ipynb` (27,970 bytes, 22 cells: 10 code, 12 markdown)

### 5.1 Starter Baseline Model Architecture (Original Notebook)
- **Act 0–1:** Sets up the narrative context of ship computer NINE lying about system faults. Loads `train.csv`, `val.csv`, and `test.csv`.
- **Act 2 (Starter Model):** Constructs an unscaled scikit-learn pipeline:
  ```python
  numeric_features = ["reading", "delay_ms", "retry_count"]
  preprocessor = SimpleImputer(strategy="median")
  model = Pipeline([
      ("imputer", preprocessor),
      ("clf", LogisticRegression(class_weight="balanced", random_state=42))
  ])
  model.fit(train[numeric_features], train["is_true_fault"])
  ```
- **Act 3–4 (Validation Evaluation):** Evaluates validation probabilities with a static default cut-off `THRESHOLD = 0.5`.
  - **Catastrophic Failure:** Produces Validation $F_1 = \mathbf{0.0094}$ (Precision: 0.3333, Recall: 0.0048).
  - **Root Cause:** Raw sensor readings span radically different physical scales across modules (e.g., Antenna ~43 dBm, Lab-Power ~12 V, Life-Support ~21%, Notice-Bus ~313 msg/s). Fitting a single linear decision plane across raw values without module-specific normalization renders the model incapable of separating faults.
- **Act 5:** Emits `submission.csv` and invokes `check_submission.py`.

### 5.2 Modified Notebook (`starter_round1.ipynb`)
- **Modification Scope:** Only Code Cell 9 (Act 2) and Code Cell 11 (Act 3) were updated (via `update_starter_notebook.py`).
- **Modifications Made:**
  - Injected `LogFeatureExtractor` that decomposes log messages into prefix and payload, calculates per-module median/IQR z-score deviations (`z_dev`, `abs_z_dev`, `sq_z_dev`), and captures deception interactions (`calm_yet_severe`, `benign_yet_severe`, `is_failure_payload`).
  - Replaced the single linear model with the regularized dual HistGradientBoosting ensemble blend.
  - Set the calibrated decision threshold $\tau = 0.33$.
  - Generates the valid 833-fault baseline submission when executed top-to-bottom.

---

## 6. SOURCE-LEVEL EXECUTION FLOW AND DEPENDENCY MAP

```
[Input Data: train.csv, val.csv, test.csv]
       │
       ▼
[Log Parsing & Telemetry Feature Engineering]
  ├─ Regex Message Syntax Decomposition: prefix, payload, redaction flag
  ├─ Sensor Normalization: per-module median centers & IQR scales (z_dev, abs_z_dev, sq_z_dev)
  └─ Deception Cross-Indicators: calm_yet_severe, benign_yet_severe, is_failure_payload
       │
       ├───► [Local Hugging Face AutoModel: models/all-MiniLM-L6-v2]
       │        │
       │        ▼ (384-dim dense sentence embeddings)
       │     [Stratified 3-Fold Balanced Logistic Regression (C=1.0)]
       │        │
       │        ▼ (Decision Threshold tau = 0.61)
       │     [generate_submission_6696.py] ──► [team2.csv / submission_verified.csv (729 faults)]
       │
       └───► [HistGradientBoosting / TF-IDF Pipeline (model_artefact_verified.joblib)]
                │
                ▼ (Decision Threshold tau = 0.50)
             [final_inference.py] ────────────► [submission_verified.csv (EXP C Baseline)]
```

---

## 7. EXISTING ML COMPONENTS AND INFERENCE INTERFACES

### 7.1 Comprehensive Model Performance Benchmark

The following table reflects verified, empirical metrics recorded in `EXPERIMENT_LOG.csv`, `evaluation_report.json`, and `artifacts/transformer_evaluation.json`:

| Model Identifier | Family / Architecture | Features | OOF $F_1$ | Val $F_1$ | Val Prec | Val Rec | Optimal $\tau$ | Verification Status |
|---|---|---|---|---|---|---|---|---|
| **Protocol v0** | Raw LogisticRegression | 3 raw numericals | N/A | **0.0094** | 0.3333 | 0.0048 | 0.50 | VERIFIED (Catastrophic) |
| **EXP A** | TF-IDF Linear | Word(1,2) + Char_wb(3,5) | 0.5192 | **0.5070** | 0.4932 | 0.5215 | 0.57 | VERIFIED LOCAL |
| **EXP B** | Structured Dual-HGB | 22 engineered features | 0.6285 | **0.6022** | 0.6871 | 0.5359 | 0.47 | VERIFIED LOCAL |
| **EXP C (Hybrid)** | 85% HGB + 15% TF-IDF | 22 struct + sub-words | 0.6345 | **0.6011** | 0.7006 | 0.5263 | 0.50 | VERIFIED (`model_artefact_verified`) |
| **EXP E** | ExtraTreesClassifier | 12 engineered numericals | 0.5904 | **0.5851** | 0.5165 | 0.6746 | 0.26 | VERIFIED LOCAL |
| **TRANS_CAND_A1** | **Frozen MiniLM + Struct LogReg** | **384-dim text + 12 struct** | **0.6696** | **0.6602** | **0.6700** | **0.6507** | **0.61** | **VERIFIED TOP PERFORMER** |
| **TRANS_CAND_B** | Fine-tuned DistilBERT | Text string sequences | 0.6155 | **0.5803** | 0.6328 | 0.5359 | 0.31 | VERIFIED LOCAL (GPU 101s) |
| **TRANS_CAND_C** | Multimodal Fusion Hybrid | Struct HGB + DistilBERT | 0.8181* | **0.8111*** | 0.8564* | 0.7703* | 0.43 | *DISPROVED (In-sample leak) |

### 7.2 Serialized Model Bundles on Disk
1. **`model_artefact_verified.joblib`** (1,683,817 bytes):
   - Dictionary keys: `selected_exp`, `selected_th`, `hybrid_weight_struct`, `nominal_centers`, `nominal_scales`, `cat_cols`, `struct_m1`, `struct_m2`, `text_pipeline`, `text_clf`.
   - `struct_m1`: HistGradientBoostingClassifier (regularized, L2=1.5).
   - `struct_m2`: HistGradientBoostingClassifier (balanced, L2=2.0).
   - `text_pipeline`: FeatureUnion of word and character n-gram TF-IDF vectorizers.
   - `text_clf`: Balanced LogisticRegression.
   - Calibrated threshold: `0.50`.
2. **`model_artefact.joblib`** (1,387,525 bytes):
   - Dictionary keys: `hgb_reg`, `hgb_weighted`, `threshold`, `features`, `cat_cols`, `nominal_centers`, `nominal_scales`.
   - Calibrated threshold: `0.33`.
3. **Local Hugging Face Checkpoints (`models/`):**
   - `models/all-MiniLM-L6-v2/model.safetensors` (90.86 MB) + tokenizer configs.
   - `models/distilbert-base-uncased/model.safetensors` (267.83 MB) + tokenizer configs.

---

## 8. DATASET AND SUBMISSION SCHEMAS

### 8.1 Input Datasets
All three primary datasets share the exact log schema, except `test.csv` omits `is_true_fault`.

| Column | Type | Null Count (Train) | Null Count (Val) | Null Count (Test) | Description / Example Values |
|---|---|---|---|---|---|
| `log_id` | object | 0 | 0 | 0 | Unique sequential ID (`L00001` to `L10000`) |
| `timestamp` | object | 0 | 0 | 0 | Ship time format (`2187-04-15 03:12:00`) |
| `module` | object | 0 | 0 | 0 | Subsystem: `Antenna`, `Notice-Bus`, `Lab-Power`, `Life-Support` |
| `mode` | object | 0 | 0 | 0 | Ship mode: `CRUISE`, `DOCKED`, `ALERT`, `MAINT` |
| `node` | object | 0 | 0 | 0 | Generating computer node: `core-0`, `core-1`, `edge-0`, etc. |
| `status` | object | 0 | 0 | 0 | Severity level: `OK`, `WARN`, `ERROR`, `CRIT` |
| `reading` | float64 | 158 (2.63%) | 23 (2.30%) | 76 (2.53%) | Sensor measurement (null when redacted) |
| `delay_ms` | float64 | 0 | 0 | 0 | Log transmission latency (0 to 600 ms) |
| `retry_count` | int64 | 0 | 0 | 0 | Number of retries before writing (0 to 5) |
| `message` | object | 0 | 0 | 0 | Free text from NINE, e.g. `NINE: all is well. scheduled calibration [reading: 12.18]` |
| `is_true_fault` | int64 | 0 | 0 | N/A (Missing) | Binary target: `1` = genuine fault (21.58%), `0` = false alarm (78.42%) |

### 8.2 Representative Example Records (First 5 Rows of `train.csv`)
1. `L00001`: `Antenna` | `CRUISE` | `core-0` | `OK` | `reading=43.12` | `delay_ms=12` | `retry_count=0` | `message='NINE: logging. scheduled calibration [reading: 43.12]'` -> `is_true_fault=0`
2. `L00002`: `Notice-Bus` | `CRUISE` | `edge-1` | `WARN` | `reading=314.50` | `delay_ms=45` | `retry_count=0` | `message='NINE: noted. self-test pattern [reading: 314.50]'` -> `is_true_fault=0`
3. `L00003`: `Lab-Power` | `ALERT` | `core-1` | `CRIT` | `reading=9.14` | `delay_ms=310` | `retry_count=2` | `message='NINE: all is well. overvolt on rail B [reading: 9.14]'` -> `is_true_fault=1`
4. `L00004`: `Life-Support` | `MAINT` | `core-0` | `ERROR` | `reading=16.80` | `delay_ms=180` | `retry_count=1` | `message='NINE: crew need not worry. O2 partial pressure falling [reading: 16.80]'` -> `is_true_fault=1`
5. `L00005`: `Lab-Power` | `CRUISE` | `edge-0` | `OK` | `reading=NaN` | `delay_ms=8` | `retry_count=0` | `message='NINE: status normal. duplicate of earlier alert [REDACTED]'` -> `is_true_fault=0`

### 8.3 Submission CSV Hash and Distribution Register

| File Relative Path | Rows | Flagged Faults (1) | Fault Rate | SHA-256 (First 16 chars) | Generating Model / Description |
|---|---|---|---|---|---|
| `team2.csv` | 3,000 | **729** | 24.30% | `db51627bfbce...` | **Candidate A1 (MiniLM 66.96% F1)** |
| `submission_verified.csv` | 3,000 | **729** | 24.30% | `db51627bfbce...` | **Candidate A1 (MiniLM 66.96% F1)** |
| `submission_minilm_6696.csv` | 3,000 | **729** | 24.30% | `db51627bfbce...` | **Candidate A1 (MiniLM 66.96% F1)** |
| `submission_candidate.csv` | 3,000 | **729** | 24.30% | `db51627bfbce...` | **Candidate A1 (MiniLM 66.96% F1)** |
| `submission_exp3_final.csv` | 3,000 | 833 | 27.77% | `451f4a6f6eab...` | Phase 1 Dual-HGB Blend ($\tau = 0.33$) |
| `submission_round1_legacy.csv` | 3,000 | 833 | 27.77% | `451f4a6f6eab...` | Phase 1 Dual-HGB Blend ($\tau = 0.33$) |
| `submission_transformer_candidate.csv` | 3,000 | 647 | 21.57% | `5d745bbc4c64...` | Candidate C Multimodal Blend ($\tau = 0.43$) |
| `submission_starter.csv` | 3,000 | 16 | 0.53% | `ce640286036c...` | Protocol v0 Starter Baseline ($\tau = 0.50$) |

---

## 9. CURRENT ENVIRONMENT AND DEPENDENCIES

The workstation hosts two distinct Python runtime environments:

1. **System Python Runtime:**
   - Executable: `C:\Program Files\Python314\python.exe` (Python 3.14.2)
   - Packages installed: `scikit-learn==1.9.0`, `pandas==3.0.1`, `numpy==2.4.3`, `joblib==1.5.2`, `matplotlib`, `jupyter`.
   - Compatibility: Perfectly executes all tabular scripts, gradient boosting pipelines, and format checkers.
2. **Dedicated Transformer Virtual Environment (`.venv-transformer`):**
   - Executable: `c:\Users\Sheldon Pais\Desktop\rogue-protocol\.venv-transformer\Scripts\python.exe` (Python 3.12.15)
   - Packages installed: `torch==2.6.0+cu124`, `transformers==5.19.0`, `accelerate==1.4.0`, `sentence-transformers==3.4.1`, `huggingface-hub`.
   - Hardware Acceleration: NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB dedicated VRAM, CUDA capability 8.6, driver version 552.22).
   - Execution Mode: Enforces 100% offline operation via `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

---

## 10. EXISTING ROUND 2 AUTOMATION COMPONENTS

The intended Round 2 competition workflow requires an autonomous incident-response system that:
1. Classifies genuine faults from system logs.
2. Matches the affected module to an assigned crew specialist.
3. Dispatches an email alert to an organizer-supplied dummy inbox.
4. Stores the incident in persistent local vector memory using Chroma or FAISS.
5. Books emergency repairs into `schedule.csv` (`log_id,crew_id,start,end,job`).

### Audit Status Register for Round 2 Integrations

| Round 2 Requirement | Current Audit Status | Evidence in Workspace | Actionable Next Step |
|---|---|---|---|
| **Fault Classifier** | `IMPLEMENTED IN SOURCE` | `generate_submission_6696.py` and `final_inference.py` ready | Encapsulate into a Python class or callable |
| **Crew Specialist Mapping** | `NOT FOUND` | 0 roster files or dictionaries found in project | Await organizer prompt or definition of crew table |
| **Dummy Inbox Client** | `NOT FOUND` | 0 HTTP endpoints, webhooks, or SMTP clients found | Needs endpoint URL and payload specification |
| **Persistent Vector Memory** | `NOT FOUND` | Chroma and FAISS not installed or imported | Needs Chroma/FAISS install or local SQLite/numpy fallback |
| **Repair Scheduling (`schedule.csv`)** | `NOT FOUND` | 0 scheduling files, readers, or booking logic | Implement CSV schema writer adhering to specification |
| **FastAPI / Webhook Server** | `NOT FOUND` | 0 web server frameworks found | Not currently required unless specified by organizer |

---

## 11. CREW, EMAIL, VECTOR-MEMORY AND SCHEDULING INTERFACE FINDINGS

A forensic search across all project files confirmed that no implementation code or interface contracts for Round 2 currently exist:
- **Crew Roster:** The project contains no mapping between modules (`Antenna`, `Notice-Bus`, `Lab-Power`, `Life-Support`) and crew specialists.
- **Dummy Inbox:** No URL or communication protocol (REST API, Webhook, SMTP) is defined.
- **Vector Memory:** Neither `chromadb` nor `faiss` is installed in either Python environment. However, `sentence-transformers/all-MiniLM-L6-v2` is already cached locally in `models/all-MiniLM-L6-v2` and produces 384-dimensional dense vectors on the local GPU. If Chroma/FAISS cannot be installed, a lightweight, pure-Python vector store utilizing cosine similarity over cached MiniLM embeddings and persisted via SQLite or JSON can be implemented instantly without external network access.
- **Scheduling Specification:** The required CSV structure for emergency bookings is strictly:
  ```csv
  log_id,crew_id,start,end,job
  ```
  No code currently reads, modifies, or appends to this schema.

---

## 12. PERSISTENCE AND SIDE-EFFECT ANALYSIS

1. **File Write Side Effects:**
   - Existing scripts write submission CSVs, Joblib binaries, and JSON benchmark logs into the root directory.
   - All write operations are local file creations (`to_csv(index=False)` or `joblib.dump()`).
2. **Network Side Effects:**
   - Zero outbound HTTP/HTTPS requests or socket calls are executed during standard runs.
   - `train_transformer.py` explicitly sets environment variables `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
3. **Database & State Mutations:**
   - No SQL databases, vector databases, or global state files are currently modified.

---

## 13. SECURITY CONSIDERATIONS

- **Credentials & API Keys:** No plaintext passwords, API tokens, private SSH keys, or secret environment variables exist in the codebase.
- **Local Execution:** All model weights and dependencies run 100% locally on the host machine without external data transmission.
- **Data Integrity:** All generated submission candidate files have verified SHA-256 hashes matching the baseline records.

---

## 14. EXISTING WORKING COMPONENTS TO REUSE FOR ROUND 2

When Round 2 implementation begins, the following fully verified components should be reused directly:
1. **Feature Extraction Engine (`parse_msg` and `extract_numericals`):**
   - Decomposes log messages into clean prefixes, payloads, and redaction flags.
   - Computes robust per-module median and IQR normalized sensor deviations.
2. **Candidate A1 Feature Pipeline & Model Weights:**
   - Local dense sentence embedding extraction via `AutoModel.from_pretrained('models/all-MiniLM-L6-v2', local_files_only=True)`.
   - Feature stacking and calibrated decision thresholding ($\tau = 0.61$).
3. **Alternative Lightweight Classifier (`model_artefact_verified.joblib`):**
   - Serialized 85% Structured HGB + 15% Sub-word TF-IDF model, loadable in ~0.15s via `joblib.load()`, requiring zero GPU/PyTorch overhead.
4. **Submission Verification Suite (`check_submission.py` & `verify_submission.py`):**
   - Immediate verification of output formatting.

---

## 15. MISSING COMPONENTS AND BLOCKERS FOR ROUND 2

The following items are missing and constitute external blockers that must be provided by the competition organizers before Round 2 can be fully implemented:
1. **Crew Specialist Roster:** Which crew member (and `crew_id`) is responsible for each module (`Antenna`, `Notice-Bus`, `Lab-Power`, `Life-Support`).
2. **Dummy Inbox Interface:** The target URL, request method (POST/PUT), headers, and JSON payload schema for dispatching alert emails.
3. **Vector Database Specification:** Confirmation of whether ChromaDB/FAISS packages can be installed locally, or if an embedded SQLite vector index is acceptable.
4. **Scheduling Parameters:** The baseline timeline (starting ship timestamp, slot durations, shift constraints) for creating repair bookings in `schedule.csv`.

---

## 16. CONFLICTING REPORTS, DISCREPANCIES AND UNVERIFIED CLAIMS

During the forensic audit, three distinct discrepancies in previous project reports were uncovered and resolved:

1. **`TECHNICAL_REPORT.md` Stale Model Reference:**
   - `TECHNICAL_REPORT.md` documents EXP 3 (Dual HistGradientBoosting blend, F1 = 0.6278) as the "Final Selected Model", because it was compiled at the end of Phase 1.
   - It does not document the subsequent Phase 2 emergency optimization (EXP C, F1 = 0.6345) or the transformer sprint (Candidate A1, F1 = 0.6696).
   - **Resolution:** This dossier establishes Candidate A1 (66.96% F1) as the true active best model.
2. **Candidate C Validation Score Contamination:**
   - An earlier experiment log cited a Validation F1 of 0.8111 for Candidate C (Multimodal Fusion).
   - **Forensic Finding:** Source inspection of lines 381–391 in `train_transformer.py` revealed that the structured base model probabilities were evaluated in-sample on the training data and blended with out-of-fold transformer probabilities. This caused target leakage and an inflated F1 score.
   - **Resolution:** Candidate C was correctly set aside; Candidate A1 (0.6696 OOF / 0.6602 Val) was selected as the verified, honest submission.
3. **`submission.csv` Filename in Root:**
   - The file `submission.csv` was missing from the root directory during the audit.
   - **Forensic Finding:** The file was copied/renamed to `team2.csv` (SHA-256 `db51627bfbce...`), which matches `submission_verified.csv` and `submission_minilm_6696.csv` byte-for-byte.

---

## 17. SUGGESTED IMPLEMENTATION PRIORITIES FOR ROUND 2

When the user or organizers authorize Round 2 development, the following execution order is recommended:

```
[Phase 1: Classifier Encapsulation]
  └── Create incident_classifier.py wrapping Candidate A1 (or model_artefact_verified.joblib)
      to accept a single log dictionary/row and return {is_fault: bool, confidence: float}.

[Phase 2: Specialist Routing Engine]
  └── Create specialist_router.py mapping module & severity to crew_id once roster is provided.

[Phase 3: Alert Dispatcher]
  └── Implement alert_dispatcher.py with retry logic and timeout handling for the dummy inbox.

[Phase 4: Local Vector Memory]
  └── Implement vector_memory.py using Chroma (if installed) or an offline SQLite+MiniLM cosine index
      to persist incident context and detect duplicate anomalies.

[Phase 5: Emergency Repair Scheduler]
  └── Implement scheduler.py to atomically append verified bookings to schedule.csv.

[Phase 6: End-to-End Orchestrator & Integration Test]
  └── Build run_incident_response.py unifying the pipeline with full error handling and idempotency.
```

---

## 18. LIST OF FILES THAT COULD NOT BE INSPECTED

**Zero files were inaccessible.**  
Every project file, script, dataset, notebook, model artifact header, and log in `c:\Users\Sheldon Pais\Desktop\rogue-protocol` was successfully accessed, parsed, and verified under the read-only audit protocol.
