# ROGUE PROTOCOL — DATA AND LABEL AUDIT REPORT (PHASE 2)

**Dataset:** Rogue Protocol Round 1 (`train.csv`, `val.csv`, `test.csv`)  
**Audit Protocol:** Descriptive statistics and hypothesis validation conducted strictly on `train.csv`, with distribution shift tests between `train.csv` and `val.csv`.

---

## 1. DATASET DIMENSIONS, SCHEMA & TARGET PREVALENCE

| Split | Rows | Columns | Target Available | Positive Class Prevalence ($y=1$) | Negative Class Count ($y=0$) |
|---|---|---|---|---|---|
| `train.csv` | 6,000 | 11 | Yes (`is_true_fault`) | 1,295 (21.58%) | 4,705 (78.42%) |
| `val.csv` | 1,000 | 11 | Yes (`is_true_fault`) | 209 (20.90%) | 791 (79.10%) |
| `test.csv` | 3,000 | 10 | No (Competition test) | Unlabelled | Unlabelled |

- **Severe Class Imbalance:** Positive class frequency is ~21.5% (approx 1:3.7). This confirms that a naive all-zero classifier scores 0.0000 $F_1$, and standard symmetric thresholding ($\tau = 0.50$) heavily suppresses recall.
- **Target Integrity:** Target is strictly binary integer $\in \{0, 1\}$. 0 nulls across train and val.

---

## 2. MISSING DATA & THE REDACTION PHENOMENON

- **Column-wise Missingness:** Across all 11 columns, only `reading` contains missing values:
  - `train.csv`: 158 nulls (2.63%)
  - `val.csv`: 23 nulls (2.30%)
  - `test.csv`: 76 nulls (2.53%)
- **Deterministic Mechanism:** In every single null row, the corresponding `message` column contains `[<Module> REDACTED]`.
- **Target Association:** In `train.csv`, logs with `REDACTED` readings have a **56.33% fault rate** ($89 / 158$). This is 2.6× the baseline fault rate, confirming that computer NINE actively redacts sensor values to conceal genuine system faults.

---

## 3. CATEGORICAL FEATURE DISTRIBUTIONS & FAULT RATES

### 3.1 Subsystem Module (`module`)
| Module | Train Count | Faults ($y=1$) | Noise ($y=0$) | Fault Rate |
|---|---|---|---|---|
| Lab-Power | 1,636 | 344 | 1,292 | 21.03% |
| Antenna | 1,621 | 348 | 1,273 | 21.47% |
| Life-Support | 1,381 | 303 | 1,078 | 21.94% |
| Notice-Bus | 1,362 | 300 | 1,062 | 22.03% |

*Finding:* Baseline fault prevalence is nearly identical across all four modules (~21%–22%). The fault mechanism is cross-subsystem.

### 3.2 Alarm Status (`status`)
| Status | Train Count | Faults ($y=1$) | Noise ($y=0$) | Fault Rate |
|---|---|---|---|---|
| `OK` | 2,677 | 310 | 2,367 | 11.58% |
| `WARN` | 1,306 | 153 | 1,153 | 11.72% |
| `ERROR` | 983 | 365 | 618 | 37.13% |
| `CRIT` | 1,034 | 467 | 567 | 45.16% |

*Finding:* While `CRIT` and `ERROR` logs have elevated fault rates, more than 54% of `CRIT` alarms are false alarms (crying wolf), and 310 genuine faults are concealed inside `OK` status logs (lying).

### 3.3 Message Prefix Deception Patterns (`prefix`)
| Prefix Type | Train Count | Faults ($y=1$) | Noise ($y=0$) | Fault Rate |
|---|---|---|---|---|
| `crew need not worry` | 567 | 207 | 360 | **36.51%** |
| `all is well` | 524 | 190 | 334 | **36.26%** |
| `nothing to see` | 547 | 190 | 357 | **34.73%** |
| `logging` | 1,472 | 249 | 1,223 | 16.92% |
| `none` (no prefix) | 1,427 | 230 | 1,197 | 16.12% |
| `noted` | 1,463 | 229 | 1,234 | 15.65% |

*Finding:* Logs where NINE explicitly provides reassurance (`all is well`, `crew need not worry`, `nothing to see`) have more than double the fault rate of routine operational logs.

---

## 4. PER-MODULE SENSOR ANOMALY INFLECTION POINTS

Analysis of non-fault (`is_true_fault == 0`) reading distributions establishes the normal physical operating bands:

| Module | Sensor Units | Normal Median ($\mu_0$) | Normal Std ($\sigma_0$) | Lower Normal Bound ($q_{05}$) | Upper Normal Bound ($q_{95}$) | Fault Rate ($< q_{05}$) | Fault Rate ($[q_{05}, q_{95}]$) | Fault Rate ($> q_{95}$) |
|---|---|---|---|---|---|---|---|---|
| **Lab-Power** | Volts (V) | 12.18 V | 1.05 V | 10.57 V | 14.05 V | **44.35%** | 15.06% | **52.31%** |
| **Life-Support** | $\%O_2$ | 21.11 % | 1.67 % | 18.68 % | 24.22 % | **51.82%** | 15.36% | **49.52%** |
| **Antenna** | Power (dBm) | 43.13 dBm | 7.95 dBm | 30.85 dBm | 56.75 dBm | **51.54%** | 13.61% | **56.03%** |
| **Notice-Bus** | msgs/sec | 313.09 msg/s | 56.31 msg/s | 227.87 msg/s | 412.78 msg/s | **43.62%** | 16.07% | **51.38%** |

*Finding:* Sensor readings exhibit a distinct U-shaped anomaly profile. Within nominal operating bounds, fault rates drop to ~14%–16%. Excursions into either tail spike the fault probability to 44%–56%.

---

## 5. TRAIN VS VALIDATION DISTRIBUTION SHIFT TESTS

Statistical hypothesis tests comparing `train.csv` (6,000 logs) and `val.csv` (1,000 logs):
- **Numerical Features (Kolmogorov-Smirnov 2-Sample Test):**
  - `reading`: $KS = 0.0384, p = 0.1644$ (Fail to reject $H_0$; no shift).
  - `delay_ms`: $KS = 0.0247, p = 0.6674$ (Fail to reject $H_0$; no shift).
  - `retry_count`: $KS = 0.0062, p = 1.0000$ (Fail to reject $H_0$; no shift).
- **Categorical Features (Pearson Chi-Square Contingency Test):**
  - `module`: $\chi^2 = 3.42, p = 0.3312$ (No shift).
  - `mode`: $\chi^2 = 2.23, p = 0.5269$ (No shift).
  - `node`: $\chi^2 = 7.20, p = 0.5153$ (No shift).
  - `status`: $\chi^2 = 8.75, p = 0.0328$ (Mild variation in alarm proportions between splits).

---

## 6. ARTIFACT & IDENTIFIER AUDIT

- **Log ID & Row Order:**
  - Correlation between `log_id` integer and `is_true_fault`: **$r = 0.0072$**.
  - Fault rate across 5 equal `log_id` chronological quintiles: $21.75\%, 20.67\%, 20.83\%, 23.75\%, 20.92\%$.
  - Per Hard Constraint 5, `log_id` contains zero legitimate predictive signal and must not be used as an input feature.
- **Autocorrelation:**
  - Chronological lag-1 autocorrelation of `is_true_fault` is **$0.0213$**. Logs do not form a Markovian burst sequence; stratified cross-validation across independent logs is valid and leakage-free.
