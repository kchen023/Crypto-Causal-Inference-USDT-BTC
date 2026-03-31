# Causal Inference on Crypto Transaction-Level Data: Does Tether Drive Bitcoin Prices?

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Domain](https://img.shields.io/badge/Domain-Financial%20Econometrics-orange)
![Methods](https://img.shields.io/badge/Methods-VAR%20%7C%20Granger%20%7C%20IRF%20%7C%20ARCH--LM%20%7C%20CUSUM-purple)

---

## Overview

Please find Python codes and outputs in Codes and Outputs.ipynb.

This project investigates whether **Tether (USDT) transaction volume causally influences Bitcoin (BTC) price movements**, using transaction-level blockchain data for Q1 2018. The analysis applies a rigorous financial econometrics pipeline, from raw big-data ingestion and entity detection, through stationarity and causality testing, to impulse response simulation and model validation, to empirically test a widely debated hypothesis in cryptocurrency research.

The broader research question connects to systemic risk in digital asset markets: *if a stablecoin issuer can print money and deploy it strategically, does that create measurable price pressure on BTC?* This project provides a reproducible, statistically grounded answer using on-chain data.

Project Goal: Investigating the causal relationship between USDT and BTC in Q1 2018.

Methodology: VAR, Granger Causality, ADF Test, AIC/BIC/HQIC, and Robustness Checks (ARCH-LM, CUSUM).

Key Words: Financial Econometrics, Causal Inference, On-Chain Forensics, Model Validation, Crypto Transaction Data Science.

Skills： Data Transformation/Merging/Alignment, Data Visualization, Time-Series/Statistical Modeling, On-Chain Forensics.

---

## Keywords & Technical Skills

**Econometric Methods:** Granger Causality · Vector Autoregression (VAR) · Orthogonalized Impulse Response Functions (OIRF) · ADF Unit Root Test · ARCH-LM Heteroskedasticity Test · CUSUM Structural Break Test · AIC / BIC / FPE / HQIC Lag Selection · Log Returns · Stationarity Testing

**Data Engineering:** Transaction-Level Big Data Processing · Mixed-Frequency Data Merging · Temporal Aggregation (Tick → Daily) · On-Chain Entity Classification · Graph-Based Wallet Fingerprinting · Quantile Threshold Design

**Financial Domain:** Cryptocurrency Market Microstructure · Stablecoin Issuance Mechanics · Whale Flow Tracking · Exchange Hot Wallet Detection · Market Manipulation Hypothesis Testing · Lead-Lag Relationship Analysis

**Tools & Libraries:** Python · Pandas · NumPy · Statsmodels · Matplotlib · Google Colab

---

## Data

| Dataset | Source | Frequency | Observations | Period |
|---|---|---|---|---|
| Bitcoin (BTC/USD) | EOD market data | Daily | 90 rows | Jan 1 – Mar 31, 2018 |
| Tether (USDT) transactions | On-chain blockchain data | Transaction-level (tick) | 567,035 rows | Jan 1 – Mar 31, 2018 |

**USDT transaction schema:** `block_time`, `sending_address`, `reference_address`, `amount`, `tx_type`

Three transaction types are present in the raw data: `Simple Send` (567,023), `Grant Property Tokens` (issuance, 11 events), and `Revoke Property Tokens` (burn, 1 event). This granularity allows us to distinguish between Tether minting, burning, and regular P2P transfers — a critical distinction for testing the manipulation hypothesis.

> ⚠️ **Data Note: **Please unzip Tether_Q1_2018.csv.zip in the same directory before running the notebook. Please also download EOD_BTC_data.csv in the same directory before unning the notebook.**

---

## Analysis Pipeline

```
Raw Data Ingestion
    │
    ├── BTC OHLCV (daily)
    └── USDT transactions (tick-level, 567K rows)
             │
             ▼
Step 1: Data Preview & Exploratory Analysis
    - Transaction type distribution
    - Unique address counts (132K senders, 158K receivers)
    - Top active wallets
             │
             ▼
Step 2: Data Transformation & Merging
    - Parse datetime, clean price strings
    - Aggregate USDT transactions: tick → daily volume
    - Inner join on date → merged daily dataset
             │
             ▼
Step 3: Time-Series Visualization
    - BTC/USD price trend, Q1 2018
    - USDT daily transaction volume
             │
             ▼
Step 4: Entity Classification (Digital Fingerprinting)
    - Treasury wallets: identified via "Grant Property Tokens" tx_type
      → 3MbYQMMmSkC3AgWkj9FMo5LsPTW1zBTwXL (primary Tether issuer)
    - Exchange hot wallets: ranked by "social reach" metric
      (unique_partners_out + unique_partners_in)
      → Top wallet: 86K+ combined partner reach (likely Bitfinex)
    - Whale detection: 99th percentile quantile threshold = $602,748 USDT
      → 11 whale-scale treasury outflows identified
             │
             ▼
Step 5: Causal Inference Pipeline
    │
    ├── 5.1 Stationarity Test (ADF)
    │       BTC log return: p < 0.0001 ✓ stationary
    │       USDT volume Δ%: p < 0.0001 ✓ stationary
    │       → Proceed with VAR in levels (no cointegration needed)
    │
    ├── 5.2 Lag Selection (Information Criteria)
    │       AIC → Lag = 1 | BIC → Lag = 0 | FPE → Lag = 1 | HQIC → Lag = 1
    │       → Optimal lag = 1 (AIC / FPE / HQIC consensus)
    │
    ├── 5.3 Granger Causality Test
    │       H₀: USDT volume does not Granger-cause BTC returns
    │       F-stat = 0.044, p = 0.834 → Fail to reject H₀
    │       → No statistically significant lead-lag relationship at daily frequency
    │
    ├── 5.4 Whale Flow Tracking
    │       99th pct threshold: $602,748 USDT
    │       Treasury whale outflows: 11 events
    │       → Qualitative overlay with price series
    │
    └── 5.5 Orthogonalized Impulse Response Function (VAR)
            Shock: 1 std-dev increase in USDT volume
            Response: BTC log return over 10 days
            → Negligible, statistically insignificant response
            → Confirms Granger null result
             │
             ▼
Step 6: Model Validation & Robustness Checks
    │
    ├── ARCH-LM Test (Heteroskedasticity)
    │       Tests for time-varying volatility (ARCH effects) in residuals
    │       Validates whether OLS/VAR standard errors are reliable
    │
    └── CUSUM Test (Structural Stability)
            Detects parameter instability over time
            Critical given BTC's 60%+ drawdown during Q1 2018
```

---

## Economic & Investment Intuition

### The Manipulation Hypothesis

A prominent hypothesis in crypto market research (Griffin & Shams, 2020) argues that Tether issuance was used to purchase Bitcoin during price downturns, artificially inflating prices during the 2017–2018 bull market. If true, a detectable lead-lag relationship should exist: large USDT issuance events should *Granger-cause* subsequent BTC price increases.

### What This Analysis Finds

The Granger causality test finds **no statistically significant causal relationship** between daily USDT transaction volume and BTC log returns at lag 1 (p = 0.834). The orthogonalized IRF corroborates this: a one standard deviation shock in USDT volume produces a negligible, economically trivial response in BTC returns that dissipates within 2 days and lies well within confidence bands.

### Why Daily Data May Not Be Enough

This null result does not exonerate Tether — it constrains the *mechanism* and *timing*:

1. **Frequency mismatch.** If price manipulation occurs over hours or minutes (intraday), aggregating to daily frequency destroys the signal. A 2018 Tether issuance event of $100M deployed over 4 hours would appear as a single daily observation, smoothing the price impact into noise.

2. **Short sample.** Q1 2018 contains only 90 daily observations. With lag = 1, the Granger test has limited statistical power to detect subtle effects. A longer panel (2017–2019) with structural break controls would substantially improve inference.

3. **Aggregation bias.** Summing all USDT transfers (including retail P2P) into a single daily volume series dilutes the signal from the 11 treasury-scale whale events. A cleaner test would isolate *Treasury-originated* flows specifically.

### Investment & Risk Implications

For practitioners assessing crypto market integrity:

- **Daily USDT issuance data cannot be used as a reliable BTC price signal** at any trading frequency. Factor models built on stablecoin issuance data at daily resolution are unlikely to generate alpha.
- **Any credible market manipulation test requires sub-daily (hourly or tick) data**, matched precisely to issuance timestamps — not aggregate daily volumes.
- **Structural break risk is high.** BTC lost ~60% of its value during Q1 2018. CUSUM stability testing is not cosmetic here; it is a prerequisite for valid inference. Models estimated on Q1 2018 data should not be extrapolated to other regimes without re-estimation.
- **Entity-level analysis is more informative than volume aggregates.** The Treasury wallet fingerprinting and whale flow tracking in Step 4 — tracking specific wallet addresses rather than aggregate flows — is a more targeted methodology for manipulation detection.

---

## Key Results Summary

| Test | Statistic | Result |
|---|---|---|
| ADF — BTC log return | p < 0.0001 | Stationary ✓ |
| ADF — USDT vol Δ% | p < 0.0001 | Stationary ✓ |
| Granger causality (lag=1) | F=0.044, p=0.834 | No causation |
| VAR Optimal Lag | AIC/FPE/HQIC → Lag 1 | Selected |
| OIRF (10-day horizon) | Negligible response | Confirms null |
| ARCH-LM | Residual diagnostics | See notebook |
| CUSUM | Parameter stability | See notebook |
| Whale events (99th pct) | 11 transactions ≥ $602K | Identified |
| Exchange wallets | Top: 86K social reach | Fingerprinted |

---

## Project Structure

```
crypto-causal-inference/
├── Big_Data_Analysis_Crypto_Causal_Inference.ipynb   # Main analysis notebook
├── README.md                                          # This file
└── requirements.txt                                   # Python dependencies
```

---

## Setup & Requirements

```bash
pip install -r requirements.txt
```

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
statsmodels>=0.14
```

**Data mounting (Google Colab):** Mount your Google Drive and place the data files at:
- `/content/drive/MyDrive/Colab/Fin_Econ/Data/EOD_BTC_data.csv`
- `/content/drive/MyDrive/Colab/Fin_Econ/Data/Tether_Q1_2018.csv`

---

## Caveats & Limitations

This project is intended as a **methodological demonstration**. Key limitations to note:

1. **Sample size.** Q1 2018 (90 daily observations) is a minimal sample for time-series inference. Findings should not be interpreted as definitive statements about the Tether–Bitcoin relationship.

2. **Frequency.** Daily aggregation likely masks intraday dynamics that are central to any manipulation hypothesis. Sub-daily data is needed for a rigorous test.

3. **Single quarter.** Q1 2018 is a high-volatility, bearish period with a structural break (peak-to-trough decline >60%). Results may not generalize to other regimes.

4. **Aggregation.** USDT volume includes all transaction types. Isolating Treasury-originated flows would yield a cleaner instrument for testing the specific manipulation channel.

5. **No causal identification.** Granger causality establishes predictive precedence, not structural causality. Ruling out confounders (e.g., macro risk-off sentiment driving both BTC and USDT redemptions simultaneously) requires instrumental variable or difference-in-differences designs.

---

## References

- Griffin, J. M., & Shams, A. (2020). *Is Bitcoin Really Untethered?* Journal of Finance, 75(4), 1913–1964.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis.* Springer.
- Hamilton, J. D. (1994). *Time Series Analysis.* Princeton University Press.

---

## Author

**Kai Chen**
PhD in Economics
[GitHub: kchen023](https://github.com/kchen023)
