# GridRunners: NYC Yellow Taxi Predictive Suite

**DSCI 592 Capstone Project**  
*Drexel University — May 2026*

## Team Members
- Caleb Solomons
- Mubeen Yaqub
- Mai Lam
- Anh Minh Tran

## Project Overview

GridRunners is an end-to-end machine learning project built on the 2023 NYC Yellow Taxi dataset. The project goes beyond exploratory analysis by delivering a unified predictive suite that estimates **fare**, **trip duration**, **tip likelihood**, and **demand** from a single shared feature contract and a single deployed interface. The system was designed to support booking-time prediction, meaning every feature used by the models is available before the trip begins. [1]

The project uses the public NYC Taxi and Limousine Commission (TLC) yellow taxi dataset, covering roughly 38.3 million raw trips from 2023. After a multi-stage cleaning pipeline and sanity filtering, the modeling workflow retains about 37.0 million usable trips, with additional sampled subsets used for evaluation and hyperparameter tuning. [1]

## Project Goals

- Build four coordinated machine learning models from one consistent feature schema. [1]
- Predict trip fare as a regression task. [1]
- Predict trip duration as a regression task. [1]
- Predict tip likelihood as a calibrated binary classification task. [1]
- Predict taxi demand as a count regression task. [1]
- Deliver all predictions through a single Streamlit application. [1]

## Dataset

- **Source:** NYC TLC 2023 Yellow Taxi public dataset. [1]
- **Raw size:** Approximately 38.3 million trips, 19 raw columns, about 14 GB. [1]
- **After cleaning:** Approximately 37.0 million rows and 35 columns after sanity filtering and feature engineering. [1]
- **Evaluation sample:** 1.5 million-row stratified sample. [1]
- **Tuning sample:** 250,000 rows for Optuna-based hyperparameter search. [1]

## Core Features

The project uses a shared 16-column feature contract across all models so that one trip row can be passed into each prediction task consistently. This contract includes six numeric features and ten categorical features, with all columns restricted to information knowable at booking time. [1]

### Numeric Features
- `trip_distance`
- `haversine_mi`
- `manhattan_mi`
- `passenger_count`
- `temp_f`
- `precip_in`  
[1]

### Categorical Features
- `pickup_hour`
- `pickup_dow`
- `is_weekend`
- `is_rainy`
- `PU_borough`
- `DO_borough`
- `VendorID`
- `is_us_holiday`
- `is_nyc_event`
- `RatecodeID`  
[1]

### Leakage Control

To preserve realistic deployment behavior, the pipeline excludes post-trip or receipt-derived fields such as `tip_amount`, `tolls_amount`, `total_amount`, `congestion_surcharge`, `airport_fee`, and `trip_duration_min` from model training. The tip model also excludes payment type because that choice is not known at booking time. [1]

## Exploratory Insights

Exploratory analysis shaped the final modeling choices in three major ways:

- **Temporal patterns:** Hour of day strongly shifts fare, tip behavior, and demand, with visible commuter and rush-period structure. [1]
- **Payment behavior:** Cash trips often appear with zero recorded tip in TLC data, which changes how the tip target should be interpreted. [1]
- **Spatial structure:** Borough geography strongly influences trip distance, fare tails, and demand concentration, motivating engineered geographic distance proxies. [1]

## Modeling Approach

The pipeline starts from cleaned parquet data and applies task-specific filters, feature engineering, stratified sampling, Optuna tuning, cross-validation, and final refitting. Different XGBoost model heads are then used for each task while preserving one shared input schema. [1]

| Target | Task Type | Model Framing |
|--------|-----------|---------------|
| Fare | Regression | Continuous USD prediction [1] |
| Duration | Regression | Continuous minutes prediction [1] |
| Tip | Classification | Binary logistic prediction with calibrated probabilities [1] |
| Demand | Count regression | Poisson regression on grouped time-location buckets [1] |

### Key Technical Decisions
- Native categorical handling in XGBoost instead of one-hot encoding. [1]
- Optuna TPE search for hyperparameter tuning. [1]
- Five-fold cross-validation for robust evaluation. [1]
- Group-aware validation for demand modeling to reduce leakage. [1]
- Fixed random seed for reproducibility across the full workflow. [1]

## Performance Snapshot

Reported metrics were evaluated on a 1.5 million-row stratified evaluation sample, with fold stability also tracked during cross-validation. [1]

| Model | Primary Metric | Reported Result |
|------|----------------|----------------|
| Fare | RMSE | 2.98 [1] |
| Duration | RMSE | 5.41 minutes [1] |
| Tip | Brier score / calibration quality | 0.159 Brier, 0.0019 ECE [1] |
| Demand | $$R^2$$ | 0.967 [1] |

Additional results in the capstone deck note that the fare model improved from a baseline RMSE of 3.82 to 2.98 after tuning and feature upgrades, a relative reduction of about 21.9%. [1]

## Application Demo

The project includes a Streamlit application that loads all four trained model artifacts and routes one typed 16-column feature row into each model. The interface exposes fare, duration, tip likelihood, and demand outputs, along with a best-time-to-leave sweep, weather override controls, and a debug panel for inspecting the live feature vector. [1]

## Repository Structure

```text
NYC-Taxi-Trip/
├── notebooks/      # DSCI 591 cleaning and DSCI 592 modeling notebooks
├── data/           # Cleaned parquet pointers and raw-source references
├── baselinefare/   # Generated metrics, CSV/JSON outputs, and figures
├── docs/           # Predictive modeling report, pitch deck, and notes
├── src/            # Planned reusable preprocessing and evaluation utilities
└── README.md       # Setup, overview, and run instructions
```


## Team Roles

**Caleb Solomons**  
Contributed to project framing, technical writing, repository organization, and model analysis. Background includes work across diverse datasets and experience spanning preprocessing, experimentation, and documentation. [1]

**Mubeen Yaqub**  
Contributed to ETL, feature engineering, modeling, and end-to-end pipeline implementation, with strong emphasis on Python-based data workflows and production-style data preparation. [1]

**Mai Lam**  
Contributed to dataset framing, feature contract design, and explanation of feature engineering and leakage-aware modeling decisions in the capstone presentation. [1]

**Anh Minh Tran**  
Contributed to EDA interpretation, modeling methodology presentation, and explanation of training strategy, demand modeling, and evaluation logic. [1]

## Challenges

- Working with a dataset large enough to require careful filtering, sampling, and memory-aware training strategy. [1]
- Designing one shared feature contract that works across four distinct prediction tasks. [1]
- Avoiding leakage by restricting features to booking-time information only. [1]
- Handling known quirks in TLC data, especially recorded tip behavior for cash trips. [1]
- Building reproducible training and evaluation pipelines with stable cross-validation results. [1]

## Next Steps

Planned next steps include upgrading borough-level inputs to full TLC zone IDs, adding lagged demand features, decomposing tip prediction into a two-stage process, supporting quantile predictions, refactoring shared feature engineering into a reusable module, and moving toward stronger deployment and MLOps practices such as FastAPI serving, model registry, drift monitoring, and CI-based contract checks. [1]
