# GridRunners: NYC Yellow Taxi Predictive Suite

**DSCI 592 Capstone Project**  
Drexel University · June 2026

GridRunners is an end-to-end machine learning project built on the 2023 NYC Yellow Taxi dataset. The project predicts fare, trip duration, tip likelihood, and demand through a shared feature pipeline and a unified application interface.

## Team Members
- Caleb Solomons
- Mubeen Yaqub
- Mai Lam
- Anh Minh Tran

## Project Overview

This capstone project extends beyond exploratory analysis into a full predictive analytics suite for urban mobility. Using New York City Yellow Taxi trip data, the project develops four coordinated machine learning models that operate from one common feature contract and support booking-time prediction.

The full workflow covers data cleaning, feature engineering, exploratory analysis, model development, evaluation, and deployment through a Streamlit-based interface. The result is a single system that can estimate:

- Fare
- Trip duration
- Tip likelihood
- Demand

## Project Goals

- Build four prediction tasks on top of one consistent feature schema.
- Use only booking-time features to avoid target leakage.
- Create an end-to-end pipeline from cleaned parquet data to deployed predictions.
- Support reproducibility through shared preprocessing logic and stable evaluation.
- Deliver all model outputs through one interactive application.

## Dataset

The project uses the **2023 NYC TLC Yellow Taxi** public dataset.

### Data Summary
- Raw data size: approximately 38.3 million trips
- Raw columns: 19
- Raw storage footprint: about 14 GB
- Cleaned dataset: approximately 37.0 million usable trips
- Modeling table width after engineering: 35 columns
- Evaluation sample: 1.5 million stratified rows
- Hyperparameter tuning sample: 250,000 rows

### Data Processing Highlights
- Removed invalid and implausible trips through a multi-stage cleaning funnel.
- Applied sanity bounds to fare, duration, and distance.
- Preserved a modeling-ready parquet dataset for downstream workflows.
- Used stratified sampling for large-scale but efficient evaluation.

## Exploratory Analysis

Exploratory analysis shaped the modeling decisions in several important ways.

### Temporal Patterns
- Hour of day influences fare, tip behavior, and demand.
- Demand follows a clear rush-hour bimodal pattern.
- Time-based behavior supported the inclusion of categorical pickup-time features.

### Payment Behavior
- Cash trips often appear with zero recorded tip in TLC data.
- This affects interpretation of the observed tip target.
- The tip model therefore focuses on recorded tip likelihood rather than assumed rider generosity.

### Spatial Structure
- Borough geography strongly affects trip distance and demand concentration.
- Manhattan and airport-linked trips contribute heavily to the fare tail.
- Spatial proxy features were engineered to improve route-aware learning.

## Feature Engineering

All models share a **16-column feature contract** so one trip row can be passed into every prediction head consistently.

### Numeric Features
- `trip_distance`
- `haversine_mi`
- `manhattan_mi`
- `passenger_count`
- `temp_f`
- `precip_in`

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

### Engineering Pillars
- Event-day flags for holidays and major NYC events
- Distance proxies using haversine and Manhattan-style approximations
- Daily weather joins for temperature and precipitation context
- Native categorical handling in XGBoost instead of one-hot encoding

## Leakage Control

A core project rule is that a feature is valid only if it would be known at booking time.

The pipeline excludes post-trip or receipt-derived fields such as:
- `tip_amount`
- `tolls_amount`
- `total_amount`
- `congestion_surcharge`
- `airport_fee`
- `trip_duration_min`

The tip model also excludes `payment_type`, since payment choice is not known before the ride starts.

## Modeling Approach

The pipeline is shared across tasks until the final prediction stage. Each target then uses a modeling setup appropriate to the prediction type.

| Target | Task Type | Modeling Framing |
|--------|-----------|------------------|
| Fare | Regression | Continuous USD prediction |
| Duration | Regression | Continuous minutes prediction |
| Tip | Classification | Binary prediction with calibrated probabilities |
| Demand | Count regression | Poisson modeling on grouped borough-hour-date buckets |

### Workflow
1. Load cleaned parquet data.
2. Apply task-specific sanity filtering.
3. Build engineered booking-time features.
4. Create evaluation and tuning samples.
5. Tune with Optuna.
6. Validate with cross-validation.
7. Refit final models and serve outputs in the demo application.

### Technical Decisions
- XGBoost with native categorical support
- Optuna TPE-based hyperparameter tuning
- Five-fold cross-validation for stable estimates
- Group-aware validation for demand modeling
- Fixed seed for reproducibility across the workflow

## Performance

The project reports mean results on a 1.5 million-row stratified evaluation sample.

| Model | Primary Result |
|-------|----------------|
| Fare | RMSE: 2.98 |
| Duration | RMSE: 5.41 minutes |
| Tip | Brier score: 0.159 |
| Demand | R²: 0.967 |

### Additional Notes
- Fare model performance improved from a baseline RMSE of 3.82 to 2.98.
- Cross-validation variance remained very small, supporting reproducibility.
- Tip calibration was especially strong, with ECE reported at 0.0019.
- Group-aware cross-validation was essential for honest demand evaluation.

## Demo Application

The project includes a Streamlit app that connects all four trained models through one shared feature row.

### App Capabilities
- Predict fare, duration, tip likelihood, and demand in one interface
- Generate a best-time-to-leave sweep over a three-hour horizon
- Support weather override for what-if analysis
- Expose the live feature vector through a debug panel

This design keeps the prediction workflow unified and makes the feature contract easy to inspect end to end.

## Repository Structure

```text
NYC-Taxi-Trip/
├── data/                   # Data assets and schema-related files
├── docs/                   # Project documents and supporting materials
├── notebook/               # EDA, modeling, and capstone notebooks
├── .gitignore              # Git ignore rules
├── GridRunners_Report.pdf  # Final project report
├── README.md               # Project overview and usage guide
└── requirements.txt        # Python dependencies
```

## Challenges

- Managing a very large dataset efficiently during experimentation
- Designing one shared schema for four separate prediction tasks
- Preventing target leakage across training and inference
- Interpreting known quirks in recorded TLC tip behavior
- Maintaining reproducible evaluation across tuning and retraining

## Next Steps

Planned improvements include:

- Upgrade borough inputs to full TLC zone IDs
- Add lagged demand features for stronger temporal signal
- Decompose tip prediction into payment and conditional-tip stages
- Add quantile outputs for uncertainty-aware decisions
- Refactor shared features into reusable modules
- Move toward FastAPI deployment and stronger MLOps support
- Add drift monitoring, versioned parameters, and CI checks
- Explore finer demand buckets and multi-year backfills

## Tech Stack

- Python
- Pandas
- XGBoost
- Optuna
- Streamlit
- Parquet-based data workflow
- Jupyter notebooks

## Report

The repository also includes the final written project report:

- `GridRunners_Report.pdf`

## License

This project was developed as an academic capstone project for learning, experimentation, and portfolio demonstration.
