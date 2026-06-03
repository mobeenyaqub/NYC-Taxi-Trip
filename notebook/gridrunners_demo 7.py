"""
GridRunners — NYC Taxi Predictor (Streamlit demo)
=================================================
DSCI 592 Capstone — Caleb Solomons, Mubeen Yaqub, Mai Lam, Anh Minh Tran

Single-file Streamlit app. Loads the v2 XGBoost fare model and runs
end-to-end fare prediction from consumer-facing inputs (zones, time,
passengers). Stubs for duration / tip / demand models so the UI works
the moment any of them are saved into ./models/.

Run:
    pip install streamlit xgboost pandas numpy requests
    streamlit run gridrunners_demo.py

Expected directory layout:
    ./gridrunners_demo.py
    ./models/xgb_v2_fare.json          ← from notebook Section 14
    ./models/xgb_v2_duration.json      ← when ready
    ./models/xgb_v2_tip.json           ← when ready
    ./models/xgb_v2_demand.json        ← when ready
    ./data/taxi_zone_lookup.csv        ← TLC zone lookup (LocationID, Borough, Zone, service_zone)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb
from pandas.tseries.holiday import USFederalHolidayCalendar


# ============================================================================
# CONFIG — paths, constants, feature schema
# ============================================================================

#MODEL_DIR = Path("./models")
#ZONE_LOOKUP_PATH = Path("./data/taxi_zone_lookup.csv")

DEMO_DIR = Path(r"C:\Users\Hush5\OneDrive\Desktop\592\demo_artifacts")

# or wherever you actually put them

MODEL_DIR = DEMO_DIR / "models"
ZONE_LOOKUP_PATH = DEMO_DIR / "taxi_zone_lookup.csv"

print(f"DEMO_DIR exists: {DEMO_DIR.exists()}")
print(f"Contents: {list(DEMO_DIR.iterdir())}")
print(f"Model path: {MODEL_DIR / 'xgb_v2_fare.json'}")
print(f"Model path exists: {(MODEL_DIR / 'xgb_v2_fare.json').exists()}")

# Borough centroids in (lat, lon). MUST match the dict used during training
# in the notebook — same keys, same values, same ordering of lookups.

BOROUGH_CENTROIDS = {
    "Manhattan":     (40.7831, -73.9712),
    "Brooklyn":      (40.6782, -73.9442),
    "Queens":        (40.7282, -73.7949),
    "Bronx":         (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
    "EWR":           (40.6895, -74.1745),
}

# TLC LocationIDs for airport zones — used to auto-infer RatecodeID

JFK_ZONE_IDS = {132}
LGA_ZONE_IDS = {138}
EWR_ZONE_IDS = {1}

# NYC-specific event days. Update yearly with real dates.

NYC_EVENTS = pd.to_datetime([
    "2026-06-28",  # NYC Pride March (estimate)
    "2026-09-13",  # US Open men's final (estimate)
    "2026-11-01",  # NYC Marathon (estimate)
    "2026-11-26",  # Macy's Thanksgiving Day Parade
    "2026-12-31",  # NYE Times Square
]).normalize()

# Empirical road-distance multiplier on Manhattan-L1 distance.
# Calibrated from NYC trip data: actual driven distance ~ 1.3× manhattan_mi

MANHATTAN_TO_ROUTE_MULTIPLIER = 1.30

# Earth radius in miles (haversine constant)
EARTH_R_MI = 3958.8

# Sanity bound from training — predictions below this are clamped up

FARE_FLOOR = 3.0

# Feature schema. ORDER MATTERS — must match what XGBoost saw at training time.

NUMERIC_FEATURES = [
    "trip_distance", "passenger_count", "haversine_mi", "manhattan_mi",
    "temp_f", "precip_in",
]
CATEGORICAL_FEATURES = [
    "pickup_hour", "pickup_dow", "is_weekend",
    "PU_borough", "DO_borough", "VendorID", "RatecodeID",
    "is_us_holiday", "is_nyc_event", "is_rainy",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Integer-backed categorical dtypes — must exactly match the notebook

INT_CAT_DTYPES = {
    "pickup_hour":   "int32",
    "pickup_dow":    "int32",
    "is_weekend":    "int64",
    "VendorID":      "int64",
    "RatecodeID":    "int8",
    "is_us_holiday": "int8",
    "is_nyc_event":  "int8",
    "is_rainy":      "int8",
}



# ============================================================================
# CACHED RESOURCE LOADERS
# ============================================================================

@st.cache_resource
def load_fare_model() -> xgb.Booster | None:

    """v2 tuned XGBoost fare model. None if not yet saved."""

    path = MODEL_DIR / "xgb_v2_fare.json"

    if not path.exists():
        return None
    
    booster = xgb.Booster()
    booster.load_model(str(path))

    return booster


@st.cache_resource
def load_duration_model() -> xgb.Booster | None:

    """Trip duration model — wire in when the team saves duration_v2."""

    path = MODEL_DIR / "xgb_v2_duration.json"

    if not path.exists():
        return None
    
    booster = xgb.Booster()
    booster.load_model(str(path))

    return booster


@st.cache_resource
def load_tip_model() -> xgb.Booster | None:

    """Tip classification model — binary probability of tipping."""

    path = MODEL_DIR / "xgb_v2_tip.json"

    if not path.exists():
        return None
    
    booster = xgb.Booster()
    booster.load_model(str(path))

    return booster


@st.cache_resource
def load_demand_model() -> xgb.Booster | None:

    """Demand forecasting model — pickups per zone/hour."""

    path = MODEL_DIR / "xgb_v2_demand.json"

    if not path.exists():
        return None
    
    booster = xgb.Booster()
    booster.load_model(str(path))

    return booster


@st.cache_data
def load_zone_lookup() -> pd.DataFrame:

    """TLC taxi zone lookup. Falls back to a minimal hardcoded set."""

    if ZONE_LOOKUP_PATH.exists():
        return pd.read_csv(ZONE_LOOKUP_PATH)
    
    # Fallback — just enough zones to demo each borough + airports

    return pd.DataFrame({
        "LocationID":   [1, 132, 138, 161, 230, 79, 33, 75, 244],
        "Borough":      ["EWR", "Queens", "Queens", "Manhattan",
                         "Manhattan", "Manhattan", "Brooklyn",
                         "Bronx", "Staten Island"],
        "Zone":         ["Newark Airport", "JFK Airport", "LaGuardia Airport",
                         "Midtown Center", "Times Sq/Theatre District",
                         "East Village", "Brooklyn Heights",
                         "East Tremont", "Stapleton"],
        "service_zone": ["EWR", "Airports", "Airports", "Yellow Zone",
                         "Yellow Zone", "Yellow Zone", "Boro Zone",
                         "Boro Zone", "Boro Zone"],
    })


@st.cache_data(ttl=3600)  # cache for 1 hour
def fetch_weather(target_date: date) -> tuple[float, float]:

    """
    Open-Meteo lookup. Picks the archive endpoint for past dates and the
    forecast endpoint for today/future. Returns (temp_f, precip_in).
    Falls back to monthly climatology on any error.
    """
    today = date.today()
    iso = target_date.isoformat()

    base = (
        "https://archive-api.open-meteo.com/v1/archive"
        if target_date < today
        else "https://api.open-meteo.com/v1/forecast"
    )

    url = (
        f"{base}?latitude=40.7831&longitude=-73.9712"
        f"&start_date={iso}&end_date={iso}"
        f"&daily=temperature_2m_mean,precipitation_sum"
        f"&temperature_unit=fahrenheit&precipitation_unit=inch"
        f"&timezone=America/New_York"
    )

    try:

        r = requests.get(url, timeout=10)
        r.raise_for_status()

        j = r.json()["daily"]

        return float(j["temperature_2m_mean"][0]), float(j["precipitation_sum"][0])
    
    except Exception:
        return _climatology_fallback(target_date)


def _climatology_fallback(target_date: date) -> tuple[float, float]:

    """Coarse NYC monthly-mean temperatures. Used only when the API fails."""

    monthly_temp = [32, 35, 43, 53, 63, 72, 78, 76, 69, 58, 47, 37]

    return float(monthly_temp[target_date.month - 1]), 0.0


# ============================================================================
# FEATURE DERIVATION — must mirror the training pipeline exactly
# ============================================================================

def haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:

    """Great-circle (bird-path) distance in miles."""

    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    return float(2 * EARTH_R_MI * np.arcsin(np.sqrt(a)))


def manhattan_l1_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:

    """L1 (taxicab) distance in miles, calibrated at ~40.7°N (NYC)."""

    return abs(lat1 - lat2) * 69.0 + abs(lon1 - lon2) * 52.5


def infer_ratecode(pu_zone_id: int, do_zone_id: int) -> int:

    """
    RatecodeID inference at booking time:
      1 = standard meter (default)
      2 = JFK flat fare
      3 = Newark
    LGA stays at 1 — it's metered, not flat.
    """

    if pu_zone_id in JFK_ZONE_IDS or do_zone_id in JFK_ZONE_IDS:
        return 2
    
    if pu_zone_id in EWR_ZONE_IDS or do_zone_id in EWR_ZONE_IDS:
        return 3
    
    return 1


def build_feature_row(
    pu_zone_id: int,
    do_zone_id: int,
    departure_dt: datetime,
    passenger_count: int,
    zone_lookup: pd.DataFrame,
    weather_override: tuple[float, float] | None = None,
) -> pd.DataFrame:
    
    """
    Build a single-row DataFrame matching the model's training schema.

    Feature alignment failures here are the #1 source of silent demo bugs.
    Column order, dtypes, and categorical encoding all have to match.
    """

    # zone -> boro

    pu_borough = zone_lookup.loc[zone_lookup["LocationID"] == pu_zone_id, "Borough"].iloc[0]
    do_borough = zone_lookup.loc[zone_lookup["LocationID"] == do_zone_id, "Borough"].iloc[0]

    # distance proxies

    pu_lat, pu_lon = BOROUGH_CENTROIDS.get(pu_borough, (np.nan, np.nan))
    do_lat, do_lon = BOROUGH_CENTROIDS.get(do_borough, (np.nan, np.nan))

    hav = haversine_mi(pu_lat, pu_lon, do_lat, do_lon)
    manh = manhattan_l1_mi(pu_lat, pu_lon, do_lat, do_lon)

    # Trip distance: routing-API estimate proxy

    trip_dist = max(manh * MANHATTAN_TO_ROUTE_MULTIPLIER, 0.3)

    #datetime features

    pickup_hour = departure_dt.hour
    pickup_dow = departure_dt.weekday()  # Mon=0 ... Sun=6
    is_weekend = int(pickup_dow >= 5)    # 5=Sat, 6=Sun -> 1, else 0

    # holiday / event flags

    pickup_date = pd.Timestamp(departure_dt.date()).normalize()
    yr = departure_dt.year

    holidays = pd.DatetimeIndex(
        USFederalHolidayCalendar().holidays(start=f"{yr}-01-01", end=f"{yr}-12-31")
    ).normalize()

    is_us_holiday = int(pickup_date in holidays)
    is_nyc_event = int(pickup_date in NYC_EVENTS)

    # weather
    if weather_override is not None:

        temp_f, precip_in = weather_override

    else:
        temp_f, precip_in = fetch_weather(departure_dt.date())

    is_rainy = int(precip_in > 0.1)

    # ratecode + vendor

    ratecode = infer_ratecode(pu_zone_id, do_zone_id)
    vendor_id = 2  # modal vendor in 2023 TLC data

    # assemble row

    row = pd.DataFrame([{
        "trip_distance":   trip_dist,
        "passenger_count": float(passenger_count),
        "haversine_mi":    hav,
        "manhattan_mi":    manh,
        "temp_f":          temp_f,
        "precip_in":       precip_in,
        "pickup_hour":     pickup_hour,
        "pickup_dow":      pickup_dow,
        "is_weekend":      is_weekend,
        "PU_borough":      pu_borough,
        "DO_borough":      do_borough,
        "VendorID":        vendor_id,
        "RatecodeID":      ratecode,
        "is_us_holiday":   is_us_holiday,
        "is_nyc_event":    is_nyc_event,
        "is_rainy":        is_rainy,
    }])

    # dtype + categorical alignmen
    # Integer-backed categoricals: cast int first, then to category, so the
    # underlying numpy dtype matches what XGBoost saw during training.

    for col, dt in INT_CAT_DTYPES.items():
        row[col] = row[col].astype(dt).astype("category")

    # String categoricals (boroughs)
    for col in ("PU_borough", "DO_borough"):
        row[col] = row[col].astype("category")

    return row[ALL_FEATURES]


# ============================================================================
# PREDICTION WRAPPERS
# ============================================================================

def _to_dmatrix(features: pd.DataFrame) -> xgb.DMatrix:

    """Booster predictions need DMatrix with enable_categorical=True."""

    return xgb.DMatrix(features, enable_categorical=True)


def predict_fare(features: pd.DataFrame, model: xgb.Booster | None) -> float | None:

    if model is None:
        return None
    
    pred = float(model.predict(_to_dmatrix(features))[0])

    return max(pred, FARE_FLOOR) 


def predict_duration(features: pd.DataFrame, model: xgb.Booster | None) -> float | None:

    if model is None:
        return None
    
    return max(float(model.predict(_to_dmatrix(features))[0]), 1.0)


def predict_tip_probability(features: pd.DataFrame, model: xgb.Booster | None) -> float | None:

    if model is None:
        return None
    
    # Assumes binary classifier with probability output

    return float(np.clip(model.predict(_to_dmatrix(features))[0], 0.0, 1.0))


def predict_demand(features: pd.DataFrame, model: xgb.Booster | None) -> float | None:

    if model is None:
        return None
    
    return float(model.predict(_to_dmatrix(features))[0])


# ============================================================================
# TIME-SWEEP OPTIMIZATION ("Best time to leave in next N hours")
# ============================================================================

def sweep_departure_times(
    pu_zone_id: int,
    do_zone_id: int,
    base_dt: datetime,
    passenger_count: int,
    zone_lookup: pd.DataFrame,
    fare_model: xgb.Booster | None,
    duration_model: xgb.Booster | None = None,
    window_hours: int = 3,
    step_minutes: int = 15,
) -> pd.DataFrame:
    
    """
    Predict fare (and optionally duration) over a sweep of candidate
    departure times. This is the consumer-facing "optimization" view —
    user can see which moment in the next few hours is cheapest.
    """

    n_steps = int((window_hours * 60) // step_minutes) + 1
    records = []

    for i in range(n_steps):

        dt = base_dt + timedelta(minutes=i * step_minutes)

        feats = build_feature_row(
            pu_zone_id, do_zone_id, dt, passenger_count, zone_lookup
        )

        records.append({
            "departure_time": dt,
            "fare": predict_fare(feats, fare_model),
            "duration": predict_duration(feats, duration_model),
        })

    return pd.DataFrame(records)


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main() -> None:

    st.set_page_config(
        page_title="GridRunners — NYC Taxi Predictor",
        page_icon="🚕",
        layout="wide",
    )

    st.title("GridRunners")

    st.caption(
        "NYC Taxi fare, duration, tip & demand predictor — "
        "DSCI 592 Capstone (Solomons, Yaqub, Lam, Anh)"
    )


    fare_model = load_fare_model()
    duration_model = load_duration_model()
    tip_model = load_tip_model()
    demand_model = load_demand_model()
    zone_lookup = load_zone_lookup()

    if fare_model is None:

        st.warning(
            "Fare model not found at `./models/xgb_v2_fare.json`. "
            "Export it from the notebook with `model_final.save_model(...)`."
        )

    # sidebar inputs

    with st.sidebar:

        st.header("Trip details")

        zones_sorted = zone_lookup.sort_values("Zone").reset_index(drop=True)

        zone_labels = [
            f"{r['Zone']} ({r['Borough']})"
            for _, r in zones_sorted.iterrows()
        ]

        zone_ids = zones_sorted["LocationID"].tolist()

        pu_idx = st.selectbox(
            "Pickup zone",
            options=range(len(zone_labels)),
            format_func=lambda i: zone_labels[i],
            index=0,
        )

        do_idx = st.selectbox(
            "Dropoff zone",
            options=range(len(zone_labels)),
            format_func=lambda i: zone_labels[i],
            index=min(1, len(zone_labels) - 1),
        )

        pu_zone_id = zone_ids[pu_idx]
        do_zone_id = zone_ids[do_idx]

        c1, c2 = st.columns(2)

        with c1:
            dep_date = st.date_input("Date", value=date.today())

        with c2:
            dep_time = st.time_input("Time", value=datetime.now().time())

        departure_dt = datetime.combine(dep_date, dep_time)

        passenger_count = st.slider("Passengers", 1, 6, 1)

        with st.expander("Advanced — weather override"):

            override = st.checkbox("Manually set weather")

            if override:

                t = st.slider("Temperature (°F)", 0, 110, 65)
                p = st.slider("Precipitation (in)", 0.0, 4.0, 0.0, step=0.1)

                weather_override = (float(t), float(p))

            else:
                weather_override = None

    #build feature row once for the displayed prediction

    feats = build_feature_row(pu_zone_id, do_zone_id, departure_dt, passenger_count,zone_lookup, weather_override,)

    fare = predict_fare(feats, fare_model)
    duration = predict_duration(feats, duration_model)
    tip_prob = predict_tip_probability(feats, tip_model)
    demand = predict_demand(feats, demand_model)
    
    #For DEBUG and Demo purposes only:
    st.write({"fare": fare, "duration": duration, "tip_prob": tip_prob, "demand": demand})

    # Top-line metrics

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "Predicted fare",
            f"${fare:.2f}" if fare is not None else "—",
            help="XGBoost v2 — RMSE $2.98 on 5-fold CV",
        )

    with m2:

        st.metric(
            "Trip duration",
            f"{duration:.0f} min" if duration is not None else "—",
            help="Awaiting duration model",
        )

    with m3:

        st.metric(
            "Tip likelihood",
            f"{tip_prob:.0%}" if tip_prob is not None else "—",
            help="Awaiting tip classification model",
        )

    with m4:

        st.metric(
            "Demand level",
            f"{demand:.0f}" if demand is not None else "—",
            help="Awaiting demand forecasting model",
        )

    # time-sweep optimization view

    st.subheader("Best time to leave (next 3 hours)")

    if fare_model is not None:

        sweep = sweep_departure_times(
            pu_zone_id, do_zone_id, departure_dt, passenger_count,
            zone_lookup, fare_model, duration_model,
            window_hours=3, step_minutes=15,
        )

        
        chart_df = sweep.set_index("departure_time")[["fare"]]

        st.line_chart(chart_df, height=240)

        
        cheapest = sweep.loc[sweep["fare"].idxmin()]
        delta = cheapest["fare"] - fare if fare is not None else 0.0

        st.info(
            f"**Cheapest departure: {cheapest['departure_time'].strftime('%I:%M %p')} "
            f"at ${cheapest['fare']:.2f}** "
            f"({'-' if delta < 0 else '+'}${abs(delta):.2f} vs. selected time)"
        )

        with st.expander("Show sweep table"):

            display = sweep.copy()
            display["departure_time"] = display["departure_time"].dt.strftime("%I:%M %p")
            display["fare"] = display["fare"].apply(lambda x: f"${x:.2f}")

            st.dataframe(display, use_container_width=True, hide_index=True)

    else:
        st.info("Time sweep unavailable until the fare model is loaded.")


    with st.expander("Feature vector being sent to model"):

        st.dataframe(feats.T.rename(columns={0: "value"}),use_container_width=True,)

    st.divider()

    st.caption(
        "Distances estimated from borough centroids × 1.30 road multiplier. "
        "Swap `MANHATTAN_TO_ROUTE_MULTIPLIER` for OSRM call in production."
    )


if __name__ == "__main__":
    main()
