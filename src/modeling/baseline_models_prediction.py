import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

# ============================================================
# CONFIG
# ============================================================

CITIES = ["pohang", "seoul"]

HORIZONS = {
    "d": 0,
    "d+1": 1,
    "d+3": 3,
    "d+7": 7,
}

MAX_HORIZON = max(HORIZONS.values())

TARGET_MAP = {
    ("pohang", "seoul"): [
        "rail_passengers_pohang_to_seoul",
        "express_bus_passengers_pohang_to_seoul",
        "intercity_bus_passengers_pohang_to_seoul",
    ],
    ("seoul", "pohang"): [
        "rail_passengers_seoul_to_pohang",
        "express_bus_passengers_seoul_to_pohang",
        "intercity_bus_passengers_seoul_to_pohang",
    ],
}

RANDOM_SEED = 26
TEST_SIZE = 0.15
N_SPLITS = 5


# ============================================================
# COLUMN DETECTION
# ============================================================

def infer_column_groups(df: pd.DataFrame, cities: list[str], target_map: dict):
    """
    Returns:
        city_cols: dict[city] -> list of columns tied to that city
        calendar_cols: columns that are neither city-specific nor targets
    """
    all_cols = list(df.columns)
    target_cols_flat = {c for cols in target_map.values() for c in cols}
    cols_lower = {c: c.lower() for c in all_cols}

    city_cols = {}
    for city in cities:
        other_cities = [c for c in cities if c != city]
        city_cols[city] = [
            c for c in all_cols
            if city in cols_lower[c]
            and not any(other in cols_lower[c] for other in other_cities)
        ]

    calendar_cols = [
        c for c in all_cols
        if c not in target_cols_flat
        and not any(city in cols_lower[c] for city in cities)
    ]

    return city_cols, calendar_cols


# ============================================================
# FAST WINDOW BUILDER
# ============================================================

def build_input_matrix_fast(df: pd.DataFrame, base_cols: list[str], horizon: int) -> pd.DataFrame:
    """
    Build the input matrix for one horizon.

    If horizon = 7, each feature c becomes:
        c_d0, c_d1, ..., c_d7

    IMPORTANT:
    - df must already be sorted by time.
    - no date column is needed.
    """
    if horizon < 0:
        raise ValueError("horizon must be >= 0")

    selected = df.loc[:, base_cols]
    arr = selected.to_numpy(dtype=np.float32, copy=False)
    n, m = arr.shape

    if horizon >= n:
        raise ValueError(f"horizon={horizon} is too large for n_rows={n}")

    out = np.full((n, m * (horizon + 1)), np.nan, dtype=np.float32)

    for k in range(horizon + 1):
        out[: n - k, k * m : (k + 1) * m] = arr[k:]

    columns = [
        f"{col}_d{k}"
        for k in range(horizon + 1)
        for col in base_cols
    ]

    return pd.DataFrame(out, columns=columns, index=df.index)


# ============================================================
# DATASET BUILDER
# ============================================================

def build_model_dataset(
    df: pd.DataFrame,
    origin: str,
    destination: str,
    horizon: int,
    feature_map: dict,
    target_map: dict,
    max_horizon: int,
    target_mode: str = "total",
) -> pd.DataFrame:
    """
    Build one dataset for one origin/destination pair and one horizon.

    target_mode:
        - "total": sum the 3 transport modes into one target
        - "modes" : keep the 3 target columns separately
    """
    origin = origin.lower()
    destination = destination.lower()

    if (origin, destination) not in target_map:
        raise ValueError(f"Unsupported route: {(origin, destination)}")

    if destination not in feature_map:
        raise ValueError(f"Unknown destination city: {destination}")

    feature_cols = feature_map[destination]
    target_cols = target_map[(origin, destination)]
    target_name = f"Transport_passengers_{origin}_to_{destination}"

    X = build_input_matrix_fast(df, feature_cols, horizon=horizon)

    if target_mode == "total":
        y = df.loc[:, target_cols].sum(axis=1).rename(target_name).to_frame()
    elif target_mode == "modes":
        y = df.loc[:, target_cols].copy()
    else:
        raise ValueError("target_mode must be 'total' or 'modes'")

    data = pd.concat([X, y], axis=1)

    # Keep a common usable prefix across all horizons for fair comparison
    valid_n = len(df) - max_horizon
    data = data.iloc[:valid_n].reset_index(drop=True)

    return data


def get_xy_holdout(
    data: pd.DataFrame,
    target_col: str,
    test_size: float = TEST_SIZE,
):
    """
    Split a time-ordered dataset into train_val and final test sets.
    """
    X = data.drop(columns=[target_col]).copy()
    y = data[target_col].copy()

    n = len(data)
    split_idx = int(n * (1 - test_size))

    X_train_val = X.iloc[:split_idx].copy()
    y_train_val = y.iloc[:split_idx].copy()
    X_final_test = X.iloc[split_idx:].copy()
    y_final_test = y.iloc[split_idx:].copy()

    return X_train_val, y_train_val, X_final_test, y_final_test


# ============================================================
# MODEL FACTORIES
# ============================================================

def make_dummy_model():
    return DummyRegressor(strategy="mean")


def make_linear_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def make_ridge_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    ])


def make_xgb_model():
    return XGBRegressor(
        n_estimators=4000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
        max_bin=256,
        verbosity=0,
        eval_metric="rmse",
    )


# ============================================================
# TIME SERIES CV
# ============================================================

def make_time_series_cv_results_sklearn(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    n_splits: int = N_SPLITS,
    gap: int = 0,
):
    """
    TimeSeriesSplit for sklearn-compatible models.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        fold_results.append({
            "fold": fold,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "mae": mean_absolute_error(y_val, pred),
            "rmse": root_mean_squared_error(y_val, pred),
            "r2": r2_score(y_val, pred),
        })

    return pd.DataFrame(fold_results)


def make_time_series_cv_results_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    n_splits: int = N_SPLITS,
    gap: int = 0,
):
    """
    TimeSeriesSplit for XGBoost models.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_factory()
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        pred = model.predict(X_val)

        fold_results.append({
            "fold": fold,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "mae": mean_absolute_error(y_val, pred),
            "rmse": root_mean_squared_error(y_val, pred),
            "r2": r2_score(y_val, pred),
        })

    return pd.DataFrame(fold_results)


# ============================================================
# EVALUATION HELPERS
# ============================================================

def fit_and_score_final_test(model_factory, X_train_val, y_train_val, X_final_test, y_final_test):
    """
    Fit one model on the train/validation part and evaluate it on the final test set.
    """
    model = model_factory()
    model.fit(X_train_val, y_train_val)
    pred_final_test = model.predict(X_final_test)

    return model, pred_final_test, {
        "mae_test": mean_absolute_error(y_final_test, pred_final_test),
        "rmse_test": root_mean_squared_error(y_final_test, pred_final_test),
        "r2_test": r2_score(y_final_test, pred_final_test),
    }


def evaluate_one_case(df, origin, destination, horizon, feature_map):
    """
    Evaluate Dummy / Linear / Ridge / XGBoost on one origin-destination-horizon case.

    Metrics are computed:
    - by TimeSeriesSplit on the train_val subset
    - on the final holdout test subset
    """
    data = build_model_dataset(
        df=df,
        origin=origin,
        destination=destination,
        horizon=horizon,
        feature_map=feature_map,
        target_map=TARGET_MAP,
        max_horizon=MAX_HORIZON,
        target_mode="total",
    )

    target_col = f"Transport_passengers_{origin}_to_{destination}"
    X_train_val, y_train_val, X_final_test, y_final_test = get_xy_holdout(data, target_col)

    model_specs = [
        ("Dummy", make_dummy_model, "sklearn"),
        ("LinearRegression", make_linear_model, "sklearn"),
        ("Ridge", make_ridge_model, "sklearn"),
        ("XGBoost", make_xgb_model, "xgb"),
    ]

    rows = []

    for model_name, factory, model_type in model_specs:
        if model_type == "xgb":
            _ = make_time_series_cv_results_xgb(
                X=X_train_val,
                y=y_train_val,
                model_factory=factory,
                n_splits=N_SPLITS,
                gap=horizon,
            )
        else:
            _ = make_time_series_cv_results_sklearn(
                X=X_train_val,
                y=y_train_val,
                model_factory=factory,
                n_splits=N_SPLITS,
                gap=horizon,
            )

        _, _, test_metrics = fit_and_score_final_test(
            model_factory=factory,
            X_train_val=X_train_val,
            y_train_val=y_train_val,
            X_final_test=X_final_test,
            y_final_test=y_final_test,
        )

        rows.append({
            "origin": origin,
            "destination": destination,
            "horizon_days": horizon,
            "model": model_name,
            "mae_test": test_metrics["mae_test"],
            "rmse_test": test_metrics["rmse_test"],
            "r2_test": test_metrics["r2_test"],
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def baseline_models_prediction():
    df = pd.read_parquet("data/processed/aggregated_dataset.parquet")
    df = df.reset_index(drop=True)  # keep chronological order

    city_cols, calendar_cols = infer_column_groups(df, CITIES, TARGET_MAP)

    feature_map = {
        "seoul": calendar_cols + city_cols["seoul"],
        "pohang": calendar_cols + city_cols["pohang"],
    }

    all_results = []

    for (origin, destination), _ in TARGET_MAP.items():
        for horizon_name, horizon in HORIZONS.items():
            print(
                f"\n{'='*80}\n"
                f"Origin: {origin} | Destination: {destination} | Horizon: {horizon_name}\n"
                f"{'='*80}"
            )

            df_case = evaluate_one_case(
                df=df,
                origin=origin,
                destination=destination,
                horizon=horizon,
                feature_map=feature_map,
            )

            print(df_case)

            all_results.append(df_case)

    summary_df = pd.concat(all_results, ignore_index=True).sort_values(
        ["origin", "destination", "model", "horizon_days"]
    ).reset_index(drop=True)

    print("\nFINAL SUMMARY")
    print(summary_df)

    summary_df.to_csv("results/model_comparison_baselines.csv", index=False)


if __name__ == "__main__":
    Path("results").mkdir(parents=True, exist_ok=True)
    baseline_models_prediction()
