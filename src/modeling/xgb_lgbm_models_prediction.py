
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import lightgbm as lgb
from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

import shap
import optuna

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

RANDOM_SEED = 26
TOP_K_FEATURES = 20
N_TRIALS = 50
N_SPLITS = 5

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

# Separate tuning spaces for each model family
XGB_PARAMETER_SPACE = {
    "max_depth": (3, 10),
    "min_child_weight": (1, 10),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "learning_rate": (0.005, 0.12),
    "reg_alpha": (1e-8, 10.0),
    "reg_lambda": (1e-3, 25.0),
    "gamma": (0.0, 5.0),
    "n_estimators": (500, 5000),
}

LGBM_PARAMETER_SPACE = {
    "num_leaves": (16, 256),
    "max_depth": (3, 12),
    "min_child_samples": (5, 100),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "learning_rate": (0.005, 0.12),
    "reg_alpha": (1e-8, 10.0),
    "reg_lambda": (1e-8, 25.0),
    "n_estimators": (500, 5000),
}


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
    Build features for days d0..dH using a single NumPy array.
    Output columns are ordered by day:
        col1_d0, col2_d0, ..., colN_d0, col1_d1, ..., colN_dH
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
        # rows 0..n-k-1 get values from rows k..n-1
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
    max_horizon: int,
    feature_map: dict,
    target_map: dict,
    target_mode: str = "total",
) -> pd.DataFrame:
    """
    Builds one dataset for one origin/destination and one horizon.

    target_mode:
        - "total": sum the 3 transport modes into one target
        - "modes": keep the 3 target columns separately
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


# ============================================================
# X / Y EXTRACTION
# ============================================================

def get_xy_holdout(
    data: pd.DataFrame,
    target_col: str,
    test_size: float = 0.15,
):
    X = data.drop(columns=[target_col])
    y = data[target_col].copy()

    n = len(data)
    split_idx = int(n * (1 - test_size))

    X_train_val = X.iloc[:split_idx].copy()
    y_train_val = y.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train_val, y_train_val, X_test, y_test


# ============================================================
# XGBOOST MODEL
# ============================================================

def make_xgb_model(params=None, early_stopping=0):
    base_params = dict(
        n_estimators=4000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
        max_bin=256,
        verbosity=0,
        eval_metric="rmse",
    )

    if early_stopping > 0:
        base_params["early_stopping_rounds"] = 50

    if params is not None:
        base_params.update(params)

    return XGBRegressor(**base_params)

# ============================================================
# LIGHTGBM MODEL
# ============================================================

def make_lgb_model(params=None):
    base_params = dict(
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="regression",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    if params is not None:
        base_params.update(params)

    return LGBMRegressor(**base_params)


# ============================================================
# SHAP
# ============================================================

def compute_shap_importance(model, X_val: pd.DataFrame) -> pd.Series:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # Depending on SHAP / model version, this may return a list
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    importance = np.abs(shap_values).mean(axis=0)
    
    return pd.Series(importance, index=X_val.columns).sort_values(ascending=False)


def save_shap_importance(
    shap_df: pd.DataFrame,
    origin: str,
    destination: str,
    horizon_name: str,
    output_dir: str,
):
    
    filename = (
        f"{origin}_to_{destination}"
        f"_{horizon_name}"
        "_shap.csv"
    )

    path = f"{output_dir}{filename}"

    (
        shap_df
        .reset_index(names="feature")
        .rename(columns={"mean_abs_shap": "importance"})
        .to_csv(path, index=False)
    )

    print(f"Saved SHAP importance to: {path}")


def compute_final_shap_analysis(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    rows = []

    for i, col in enumerate(X.columns):

        x = X[col].values
        s = shap_values[:, i]

        x_std = np.std(x)
        shap_std = np.std(s)

        if x_std < 1e-12:
            corr = np.nan
            status = "constant_feature"

        elif shap_std < 1e-12:
            corr = np.nan
            status = "ignored_by_model"

        else:
            corr = np.corrcoef(x, s)[0, 1]
            status = "valid"

        rows.append({
            "feature": col,
            "mean_abs_shap": np.abs(s).mean(),
            "feature_shap_corr": corr,
            "corr_status": status,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("mean_abs_shap", ascending=False)
    )


# ============================================================
# TIME SERIES CV
# ============================================================

def make_time_series_cv_results(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    n_splits: int = N_SPLITS,
    gap: int = 0,
    compute_shap: bool = True,
    shap_sample_size: int | None = 1000,
):
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    fold_results = []
    models = []
    shap_importances = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_factory()

        if isinstance(model, XGBRegressor):
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            pred = model.predict(X_val)

        elif isinstance(model, LGBMRegressor):
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )

            best_iter = getattr(model, "best_iteration_", None)
            if best_iter is not None:
                pred = model.predict(X_val, num_iteration=best_iter)
            else:
                pred = model.predict(X_val)

        else:
            raise ValueError("The model factory is not supported.")

        fold_results.append({
            "fold": fold,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "mae": mean_absolute_error(y_val, pred),
            "rmse": root_mean_squared_error(y_val, pred),
            "r2": r2_score(y_val, pred),
        })
        models.append(model)

        if compute_shap:
            X_shap = X_val
            if shap_sample_size is not None and len(X_shap) > shap_sample_size:
                X_shap = X_shap.sample(shap_sample_size, random_state=RANDOM_SEED)

            shap_fold_imp = compute_shap_importance(model, X_shap)
            shap_importances.append(shap_fold_imp)

    cv_df = pd.DataFrame(fold_results)

    shap_df = None
    if compute_shap and len(shap_importances) > 0:
        shap_df = pd.concat(shap_importances, axis=1).mean(axis=1)
        shap_df = shap_df.sort_values(ascending=False).to_frame("mean_abs_shap")

    return cv_df, models, shap_df


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features(X: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    return X.loc[:, selected_features].copy()


# ============================================================
# OPTUNA PARAMETER SUGGESTIONS
# ============================================================

def suggest_xgb_params(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", XGB_PARAMETER_SPACE["max_depth"][0], XGB_PARAMETER_SPACE["max_depth"][1]),
        "min_child_weight": trial.suggest_int("min_child_weight", XGB_PARAMETER_SPACE["min_child_weight"][0], XGB_PARAMETER_SPACE["min_child_weight"][1]),
        "subsample": trial.suggest_float("subsample", XGB_PARAMETER_SPACE["subsample"][0], XGB_PARAMETER_SPACE["subsample"][1]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", XGB_PARAMETER_SPACE["colsample_bytree"][0], XGB_PARAMETER_SPACE["colsample_bytree"][1]),
        "learning_rate": trial.suggest_float("learning_rate", XGB_PARAMETER_SPACE["learning_rate"][0], XGB_PARAMETER_SPACE["learning_rate"][1], log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", XGB_PARAMETER_SPACE["reg_alpha"][0], XGB_PARAMETER_SPACE["reg_alpha"][1], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", XGB_PARAMETER_SPACE["reg_lambda"][0], XGB_PARAMETER_SPACE["reg_lambda"][1], log=True),
        "gamma": trial.suggest_float("gamma", XGB_PARAMETER_SPACE["gamma"][0], XGB_PARAMETER_SPACE["gamma"][1]),
        "n_estimators": trial.suggest_int("n_estimators", XGB_PARAMETER_SPACE["n_estimators"][0], XGB_PARAMETER_SPACE["n_estimators"][1]),
    }


def suggest_lgbm_params(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", LGBM_PARAMETER_SPACE["num_leaves"][0], LGBM_PARAMETER_SPACE["num_leaves"][1]),
        "max_depth": trial.suggest_int("max_depth", LGBM_PARAMETER_SPACE["max_depth"][0], LGBM_PARAMETER_SPACE["max_depth"][1]),
        "min_child_samples": trial.suggest_int("min_child_samples", LGBM_PARAMETER_SPACE["min_child_samples"][0], LGBM_PARAMETER_SPACE["min_child_samples"][1]),
        "subsample": trial.suggest_float("subsample", LGBM_PARAMETER_SPACE["subsample"][0], LGBM_PARAMETER_SPACE["subsample"][1]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", LGBM_PARAMETER_SPACE["colsample_bytree"][0], LGBM_PARAMETER_SPACE["colsample_bytree"][1]),
        "learning_rate": trial.suggest_float("learning_rate", LGBM_PARAMETER_SPACE["learning_rate"][0], LGBM_PARAMETER_SPACE["learning_rate"][1], log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", LGBM_PARAMETER_SPACE["reg_alpha"][0], LGBM_PARAMETER_SPACE["reg_alpha"][1], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", LGBM_PARAMETER_SPACE["reg_lambda"][0], LGBM_PARAMETER_SPACE["reg_lambda"][1], log=True),
        "n_estimators": trial.suggest_int("n_estimators", LGBM_PARAMETER_SPACE["n_estimators"][0], LGBM_PARAMETER_SPACE["n_estimators"][1]),
    }


# ============================================================
# OPTUNA OPTIMIZATION
# ============================================================

def optimize_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    gap: int,
    n_trials: int = N_TRIALS,
    n_splits: int = N_SPLITS,
    study_name: str | None = None,
    storage: str | None = None,
):
    """
    Stage 2: after SHAP feature selection, optimize hyperparameters with Optuna.
    Objective = mean MAE across TimeSeriesSplit folds.
    """
    if model_name not in {"xgboost", "lightgbm"}:
        raise ValueError(f"{model_name} is not supported")

    def objective(trial: optuna.Trial):
        if model_name == "xgboost":
            params = suggest_xgb_params(trial)
            model_factory = lambda: make_xgb_model(params=params, early_stopping=50)
        else:
            params = suggest_lgbm_params(trial)
            model_factory = lambda: make_lgb_model(params=params)

        cv_df, _, _ = make_time_series_cv_results(
            X=X,
            y=y,
            model_factory=model_factory,
            n_splits=n_splits,
            gap=gap,
            compute_shap=False,
        )

        mean_mae = float(cv_df["mae"].mean())
        mean_rmse = float(cv_df["rmse"].mean())
        mean_r2 = float(cv_df["r2"].mean())

        trial.set_user_attr("mean_rmse", mean_rmse)
        trial.set_user_attr("mean_r2", mean_r2)
        trial.set_user_attr("model_name", model_name)

        return mean_mae

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None and study_name is not None,
    )

    study.optimize(objective, n_trials=n_trials)

    trials_df = study.trials_dataframe()

    best_params = study.best_trial.params.copy()
    best_params_df = pd.DataFrame([{
        **best_params,
        "best_mae": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_rmse": study.best_trial.user_attrs.get("mean_rmse"),
        "best_r2": study.best_trial.user_attrs.get("mean_r2"),
    }])

    return study, trials_df, best_params_df


def evaluate_best_params(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    best_params: dict,
    gap: int,
    n_splits: int = N_SPLITS,
):
    if model_name == "xgboost":
        model_factory = lambda: make_xgb_model(params=best_params, early_stopping=50)
    elif model_name == "lightgbm":
        model_factory = lambda: make_lgb_model(params=best_params)
    else:
        raise ValueError(f"{model_name} is not supported.")

    cv_df, models, _ = make_time_series_cv_results(
        X=X,
        y=y,
        model_factory=model_factory,
        n_splits=n_splits,
        gap=gap,
        compute_shap=False,
    )
    return cv_df, models


# ============================================================
# MAIN
# ============================================================

def model_prediction(model_name: str):
    if model_name == 'xgboost':
        model_factory = lambda: make_xgb_model(early_stopping=50)
    elif model_name == 'lightgbm':
        model_factory = make_lgb_model
    else:
        raise ValueError(f"{model_name} is not supported.")

    df = pd.read_parquet("data/processed/aggregated_dataset.parquet")
    df = df.reset_index(drop=True)  # keep chronological order

    city_cols, calendar_cols = infer_column_groups(df, CITIES, TARGET_MAP)

    FEATURE_MAP = {
        "seoul": calendar_cols + city_cols["seoul"],
        "pohang": calendar_cols + city_cols["pohang"],
    }

    all_results = []
    all_models = {}
    all_shap_rows = []
    all_trials = []
    all_best_params = []

    for (origin, destination), _ in TARGET_MAP.items():
        for horizon_name, horizon in HORIZONS.items():
            print(
                f"\n{'='*80}\n"
                f"Model: {model_name} | Origin: {origin} | Destination: {destination} | Horizon: {horizon_name}\n"
                f"{'='*80}"
            )

            data = build_model_dataset(
                df=df,
                origin=origin,
                destination=destination,
                horizon=horizon,
                max_horizon=MAX_HORIZON,
                feature_map=FEATURE_MAP,
                target_map=TARGET_MAP,
                target_mode="total",
            )

            target_col = f"Transport_passengers_{origin}_to_{destination}"
            X_train_val, y_train_val, X_final_test, y_final_test = get_xy_holdout(data, target_col)

            # SHAP feature ranking
            cv_full, models_full, shap_df = make_time_series_cv_results(
                X=X_train_val,
                y=y_train_val,
                model_factory=model_factory,
                n_splits=N_SPLITS,
                gap=horizon,
                compute_shap=True,
            )

            save_shap_importance(
                shap_df=shap_df,
                origin=origin,
                destination=destination,
                horizon_name=horizon_name,
                output_dir=f"results/shap_{model_name}/"
            )

            shap_out = (
                shap_df.reset_index()
                .rename(columns={"index": "feature", "mean_abs_shap": "importance"})
            )
            shap_out["origin"] = origin
            shap_out["destination"] = destination
            shap_out["horizon"] = horizon_name
            shap_out["horizon_days"] = horizon
            all_shap_rows.append(shap_out)

            # Keep top-k features for THIS model only
            top_features = shap_df.head(TOP_K_FEATURES).index.tolist()
            X_train_val_sel = select_features(X_train_val, top_features)
            X_final_test_sel = select_features(X_final_test, top_features)

            # Refit on selected features
            cv_sel, models_sel, _ = make_time_series_cv_results(
                X=X_train_val_sel,
                y=y_train_val,
                model_factory=model_factory,
                n_splits=N_SPLITS,
                gap=horizon,
                compute_shap=False,
            )

            # Optuna tuning on selected features only
            study_name = f"{model_name}_{origin}_{destination}_{horizon_name}_optuna"
            study, trials_df, best_params_df = optimize_with_optuna(
                X=X_train_val_sel,
                y=y_train_val,
                model_name=model_name,
                gap=horizon,
                n_trials=N_TRIALS,
                n_splits=N_SPLITS,
                study_name=study_name,
            )

            # Export Optuna results
            case_prefix = f"{model_name}_{origin}_to_{destination}_{horizon_name}"
            trials_df.to_csv(f"results/optuna/{case_prefix}_trials.csv", index=False)
            best_params_df.to_csv(f"results/optuna/{case_prefix}_best_params.csv", index=False)

            # Evaluate best params once more for clean final score
            best_params = study.best_trial.params.copy()
            cv_best, models_best = evaluate_best_params(
                X=X_train_val_sel,
                y=y_train_val,
                model_name=model_name,
                best_params=best_params,
                gap=horizon,
                n_splits=N_SPLITS,
            )

            if model_name == 'xgboost':
                models_final = make_xgb_model(params=best_params)
            elif model_name == 'lightgbm':
                models_final = make_lgb_model(params=best_params)
            else:
                raise ValueError(f"{model_name} is not supported.")

            models_final.fit(X_train_val_sel, y_train_val)

            final_shap_df = compute_final_shap_analysis(models_final, X_train_val_sel)
            final_shap_df.to_csv(f"results/shap_{model_name}/{origin}_to_{destination}_{horizon_name}_final_shap.csv", index=False)

            pred_final_test = models_final.predict(X_final_test_sel)

            pred_df = pd.DataFrame({
                "y_true": y_final_test.values,
                "y_pred": pred_final_test,
            })

            pred_df["error"] = pred_df["y_true"] - pred_df["y_pred"]
            pred_df["abs_error"] = pred_df["error"].abs()

            pred_df.to_csv(
                f"results/final_test_predictions/"
                f"{model_name}_{origin}_to_{destination}_{horizon_name}.csv",
                index=False,
            )

            test_mae = mean_absolute_error(y_final_test, pred_final_test)
            test_rmse = root_mean_squared_error(y_final_test, pred_final_test)
            test_r2 = r2_score(y_final_test, pred_final_test)

            row = {
                "origin": origin,
                "destination": destination,
                "horizon": horizon_name,
                "horizon_days": horizon,
                "n_features_selected": len(top_features),
                "mae_full": cv_full["mae"].mean(),
                "rmse_full": cv_full["rmse"].mean(),
                "r2_full": cv_full["r2"].mean(),
                "mae_shap": cv_sel["mae"].mean(),
                "rmse_shap": cv_sel["rmse"].mean(),
                "r2_shap": cv_sel["r2"].mean(),
                "best_trial_number": int(study.best_trial.number),
                "mae_best": float(cv_best["mae"].mean()),
                "rmse_best": float(cv_best["rmse"].mean()),
                "r2_best": float(cv_best["r2"].mean()),
                "n_trials": int(N_TRIALS),
                "mae_test": test_mae,
                "rmse_test": test_rmse,
                "r2_test": test_r2,
            }
            all_results.append(row)

            all_models[(origin, destination, horizon_name)] = {
                "models_full": models_full,
                "models_shap": models_sel,
                "models_best": models_best,
                "models_final": models_final,
                "selected_features": top_features,
                "cv_full": cv_full,
                "cv_selected": cv_sel,
                "cv_best": cv_best,
            }

            all_trials.append(
                trials_df.assign(
                    model=model_name,
                    origin=origin,
                    destination=destination,
                    horizon=horizon_name,
                    horizon_days=horizon,
                )
            )

            all_best_params.append(
                best_params_df.assign(
                    model=model_name,
                    origin=origin,
                    destination=destination,
                    horizon=horizon_name,
                    horizon_days=horizon,
                )
            )

            print(f"\nFULL: MAE={row['mae_full']:.3f} | RMSE={row['rmse_full']:.3f} | R²={row['r2_full']:.3f}")
            print(f"SHAP: MAE={row['mae_shap']:.3f} | RMSE={row['rmse_shap']:.3f} | R²={row['r2_shap']:.3f}")
            print(f"BEST: MAE={row['mae_best']:.3f} | RMSE={row['rmse_best']:.3f} | R²={row['r2_best']:.3f}")
            print(f"TEST: MAE={row['mae_test']:.3f} | RMSE={row['rmse_test']:.3f} | R²={row['r2_test']:.3f}")
            print(f"Top features: {top_features[:10]}")

    summary_df = pd.DataFrame(all_results).sort_values(
        ["origin", "destination", "horizon_days"]
    ).reset_index(drop=True)

    print("\nFINAL SUMMARY")
    print(summary_df)

    all_trials_df = pd.concat(all_trials, ignore_index=True)
    all_best_params_df = pd.concat(all_best_params, ignore_index=True)

    all_trials_df.to_csv(f"results/optuna/{model_name}_all_trials.csv", index=False)
    all_best_params_df.to_csv(f"results/optuna/{model_name}_all_best_params.csv", index=False)
    summary_df.to_csv(f"results/model_comparison_{model_name}_shap_top_20.csv", index=False)

    if all_shap_rows:
        shap_df_all = pd.concat(all_shap_rows, ignore_index=True)
        shap_df_all.to_csv(f"results/{model_name}_shap_importance.csv", index=False)
        print(f"Saved global SHAP table to results/{model_name}_shap_importance.csv")


if __name__ == "__main__":
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/shap_xgboost").mkdir(parents=True, exist_ok=True)
    Path("results/shap_lightgbm").mkdir(parents=True, exist_ok=True)
    Path("results/optuna").mkdir(parents=True, exist_ok=True)
    Path("results/final_test_predictions").mkdir(parents=True, exist_ok=True)
    model_prediction("xgboost")
    model_prediction("lightgbm")
