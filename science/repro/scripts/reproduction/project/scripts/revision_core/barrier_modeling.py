"""No-leakage grouped-CV model selection and strict OOF evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from revision_core.barrier_features import assert_no_leakage_features
from revision_core.data_governance import canonical_json


def deterministic_runtime_environment() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def derive_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed)] + [str(value) for value in parts])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


class MedianMissingFeaturizer(object):
    """Fold-fitted median imputation with explicit missing indicators.

    Columns that are entirely unavailable in a training fold are not assigned a
    numeric placeholder; the value column is dropped and its missing indicator
    remains.  No feature value is filled with semantic zero.
    """

    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)
        self.value_columns: List[str] = []
        self.medians: Dict[str, float] = {}
        self.output_columns: List[str] = []

    @staticmethod
    def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        output = pd.DataFrame(index=frame.index)
        for column in columns:
            if column not in frame.columns:
                output[column] = np.nan
            else:
                output[column] = pd.to_numeric(frame[column], errors="coerce")
        return output.replace([np.inf, -np.inf], np.nan)

    def fit(self, frame: pd.DataFrame) -> "MedianMissingFeaturizer":
        numeric = self._numeric(frame, self.columns)
        self.value_columns = []
        self.medians = {}
        for column in self.columns:
            finite = numeric[column].dropna()
            if len(finite):
                self.value_columns.append(column)
                self.medians[column] = float(finite.median())
        self.output_columns = list(self.value_columns) + ["missing__%s" % column for column in self.columns]
        return self

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = self._numeric(frame, self.columns)
        output = pd.DataFrame(index=frame.index)
        for column in self.value_columns:
            output[column] = numeric[column].fillna(self.medians[column]).astype(float)
        for column in self.columns:
            output["missing__%s" % column] = numeric[column].isna().astype(float)
        return output[self.output_columns]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform_frame(frame).values.astype(float)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.fit(frame)
        return self.transform(frame)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "columns": self.columns,
            "value_columns": self.value_columns,
            "medians": self.medians,
            "output_columns": self.output_columns,
            "all_missing_policy": "drop value column, retain missing indicator",
            "partial_missing_policy": "fold-training median plus missing indicator",
        }


@dataclass
class RevisionRegressorPipeline:
    model_id: str
    feature_columns: List[str]
    params: Dict[str, Any]
    seed: int
    featurizer: Optional[MedianMissingFeaturizer] = None
    scaler: Optional[StandardScaler] = None
    estimator: Any = None

    def fit(self, frame: pd.DataFrame, target: Sequence[float]) -> "RevisionRegressorPipeline":
        self.featurizer = MedianMissingFeaturizer(self.feature_columns)
        matrix = self.featurizer.fit_transform(frame)
        self.scaler = StandardScaler() if self.model_id == "SVR" else None
        if self.scaler is not None:
            matrix = self.scaler.fit_transform(matrix)
        self.estimator = make_estimator(self.model_id, self.params, self.seed)
        self.estimator.fit(matrix, np.asarray(target, dtype=float))
        return self

    def encoded_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if self.featurizer is None:
            raise RuntimeError("Pipeline is not fitted")
        matrix = self.featurizer.transform(frame)
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        return matrix

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(self.encoded_matrix(frame)), dtype=float)


def make_estimator(model_id: str, params: Dict[str, Any], seed: int) -> Any:
    clean = dict(params)
    if model_id == "DummyMean":
        return DummyRegressor(strategy="mean")
    if model_id == "SVR":
        return SVR(**clean)
    if model_id == "ExtraTrees":
        return ExtraTreesRegressor(random_state=seed, **clean)
    if model_id == "RandomForest":
        return RandomForestRegressor(random_state=seed, **clean)
    if model_id == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=seed, **clean)
    if model_id == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            seed=seed,
            verbosity=0,
            **clean
        )
    if model_id == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            random_state=seed,
            bagging_seed=seed,
            feature_fraction_seed=seed,
            data_random_seed=seed,
            verbosity=-1,
            **clean
        )
    if model_id == "CatBoost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(random_seed=seed, **clean)
    raise ValueError("Unknown model_id: %s" % model_id)


def available_model_versions(model_ids: Iterable[str]) -> Dict[str, str]:
    import sklearn

    versions = {"scikit-learn": sklearn.__version__}
    if "XGBoost" in model_ids:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    if "LightGBM" in model_ids:
        import lightgbm

        versions["lightgbm"] = lightgbm.__version__
    if "CatBoost" in model_ids:
        import catboost

        versions["catboost"] = catboost.__version__
    return versions


def make_validation_splits(
    frame: pd.DataFrame,
    validation_id: str,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    primary = config["validation"]["primary_model_selection"]
    stress = {item["id"]: item for item in config["validation"]["stress_tests"]}
    positions = np.arange(len(frame), dtype=int)
    splits: List[Dict[str, Any]] = []
    if validation_id == primary["id"]:
        group_column = primary["group_column"]
        groups = frame[group_column].astype(str).values
        unique_count = len(np.unique(groups))
        n_splits = int(primary["outer_splits"])
        if unique_count < n_splits:
            raise RuntimeError("Insufficient groups for primary GroupKFold; ordinary KFold fallback is forbidden")
        splitter = GroupKFold(n_splits=n_splits)
        iterator = splitter.split(positions, np.zeros(len(frame)), groups)
        for fold, (train, test) in enumerate(iterator, start=1):
            splits.append({"fold": fold, "train": train, "test": test, "group_column": group_column})
    elif validation_id in stress and stress[validation_id]["splitter"] == "LeaveOneGroupOut":
        group_column = stress[validation_id]["group_column"]
        groups = frame[group_column].astype(str).values
        if len(np.unique(groups)) < 2:
            raise RuntimeError("LeaveOneGroupOut requires at least two groups")
        for fold, (train, test) in enumerate(LeaveOneGroupOut().split(positions, np.zeros(len(frame)), groups), start=1):
            splits.append({"fold": fold, "train": train, "test": test, "group_column": group_column})
    elif validation_id in stress and stress[validation_id]["splitter"] == "NASICONCompositionLOGO":
        group_column = stress[validation_id]["group_column"]
        scoped = frame[frame["is_nasicon"].astype(int).eq(1)]
        groups = sorted(scoped[group_column].astype(str).unique())
        if len(groups) < 2:
            raise RuntimeError("NASICON composition validation has fewer than two groups")
        all_groups = frame[group_column].astype(str).values
        for fold, held_group in enumerate(groups, start=1):
            test = np.where((frame["is_nasicon"].astype(int).values == 1) & (all_groups == held_group))[0]
            train = np.where(all_groups != held_group)[0]
            splits.append({"fold": fold, "train": train, "test": test, "group_column": group_column})
    else:
        raise ValueError("Unknown validation_id: %s" % validation_id)
    validate_splits(frame, validation_id, splits)
    return splits


def validate_splits(frame: pd.DataFrame, validation_id: str, splits: Sequence[Dict[str, Any]]) -> None:
    seen = CounterLike()
    for split in splits:
        train = np.asarray(split["train"], dtype=int)
        test = np.asarray(split["test"], dtype=int)
        if len(set(train).intersection(set(test))):
            raise RuntimeError("Train/test row overlap in %s fold %s" % (validation_id, split["fold"]))
        group_column = split["group_column"]
        train_groups = set(frame.iloc[train][group_column].astype(str))
        test_groups = set(frame.iloc[test][group_column].astype(str))
        if train_groups.intersection(test_groups):
            raise RuntimeError("Train/test group leakage in %s fold %s" % (validation_id, split["fold"]))
        for index in test:
            seen.add(int(index))
    expected = set(frame.index[frame["is_nasicon"].astype(int).eq(1)]) if "NASICON" in validation_id else set(frame.index)
    observed = set(seen.keys())
    if observed != expected or any(seen[index] != 1 for index in observed):
        raise RuntimeError("OOF coverage is not exactly once for validation %s" % validation_id)


class CounterLike(dict):
    def add(self, key: int) -> None:
        self[key] = int(self.get(key, 0)) + 1


def _split_hash(frame: pd.DataFrame, train: Sequence[int], test: Sequence[int]) -> str:
    payload = {
        "train_record_ids": sorted(frame.iloc[list(train)]["record_id"].astype(str)),
        "test_record_ids": sorted(frame.iloc[list(test)]["record_id"].astype(str)),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def run_oof(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    feature_set_id: str,
    model_id: str,
    validation_id: str,
    cohort_id: str,
    role: str,
    config: Dict[str, Any],
    splits: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    assert_no_leakage_features(feature_columns, config)
    if not feature_columns and model_id != "DummyMean":
        raise ValueError("Non-baseline model has no features")
    data = frame.reset_index(drop=True)
    y = pd.to_numeric(data["barrier_eV"], errors="coerce").values.astype(float)
    if not np.all(np.isfinite(y)):
        raise ValueError("Target contains non-finite values")
    configured_splits = list(splits or make_validation_splits(data, validation_id, config))
    base_seed = int(config["random_seeds"]["base_seed"])
    rows: List[Dict[str, Any]] = []
    fold_details: List[Dict[str, Any]] = []
    for split in configured_splits:
        fold = int(split["fold"])
        train = np.asarray(split["train"], dtype=int)
        test = np.asarray(split["test"], dtype=int)
        seed = derive_seed(base_seed, cohort_id, validation_id, feature_set_id, model_id, fold)
        params = dict(config["models"][model_id])
        pipeline = RevisionRegressorPipeline(model_id, list(feature_columns), params, seed)
        started = time.time()
        pipeline.fit(data.iloc[train], y[train])
        prediction = pipeline.predict(data.iloc[test])
        elapsed = time.time() - started
        split_hash = _split_hash(data, train, test)
        group_column = split["group_column"]
        for position, predicted in zip(test, prediction):
            row = data.iloc[int(position)]
            rows.append(
                {
                    "record_id": str(row["record_id"]),
                    "raw_index": int(row["raw_index"]),
                    "jid": str(row["jid"]),
                    "cohort_id": cohort_id,
                    "validation_id": validation_id,
                    "role": role,
                    "feature_set_id": feature_set_id,
                    "model_id": model_id,
                    "fold_id": fold,
                    "held_out_group": str(row[group_column]),
                    "y_true_eV": float(y[int(position)]),
                    "y_pred_eV": float(predicted),
                    "residual_eV": float(y[int(position)] - predicted),
                    "absolute_error_eV": float(abs(y[int(position)] - predicted)),
                    "formula": str(row["formula"]),
                    "reduced_formula": str(row["reduced_formula"]),
                    "migrating_species": str(row["migrating_species"]),
                    "source_group": str(row["source_group"]),
                    "structure_family_normalized": str(row["structure_family_normalized"]),
                    "is_nasicon": int(row["is_nasicon"]),
                    "seed": seed,
                    "train_n": len(train),
                    "test_n": len(test),
                    "raw_feature_count": len(feature_columns),
                    "encoded_feature_count": len(pipeline.featurizer.output_columns),
                    "split_sha256": split_hash,
                }
            )
        fold_details.append(
            {
                "cohort_id": cohort_id,
                "validation_id": validation_id,
                "feature_set_id": feature_set_id,
                "model_id": model_id,
                "fold_id": fold,
                "seed": seed,
                "train_n": len(train),
                "test_n": len(test),
                "held_out_groups": sorted(set(data.iloc[test][group_column].astype(str))),
                "split_sha256": split_hash,
                "fit_predict_seconds": elapsed,
                "encoded_feature_count": len(pipeline.featurizer.output_columns),
            }
        )
    predictions = pd.DataFrame(rows).sort_values(["fold_id", "record_id"]).reset_index(drop=True)
    expected_n = int(data["is_nasicon"].sum()) if "NASICON" in validation_id else len(data)
    if len(predictions) != expected_n or predictions["record_id"].duplicated().any():
        raise RuntimeError("OOF output is incomplete or duplicated for %s" % validation_id)
    metric = aggregate_metrics(predictions, cohort_id, validation_id, feature_set_id, model_id, role)
    return predictions, metric, fold_details


def aggregate_metrics(
    predictions: pd.DataFrame,
    cohort_id: str,
    validation_id: str,
    feature_set_id: str,
    model_id: str,
    role: str,
) -> Dict[str, Any]:
    true = predictions["y_true_eV"].values.astype(float)
    pred = predictions["y_pred_eV"].values.astype(float)
    group_mae = predictions.groupby("held_out_group")["absolute_error_eV"].mean()
    fold_mae = predictions.groupby("fold_id")["absolute_error_eV"].mean()
    rho = float(spearmanr(true, pred).correlation) if len(true) >= 3 else math.nan
    tau = float(kendalltau(true, pred).correlation) if len(true) >= 3 else math.nan
    status = "OK"
    r2 = float(r2_score(true, pred)) if len(true) >= 2 else math.nan
    if "NASICON" in validation_id and cohort_id == "strict_local":
        status = "INSUFFICIENT_SCOPE_DESCRIPTIVE_ONLY"
        rho = math.nan
        tau = math.nan
        r2 = math.nan
    if "NASICON" in validation_id and cohort_id == "composition_all619":
        status = "SENSITIVITY_ONLY_SOURCE_XC_CONFOUNDED"
    return {
        "cohort_id": cohort_id,
        "validation_id": validation_id,
        "role": role,
        "feature_set_id": feature_set_id,
        "model_id": model_id,
        "n_predictions": len(predictions),
        "n_groups": int(predictions["held_out_group"].nunique()),
        "MAE_eV": float(mean_absolute_error(true, pred)),
        "RMSE_eV": float(math.sqrt(mean_squared_error(true, pred))),
        "MedianAE_eV": float(median_absolute_error(true, pred)),
        "R2": r2,
        "Spearman_rho": rho,
        "Kendall_tau": tau,
        "group_macro_MAE_eV": float(group_mae.mean()),
        "group_macro_MAE_std_eV": float(group_mae.std(ddof=0)),
        "fold_MAE_mean_eV": float(fold_mae.mean()),
        "fold_MAE_std_eV": float(fold_mae.std(ddof=0)),
        "raw_feature_count": int(predictions["raw_feature_count"].iloc[0]),
        "encoded_feature_count_mean": float(predictions["encoded_feature_count"].mean()),
        "status": status,
    }


def rank_model_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics.copy()
    ranked["_rho_sort"] = -pd.to_numeric(ranked["Spearman_rho"], errors="coerce").fillna(-np.inf)
    ranked = ranked.sort_values(["MAE_eV", "RMSE_eV", "_rho_sort", "model_id"]).drop(columns=["_rho_sort"])
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
    return ranked.reset_index(drop=True)


def _inner_group_splits(frame: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    primary = config["validation"]["primary_model_selection"]
    group_column = primary["group_column"]
    groups = frame[group_column].astype(str).values
    n_splits = int(primary["inner_splits"])
    if len(np.unique(groups)) < n_splits:
        raise RuntimeError("Insufficient inner groups; ordinary KFold fallback is forbidden")
    output = []
    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=n_splits).split(np.arange(len(frame)), np.zeros(len(frame)), groups), start=1
    ):
        output.append({"fold": fold, "train": train, "test": test, "group_column": group_column})
    validate_splits(frame.reset_index(drop=True), "inner_primary", output)
    return output


def nested_selected_primary_oof(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    feature_set_id: str,
    model_ids: Sequence[str],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    data = frame.reset_index(drop=True)
    primary_id = config["validation"]["primary_model_selection"]["id"]
    outer_splits = make_validation_splits(data, primary_id, config)
    y = data["barrier_eV"].astype(float).values
    base_seed = int(config["random_seeds"]["base_seed"])
    all_rows: List[Dict[str, Any]] = []
    selection_details: List[Dict[str, Any]] = []
    for outer in outer_splits:
        outer_fold = int(outer["fold"])
        outer_train = np.asarray(outer["train"], dtype=int)
        outer_test = np.asarray(outer["test"], dtype=int)
        train_frame = data.iloc[outer_train].reset_index(drop=True)
        inner_splits = _inner_group_splits(train_frame, config)
        candidate_metrics = []
        candidate_split_details: Dict[str, Any] = {}
        for model_id in model_ids:
            _, metric, inner_details = run_oof(
                train_frame,
                feature_columns,
                feature_set_id,
                model_id,
                primary_id,
                "strict_local_inner_outer_%d" % outer_fold,
                "inner_model_selection",
                config,
                splits=inner_splits,
            )
            candidate_metrics.append(metric)
            candidate_split_details[model_id] = inner_details
        ranked = rank_model_metrics(pd.DataFrame(candidate_metrics))
        selected_model = str(ranked.iloc[0]["model_id"])
        seed = derive_seed(base_seed, "nested_outer", outer_fold, selected_model)
        pipeline = RevisionRegressorPipeline(
            selected_model,
            list(feature_columns),
            dict(config["models"][selected_model]),
            seed,
        )
        pipeline.fit(data.iloc[outer_train], y[outer_train])
        predicted = pipeline.predict(data.iloc[outer_test])
        split_hash = _split_hash(data, outer_train, outer_test)
        group_column = outer["group_column"]
        for position, value in zip(outer_test, predicted):
            row = data.iloc[int(position)]
            all_rows.append(
                {
                    "record_id": str(row["record_id"]),
                    "raw_index": int(row["raw_index"]),
                    "jid": str(row["jid"]),
                    "cohort_id": "strict_local",
                    "validation_id": primary_id,
                    "role": "nested_selected_unbiased",
                    "feature_set_id": feature_set_id,
                    "model_id": "NestedSelected",
                    "fold_selected_model_id": selected_model,
                    "fold_id": outer_fold,
                    "held_out_group": str(row[group_column]),
                    "y_true_eV": float(y[int(position)]),
                    "y_pred_eV": float(value),
                    "residual_eV": float(y[int(position)] - value),
                    "absolute_error_eV": float(abs(y[int(position)] - value)),
                    "formula": str(row["formula"]),
                    "reduced_formula": str(row["reduced_formula"]),
                    "migrating_species": str(row["migrating_species"]),
                    "source_group": str(row["source_group"]),
                    "structure_family_normalized": str(row["structure_family_normalized"]),
                    "is_nasicon": int(row["is_nasicon"]),
                    "seed": seed,
                    "train_n": len(outer_train),
                    "test_n": len(outer_test),
                    "raw_feature_count": len(feature_columns),
                    "encoded_feature_count": len(pipeline.featurizer.output_columns),
                    "split_sha256": split_hash,
                }
            )
        selection_details.append(
            {
                "outer_fold": outer_fold,
                "selected_model_id": selected_model,
                "candidate_ranking": ranked.to_dict(orient="records"),
                "inner_split_details": candidate_split_details,
                "split_sha256": split_hash,
            }
        )
    predictions = pd.DataFrame(all_rows).sort_values(["fold_id", "record_id"]).reset_index(drop=True)
    if len(predictions) != len(data) or predictions["record_id"].duplicated().any():
        raise RuntimeError("Nested selected OOF is incomplete")
    metric = aggregate_metrics(
        predictions,
        "strict_local",
        primary_id,
        feature_set_id,
        "NestedSelected",
        "nested_selected_unbiased",
    )
    return predictions, metric, selection_details


def fit_final_pipeline(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    model_id: str,
    config: Dict[str, Any],
) -> RevisionRegressorPipeline:
    assert_no_leakage_features(feature_columns, config)
    seed = derive_seed(int(config["random_seeds"]["base_seed"]), "final", model_id)
    pipeline = RevisionRegressorPipeline(
        model_id,
        list(feature_columns),
        dict(config["models"][model_id]),
        seed,
    )
    return pipeline.fit(frame.reset_index(drop=True), frame["barrier_eV"].astype(float).values)


def save_native_model_if_available(pipeline: RevisionRegressorPipeline, directory: Any) -> Optional[str]:
    output_dir = directory
    if pipeline.model_id == "XGBoost":
        path = output_dir / "selected_model_native.json"
        pipeline.estimator.save_model(str(path))
        return str(path)
    if pipeline.model_id == "LightGBM":
        path = output_dir / "selected_model_native.txt"
        pipeline.estimator.booster_.save_model(str(path))
        return str(path)
    if pipeline.model_id == "CatBoost":
        path = output_dir / "selected_model_native.cbm"
        pipeline.estimator.save_model(str(path))
        return str(path)
    return None
