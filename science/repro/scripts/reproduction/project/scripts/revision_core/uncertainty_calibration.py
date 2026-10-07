"""Strict group-aware calibration and train-only applicability diagnostics.

This module deliberately separates two quantities that answer different
questions:

* :class:`GroupMaxSplitConformal` calibrates prediction intervals from labelled
  calibration groups.  Its nonconformity unit is the maximum absolute residual
  within a group, so its primary reported coverage is simultaneous coverage of
  held-out groups.
* :class:`FourBlockPCADistanceOOD` is a label-free applicability diagnostic.  It
  fits imputation, scaling, block balancing, PCA, and the reference ECDF on a
  supplied training fold only.  A test/query fold is accepted only by
  :meth:`score` after the reference has been frozen.

Tree-to-tree spread is exposed only as ``ensemble_disagreement``.  It is not a
calibrated interval and no standard-deviation alias is provided.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from revision_core.barrier_features import LeakageFeatureError, assert_no_leakage_features
from revision_core.barrier_modeling import MedianMissingFeaturizer


FOUR_FEATURE_BLOCKS: Tuple[str, ...] = (
    "A_composition",
    "B_local_geometry",
    "C_local_chemistry",
    "D_occupancy_dopant_configuration",
)

ENSEMBLE_DISAGREEMENT_COLUMN = "ensemble_disagreement_eV"

DEFAULT_FORBIDDEN_INPUT_PATTERNS: Tuple[str, ...] = (
    "barrier",
    "target",
    "predicted",
    "prediction",
    "mfpt",
    "mean_free_passage",
    "meanfreepassage",
    "mean_first_passage",
    "first_passage",
    "random_walk",
    "randomwalk",
    "percolation",
    "eperc",
    "critical",
    "criticality",
    "topology",
    "winding",
    "periodic_dimensionality",
    "redundancy_count",
    "arrhenius",
    "kinetic",
    "diffusion_rate",
    "conductivity",
    "activation_energy",
    "uncertainty",
    "selection_score",
)

DEFAULT_FORBIDDEN_METADATA_INPUTS: Tuple[str, ...] = (
    "formula",
    "reduced_formula",
    "source_group",
    "xc",
    "bibtex",
    "metadata_completeness",
    "quality_flag",
    "duplicate_group_id",
    "crystal_class",
    "space_group",
    "sys_name",
    "structure_ini_hash",
    "structure_fin_hash",
    "raw_record_sha256",
)


class FiniteSampleCalibrationError(ValueError):
    """Raised when an exact split-conformal quantile does not exist."""


class OODReferenceError(ValueError):
    """Raised when a train-only OOD reference cannot be constructed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _as_positions(values: Sequence[int], n_rows: int, label: str) -> np.ndarray:
    positions = np.asarray(values, dtype=int)
    if positions.ndim != 1:
        raise ValueError("%s indices must be one-dimensional" % label)
    if len(np.unique(positions)) != len(positions):
        raise ValueError("%s indices contain duplicates" % label)
    if len(positions) and (int(positions.min()) < 0 or int(positions.max()) >= n_rows):
        raise IndexError("%s indices are outside the frame" % label)
    return positions


def _as_groups(values: Sequence[Any], expected_n: Optional[int] = None) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError("Group labels must be one-dimensional")
    if expected_n is not None and len(raw) != expected_n:
        raise ValueError("Group-label length does not match row count")
    if any(pd.isna(value) or not str(value).strip() for value in raw):
        raise ValueError("Group labels must be non-missing and non-empty")
    return np.asarray([str(value) for value in raw], dtype=object)


def _as_finite_vector(values: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("%s must be one-dimensional" % label)
    if not np.all(np.isfinite(array)):
        raise ValueError("%s contains non-finite values" % label)
    return array


def minimum_calibration_groups(confidence_level: float) -> int:
    """Return the smallest group count supporting the exact finite-sample rank."""

    level = float(confidence_level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")
    count = 1
    while int(math.ceil((count + 1) * level - 1e-12)) > count:
        count += 1
    return count


@dataclass
class GroupTrainCalibrationTestSplit:
    """Immutable-by-convention row and group membership for one outer fold."""

    proper_train_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    proper_train_groups: List[str]
    calibration_groups: List[str]
    test_groups: List[str]
    group_column: str
    calibration_fraction_requested: float
    selection_context: str
    split_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proper_train_indices": [int(value) for value in self.proper_train_indices],
            "calibration_indices": [int(value) for value in self.calibration_indices],
            "test_indices": [int(value) for value in self.test_indices],
            "proper_train_groups": list(self.proper_train_groups),
            "calibration_groups": list(self.calibration_groups),
            "test_groups": list(self.test_groups),
            "group_column": self.group_column,
            "calibration_fraction_requested": float(self.calibration_fraction_requested),
            "selection_context": self.selection_context,
            "split_sha256": self.split_sha256,
            "selection_policy": "SHA256 rank of group labels and context; no target values",
        }


def validate_group_partition(
    frame: pd.DataFrame,
    proper_train_indices: Sequence[int],
    calibration_indices: Sequence[int],
    test_indices: Sequence[int],
    group_column: str,
    require_all_rows: bool = True,
) -> None:
    """Fail closed on row overlap, group overlap, or incomplete membership."""

    if group_column not in frame.columns:
        raise KeyError("Missing group column: %s" % group_column)
    groups = _as_groups(frame[group_column].values, len(frame))
    train = _as_positions(proper_train_indices, len(frame), "proper_train")
    calibration = _as_positions(calibration_indices, len(frame), "calibration")
    test = _as_positions(test_indices, len(frame), "test")
    row_sets = [set(train), set(calibration), set(test)]
    if row_sets[0] & row_sets[1] or row_sets[0] & row_sets[2] or row_sets[1] & row_sets[2]:
        raise ValueError("Proper-train, calibration, and test rows must be disjoint")
    if require_all_rows and set().union(*row_sets) != set(range(len(frame))):
        raise ValueError("Proper-train, calibration, and test rows do not cover the frame exactly once")
    group_sets = [set(groups[train]), set(groups[calibration]), set(groups[test])]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise ValueError("Proper-train, calibration, and test groups must be disjoint")
    if not len(train) or not len(calibration) or not len(test):
        raise ValueError("Proper-train, calibration, and test partitions must all be non-empty")


def deterministic_group_train_calibration_test_split(
    frame: pd.DataFrame,
    outer_train_indices: Sequence[int],
    outer_test_indices: Sequence[int],
    group_column: str,
    calibration_fraction: float = 0.2,
    base_seed: int = 20260807,
    context: str = "",
    confidence_levels: Sequence[float] = (0.8, 0.9),
    require_all_rows: bool = True,
) -> GroupTrainCalibrationTestSplit:
    """Split outer-training groups deterministically without inspecting targets.

    The requested fraction is increased, when necessary, to the minimum number
    of calibration groups required by all requested confidence levels.  It is
    never allowed to consume every outer-training group.
    """

    fraction = float(calibration_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between zero and one")
    if group_column not in frame.columns:
        raise KeyError("Missing group column: %s" % group_column)
    outer_train = _as_positions(outer_train_indices, len(frame), "outer_train")
    outer_test = _as_positions(outer_test_indices, len(frame), "outer_test")
    if set(outer_train) & set(outer_test):
        raise ValueError("Outer train/test rows overlap")
    if require_all_rows and set(outer_train) | set(outer_test) != set(range(len(frame))):
        raise ValueError("Outer train/test rows do not cover the frame exactly once")
    all_groups = _as_groups(frame[group_column].values, len(frame))
    outer_train_groups = sorted(set(all_groups[outer_train]))
    test_groups = sorted(set(all_groups[outer_test]))
    if set(outer_train_groups) & set(test_groups):
        raise ValueError("Outer train/test groups overlap")
    if len(outer_train_groups) < 2:
        raise ValueError("At least two outer-training groups are required")

    levels = tuple(float(value) for value in confidence_levels)
    minimum = max([minimum_calibration_groups(value) for value in levels] or [1])
    requested = int(math.ceil(fraction * len(outer_train_groups)))
    calibration_count = max(requested, minimum)
    if calibration_count >= len(outer_train_groups):
        raise FiniteSampleCalibrationError(
            "Outer fold has %d training groups but requested intervals require at least %d "
            "calibration groups plus one proper-training group"
            % (len(outer_train_groups), minimum)
        )

    def rank(group: str) -> Tuple[str, str]:
        payload = {
            "base_seed": int(base_seed),
            "context": str(context),
            "group_column": str(group_column),
            "group": group,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(), group

    ranked_groups = sorted(outer_train_groups, key=rank)
    calibration_groups = sorted(ranked_groups[:calibration_count])
    proper_train_groups = sorted(set(outer_train_groups) - set(calibration_groups))
    calibration_set = set(calibration_groups)
    proper_set = set(proper_train_groups)
    calibration = outer_train[np.asarray([group in calibration_set for group in all_groups[outer_train]])]
    proper_train = outer_train[np.asarray([group in proper_set for group in all_groups[outer_train]])]
    proper_train = np.sort(proper_train)
    calibration = np.sort(calibration)
    outer_test = np.sort(outer_test)
    validate_group_partition(
        frame,
        proper_train,
        calibration,
        outer_test,
        group_column,
        require_all_rows=require_all_rows,
    )
    payload = {
        "proper_train_rows": [int(value) for value in proper_train],
        "calibration_rows": [int(value) for value in calibration],
        "test_rows": [int(value) for value in outer_test],
        "proper_train_groups": proper_train_groups,
        "calibration_groups": calibration_groups,
        "test_groups": test_groups,
        "group_column": group_column,
        "base_seed": int(base_seed),
        "context": str(context),
    }
    return GroupTrainCalibrationTestSplit(
        proper_train_indices=proper_train,
        calibration_indices=calibration,
        test_indices=outer_test,
        proper_train_groups=proper_train_groups,
        calibration_groups=calibration_groups,
        test_groups=test_groups,
        group_column=group_column,
        calibration_fraction_requested=fraction,
        selection_context=str(context),
        split_sha256=hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
    )


def validate_interval_order(
    lower: Sequence[float],
    prediction: Sequence[float],
    upper: Sequence[float],
) -> None:
    """Validate finite, ordered intervals containing their point predictions."""

    low = _as_finite_vector(lower, "interval lower bounds")
    point = _as_finite_vector(prediction, "point predictions")
    high = _as_finite_vector(upper, "interval upper bounds")
    if not (len(low) == len(point) == len(high)):
        raise ValueError("Interval arrays have different lengths")
    if np.any(low > point) or np.any(point > high):
        raise ValueError("Intervals must satisfy lower <= prediction <= upper")


def coverage_summary(
    y_true: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    groups: Sequence[Any],
) -> Dict[str, Any]:
    """Return record coverage and group-equal simultaneous coverage."""

    true = _as_finite_vector(y_true, "coverage targets")
    low = _as_finite_vector(lower, "coverage lower bounds")
    high = _as_finite_vector(upper, "coverage upper bounds")
    labels = _as_groups(groups, len(true))
    if not (len(low) == len(high) == len(true)):
        raise ValueError("Coverage arrays have different lengths")
    if np.any(low > high):
        raise ValueError("Coverage interval lower bound exceeds upper bound")
    covered = (true >= low) & (true <= high)
    group_results = []
    for group in sorted(set(labels)):
        group_results.append(bool(np.all(covered[labels == group])))
    widths = high - low
    return {
        "n_records": int(len(true)),
        "n_groups": int(len(group_results)),
        "record_coverage": float(np.mean(covered)),
        "records_covered": int(np.sum(covered)),
        "group_simultaneous_coverage": float(np.mean(group_results)),
        "groups_simultaneously_covered": int(np.sum(group_results)),
        "mean_interval_width_eV": float(np.mean(widths)),
        "median_interval_width_eV": float(np.median(widths)),
        "max_interval_width_eV": float(np.max(widths)),
    }


class GroupMaxSplitConformal(object):
    """Exact finite-sample split conformal using one max-residual score/group."""

    SERIALIZATION_VERSION = 1

    def __init__(self, confidence_levels: Sequence[float] = (0.8, 0.9)):
        levels = tuple(sorted(set(float(value) for value in confidence_levels)))
        if not levels:
            raise ValueError("At least one confidence level is required")
        for level in levels:
            minimum_calibration_groups(level)
        self.confidence_levels = levels
        self.quantiles_eV_: Dict[float, float] = {}
        self.quantile_ranks_: Dict[float, int] = {}
        self.calibration_group_scores_: Optional[pd.DataFrame] = None
        self.n_calibration_records_: Optional[int] = None
        self.n_calibration_groups_: Optional[int] = None
        self.calibration_signature_sha256_: Optional[str] = None

    def _require_fitted(self) -> None:
        if not self.quantiles_eV_ or self.calibration_group_scores_ is None:
            raise RuntimeError("GroupMaxSplitConformal is not fitted")

    def fit(
        self,
        calibration_true: Sequence[float],
        calibration_prediction: Sequence[float],
        calibration_groups: Sequence[Any],
    ) -> "GroupMaxSplitConformal":
        true = _as_finite_vector(calibration_true, "calibration targets")
        prediction = _as_finite_vector(calibration_prediction, "calibration predictions")
        groups = _as_groups(calibration_groups, len(true))
        if len(true) != len(prediction):
            raise ValueError("Calibration targets and predictions have different lengths")
        if not len(true):
            raise ValueError("Calibration data are empty")
        residual = np.abs(true - prediction)
        scores = []
        for group in sorted(set(groups)):
            values = residual[groups == group]
            scores.append(
                {
                    "calibration_group": group,
                    "group_record_count": int(len(values)),
                    "group_max_absolute_residual_eV": float(np.max(values)),
                }
            )
        score_frame = pd.DataFrame(scores).sort_values("calibration_group").reset_index(drop=True)
        ordered = np.sort(score_frame["group_max_absolute_residual_eV"].values.astype(float))
        count = len(ordered)
        quantiles: Dict[float, float] = {}
        ranks: Dict[float, int] = {}
        for level in self.confidence_levels:
            rank = int(math.ceil((count + 1) * level - 1e-12))
            if rank > count:
                raise FiniteSampleCalibrationError(
                    "Exact %.1f%% interval requires at least %d calibration groups; found %d"
                    % (100.0 * level, minimum_calibration_groups(level), count)
                )
            quantiles[level] = float(ordered[rank - 1])
            ranks[level] = rank
        signature_payload = [
            {
                "group": str(group),
                "y_true": float(target),
                "y_prediction": float(predicted),
            }
            for group, target, predicted in zip(groups, true, prediction)
        ]
        self.quantiles_eV_ = quantiles
        self.quantile_ranks_ = ranks
        self.calibration_group_scores_ = score_frame
        self.n_calibration_records_ = int(len(true))
        self.n_calibration_groups_ = int(count)
        self.calibration_signature_sha256_ = hashlib.sha256(
            _canonical_json(signature_payload).encode("utf-8")
        ).hexdigest()
        return self

    def quantile(self, confidence_level: float) -> float:
        self._require_fitted()
        level = float(confidence_level)
        if level not in self.quantiles_eV_:
            raise KeyError("Confidence level %.12g was not fitted" % level)
        return float(self.quantiles_eV_[level])

    def interval(
        self,
        prediction: Sequence[float],
        confidence_level: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        point = _as_finite_vector(prediction, "interval predictions")
        quantile = self.quantile(confidence_level)
        lower = point - quantile
        upper = point + quantile
        validate_interval_order(lower, point, upper)
        return lower, upper

    def interval_frame(self, prediction: Sequence[float]) -> pd.DataFrame:
        point = _as_finite_vector(prediction, "interval predictions")
        output = pd.DataFrame({"prediction_eV": point})
        for level in self.confidence_levels:
            lower, upper = self.interval(point, level)
            label = str(int(round(level * 100.0)))
            output["conformal_%s_lower_eV" % label] = lower
            output["conformal_%s_upper_eV" % label] = upper
            output["conformal_%s_width_eV" % label] = upper - lower
        return output

    def evaluate(
        self,
        y_true: Sequence[float],
        prediction: Sequence[float],
        groups: Sequence[Any],
        confidence_level: float,
    ) -> Dict[str, Any]:
        point = _as_finite_vector(prediction, "evaluation predictions")
        lower, upper = self.interval(point, confidence_level)
        summary = coverage_summary(y_true, lower, upper, groups)
        summary.update(
            {
                "confidence_level": float(confidence_level),
                "target_coverage": float(confidence_level),
                "group_max_quantile_eV": self.quantile(confidence_level),
                "quantile_rank": int(self.quantile_ranks_[float(confidence_level)]),
                "calibration_group_count": int(self.n_calibration_groups_),
                "coverage_primary": "group_simultaneous_coverage",
                "record_coverage_role": "secondary",
            }
        )
        return summary

    def to_dict(self) -> Dict[str, Any]:
        self._require_fitted()
        return {
            "class": self.__class__.__name__,
            "serialization_version": self.SERIALIZATION_VERSION,
            "confidence_levels": list(self.confidence_levels),
            "quantiles_eV": {str(level): value for level, value in self.quantiles_eV_.items()},
            "quantile_ranks": {str(level): value for level, value in self.quantile_ranks_.items()},
            "n_calibration_records": int(self.n_calibration_records_),
            "n_calibration_groups": int(self.n_calibration_groups_),
            "nonconformity_score": "maximum absolute residual within each calibration group",
            "quantile_rule": "k=ceil((m+1)*coverage), sorted_group_scores[k-1], no interpolation",
            "calibration_signature_sha256": self.calibration_signature_sha256_,
        }

    def save(self, path: Path) -> None:
        self._require_fitted()
        joblib.dump(self, str(Path(path)))

    @classmethod
    def load(cls, path: Path) -> "GroupMaxSplitConformal":
        loaded = joblib.load(str(Path(path)))
        if not isinstance(loaded, cls):
            raise TypeError("Serialized object is not %s" % cls.__name__)
        loaded._require_fitted()
        return loaded


def ensemble_disagreement(estimator_or_pipeline: Any, frame_or_matrix: Any) -> np.ndarray:
    """Return tree-to-tree prediction spread, explicitly not a calibrated PI."""

    if hasattr(estimator_or_pipeline, "encoded_matrix") and hasattr(estimator_or_pipeline, "estimator"):
        estimator = estimator_or_pipeline.estimator
        matrix = estimator_or_pipeline.encoded_matrix(frame_or_matrix)
    else:
        estimator = estimator_or_pipeline
        matrix = np.asarray(frame_or_matrix, dtype=float)
    if not hasattr(estimator, "estimators_"):
        raise TypeError("Estimator does not expose an ensemble of fitted base estimators")
    base_estimators = np.asarray(estimator.estimators_, dtype=object).ravel().tolist()
    if len(base_estimators) < 2 or any(not hasattr(item, "predict") for item in base_estimators):
        raise TypeError("At least two fitted predictive base estimators are required")
    predictions = np.vstack(
        [np.asarray(item.predict(matrix), dtype=float).reshape(-1) for item in base_estimators]
    )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Base-estimator predictions contain non-finite values")
    return np.std(predictions, axis=0, ddof=0)


def validate_four_block_feature_inputs(
    feature_columns: Sequence[str],
    feature_blocks: Mapping[str, Sequence[str]],
    leakage_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate an exact four-block, no-leakage input allowlist."""

    columns = [str(value) for value in feature_columns]
    if not columns:
        raise LeakageFeatureError("Feature input list is empty")
    if len(columns) != len(set(columns)):
        raise LeakageFeatureError("Feature input list contains duplicate columns")
    if set(feature_blocks) != set(FOUR_FEATURE_BLOCKS):
        raise LeakageFeatureError("Feature blocks must be exactly %s" % (FOUR_FEATURE_BLOCKS,))
    flattened: List[str] = []
    for block in FOUR_FEATURE_BLOCKS:
        block_columns = [str(value) for value in feature_blocks[block]]
        if not block_columns:
            raise LeakageFeatureError("Feature block %s is empty" % block)
        flattened.extend(block_columns)
    if len(flattened) != len(set(flattened)):
        raise LeakageFeatureError("A feature is assigned to more than one block")
    if set(flattened) != set(columns):
        raise LeakageFeatureError("Feature columns and four-block membership differ")

    normalised_patterns = [_normalise_name(value) for value in DEFAULT_FORBIDDEN_INPUT_PATTERNS]
    normalised_metadata = {_normalise_name(value) for value in DEFAULT_FORBIDDEN_METADATA_INPUTS}
    violations = []
    for column in columns:
        normalised = _normalise_name(column)
        padded = "_%s_" % normalised
        metadata_match = any("_%s_" % item in padded for item in normalised_metadata)
        if metadata_match or any(pattern in normalised for pattern in normalised_patterns):
            violations.append(column)
    if violations:
        raise LeakageFeatureError("Forbidden applicability-input fields: %s" % sorted(violations))
    if leakage_config is not None:
        assert_no_leakage_features(columns, leakage_config, allowed_feature_columns=columns)


assert_uncertainty_input_columns = validate_four_block_feature_inputs


def _weighted_step_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Quantile probability must be strictly between zero and one")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / float(np.sum(sorted_weights))
    position = int(np.searchsorted(cumulative, probability, side="left"))
    return float(sorted_values[min(position, len(sorted_values) - 1)])


class FourBlockPCADistanceOOD(object):
    """Train-fold-only, four-block-balanced PCA fifth-group distance score."""

    SERIALIZATION_VERSION = 1

    def __init__(
        self,
        feature_blocks: Mapping[str, Sequence[str]],
        variance_retained: float = 0.95,
        neighbor_rank: int = 5,
        variance_tolerance: float = 1e-12,
        leakage_config: Optional[Dict[str, Any]] = None,
    ):
        self.feature_blocks = {block: list(feature_blocks.get(block, [])) for block in FOUR_FEATURE_BLOCKS}
        self.feature_columns = [
            column for block in FOUR_FEATURE_BLOCKS for column in self.feature_blocks[block]
        ]
        validate_four_block_feature_inputs(self.feature_columns, self.feature_blocks, leakage_config)
        self.variance_retained = float(variance_retained)
        if not 0.0 < self.variance_retained <= 1.0:
            raise ValueError("variance_retained must be in (0, 1]")
        self.neighbor_rank = int(neighbor_rank)
        if self.neighbor_rank < 1:
            raise ValueError("neighbor_rank must be positive")
        self.variance_tolerance = float(variance_tolerance)
        self.leakage_config = leakage_config
        self.featurizer_: Optional[MedianMissingFeaturizer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self.retained_encoded_indices_: Optional[np.ndarray] = None
        self.retained_encoded_columns_: List[str] = []
        self.constant_encoded_columns_: List[str] = []
        self.encoded_column_blocks_: List[str] = []
        self.encoded_block_counts_: Dict[str, int] = {}
        self.block_balance_factors_: Dict[str, float] = {}
        self.training_embedding_: Optional[np.ndarray] = None
        self.training_groups_: Optional[np.ndarray] = None
        self.training_group_indices_: Dict[str, np.ndarray] = {}
        self.training_reference_distances_: Optional[np.ndarray] = None
        self.training_reference_weights_: Optional[np.ndarray] = None
        self.training_reference_nearest_groups_: List[List[str]] = []
        self.reference_group_max_distances_: Dict[str, float] = {}
        self.warning_threshold_95_: Optional[float] = None
        self.warning_threshold_99_: Optional[float] = None
        self.graph_warning_threshold_95_: Optional[float] = None
        self.graph_warning_threshold_99_: Optional[float] = None
        self.training_signature_sha256_: Optional[str] = None

    def _require_fitted(self) -> None:
        if (
            self.featurizer_ is None
            or self.scaler_ is None
            or self.pca_ is None
            or self.training_embedding_ is None
            or self.training_groups_ is None
            or self.training_reference_distances_ is None
        ):
            raise RuntimeError("FourBlockPCADistanceOOD is not fitted")

    @staticmethod
    def _source_column(encoded_column: str) -> str:
        return encoded_column[len("missing__") :] if encoded_column.startswith("missing__") else encoded_column

    def _balanced_matrix_fit(self, training_frame: pd.DataFrame) -> np.ndarray:
        self.featurizer_ = MedianMissingFeaturizer(self.feature_columns)
        encoded = self.featurizer_.fit_transform(training_frame)
        variances = np.var(encoded, axis=0)
        keep = np.isfinite(variances) & (variances > self.variance_tolerance)
        if not np.any(keep):
            raise OODReferenceError("All encoded applicability features are constant")
        self.retained_encoded_indices_ = np.where(keep)[0]
        self.retained_encoded_columns_ = [
            self.featurizer_.output_columns[index] for index in self.retained_encoded_indices_
        ]
        self.constant_encoded_columns_ = [
            self.featurizer_.output_columns[index] for index in np.where(~keep)[0]
        ]
        block_by_source = {
            column: block for block, block_columns in self.feature_blocks.items() for column in block_columns
        }
        self.encoded_column_blocks_ = [
            block_by_source[self._source_column(column)] for column in self.retained_encoded_columns_
        ]
        self.encoded_block_counts_ = {
            block: int(sum(value == block for value in self.encoded_column_blocks_))
            for block in FOUR_FEATURE_BLOCKS
        }
        empty = [block for block, count in self.encoded_block_counts_.items() if count == 0]
        if empty:
            raise OODReferenceError("No non-constant encoded columns remain for blocks: %s" % empty)
        self.block_balance_factors_ = {
            block: 1.0 / math.sqrt(float(count))
            for block, count in self.encoded_block_counts_.items()
        }
        selected = encoded[:, self.retained_encoded_indices_]
        self.scaler_ = StandardScaler().fit(selected)
        standardised = self.scaler_.transform(selected)
        factors = np.asarray(
            [self.block_balance_factors_[block] for block in self.encoded_column_blocks_], dtype=float
        )
        return standardised * factors

    def _balanced_matrix_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        encoded = self.featurizer_.transform(frame)
        selected = encoded[:, self.retained_encoded_indices_]
        standardised = self.scaler_.transform(selected)
        factors = np.asarray(
            [self.block_balance_factors_[block] for block in self.encoded_column_blocks_], dtype=float
        )
        return standardised * factors

    def _group_distances(
        self,
        query_embedding: np.ndarray,
        excluded_group: Optional[str] = None,
    ) -> List[Tuple[float, str]]:
        distances: List[Tuple[float, str]] = []
        for group in sorted(self.training_group_indices_):
            if excluded_group is not None and group == excluded_group:
                continue
            positions = self.training_group_indices_[group]
            difference = self.training_embedding_[positions] - query_embedding.reshape(1, -1)
            distance = float(np.min(np.sqrt(np.sum(difference * difference, axis=1))))
            distances.append((distance, group))
        distances.sort(key=lambda item: (item[0], item[1]))
        if len(distances) < self.neighbor_rank:
            raise OODReferenceError(
                "Only %d distinct eligible training groups; d%d requires at least %d"
                % (len(distances), self.neighbor_rank, self.neighbor_rank)
            )
        return distances

    def fit(self, training_frame: pd.DataFrame, training_groups: Sequence[Any]) -> "FourBlockPCADistanceOOD":
        """Fit every OOD transformation and reference using training rows only."""

        groups = _as_groups(training_groups, len(training_frame))
        unique_groups = sorted(set(groups))
        if len(unique_groups) < self.neighbor_rank + 1:
            raise OODReferenceError(
                "Training reference needs at least %d groups so own-group exclusion leaves d%d"
                % (self.neighbor_rank + 1, self.neighbor_rank)
            )
        balanced = self._balanced_matrix_fit(training_frame)
        if min(balanced.shape) < 2:
            raise OODReferenceError("PCA applicability reference needs at least two rows and columns")
        n_components: Any = self.variance_retained
        if self.variance_retained == 1.0:
            n_components = min(balanced.shape)
        self.pca_ = PCA(n_components=n_components, svd_solver="full")
        self.training_embedding_ = self.pca_.fit_transform(balanced)
        self.training_groups_ = groups
        self.training_group_indices_ = {
            group: np.where(groups == group)[0] for group in unique_groups
        }

        reference_distances = []
        nearest_groups = []
        for position, group in enumerate(groups):
            distances = self._group_distances(self.training_embedding_[position], excluded_group=str(group))
            reference_distances.append(distances[self.neighbor_rank - 1][0])
            nearest_groups.append([item[1] for item in distances[: self.neighbor_rank]])
        self.training_reference_distances_ = np.asarray(reference_distances, dtype=float)
        self.training_reference_nearest_groups_ = nearest_groups
        group_sizes = {group: int(np.sum(groups == group)) for group in unique_groups}
        self.training_reference_weights_ = np.asarray(
            [1.0 / (len(unique_groups) * group_sizes[str(group)]) for group in groups], dtype=float
        )
        self.reference_group_max_distances_ = {
            group: float(np.max(self.training_reference_distances_[groups == group]))
            for group in unique_groups
        }
        self.warning_threshold_95_ = _weighted_step_quantile(
            self.training_reference_distances_, self.training_reference_weights_, 0.95
        )
        self.warning_threshold_99_ = _weighted_step_quantile(
            self.training_reference_distances_, self.training_reference_weights_, 0.99
        )
        graph_reference = np.asarray(list(self.reference_group_max_distances_.values()), dtype=float)
        equal_group_weights = np.full(len(graph_reference), 1.0 / len(graph_reference), dtype=float)
        self.graph_warning_threshold_95_ = _weighted_step_quantile(graph_reference, equal_group_weights, 0.95)
        self.graph_warning_threshold_99_ = _weighted_step_quantile(graph_reference, equal_group_weights, 0.99)
        signature = hashlib.sha256()
        signature.update(np.ascontiguousarray(balanced, dtype=np.float64).tobytes())
        signature.update(_canonical_json([str(value) for value in groups]).encode("utf-8"))
        signature.update(_canonical_json(self.feature_blocks).encode("utf-8"))
        self.training_signature_sha256_ = signature.hexdigest()
        return self

    def _reference_percentile(self, distance: float) -> float:
        return float(
            np.sum(
                self.training_reference_weights_[
                    self.training_reference_distances_ <= float(distance) + 1e-15
                ]
            )
        )

    def score(
        self,
        query_frame: pd.DataFrame,
        query_groups: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        """Score frozen-reference queries; queries never update the OOD ruler."""

        self._require_fitted()
        if query_groups is None:
            groups: List[Optional[str]] = [None] * len(query_frame)
        else:
            groups = [str(value) for value in _as_groups(query_groups, len(query_frame))]
        embedding = self.transform_embedding(query_frame)
        rows = []
        training_group_set = set(self.training_group_indices_)
        for position, query_group in enumerate(groups):
            excluded = query_group if query_group in training_group_set else None
            distances = self._group_distances(embedding[position], excluded_group=excluded)
            distance = float(distances[self.neighbor_rank - 1][0])
            percentile = min(1.0, max(0.0, self._reference_percentile(distance)))
            rows.append(
                {
                    "ood_distance_d%d" % self.neighbor_rank: distance,
                    "ood_percentile_group_equal": percentile,
                    "applicability_score": 1.0 - percentile,
                    "ood_warning_95": int(percentile >= 0.95),
                    "ood_warning_99": int(percentile >= 0.99),
                    "nearest_training_groups": ";".join(
                        item[1] for item in distances[: self.neighbor_rank]
                    ),
                    "nearest_training_group_count": self.neighbor_rank,
                    "query_group_excluded_from_reference": int(excluded is not None),
                }
            )
        return pd.DataFrame(rows)

    def transform_embedding(self, frame: pd.DataFrame) -> np.ndarray:
        """Project rows with the already-frozen training-fold transformations."""

        self._require_fitted()
        embedding = np.asarray(
            self.pca_.transform(self._balanced_matrix_transform(frame)), dtype=float
        )
        if embedding.ndim != 2 or len(embedding) != len(frame):
            raise RuntimeError("Frozen OOD embedding has an unexpected shape")
        if not np.all(np.isfinite(embedding)):
            raise ValueError("Frozen OOD embedding contains non-finite values")
        return embedding

    def score_graph(
        self,
        query_frame: pd.DataFrame,
        query_groups: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """Score a graph by its worst edge against per-training-group maxima."""

        edge_scores = self.score(query_frame, query_groups=query_groups)
        distance_column = "ood_distance_d%d" % self.neighbor_rank
        maximum = float(edge_scores[distance_column].max())
        reference = np.asarray(list(self.reference_group_max_distances_.values()), dtype=float)
        percentile = float(np.mean(reference <= maximum + 1e-15))
        return {
            "graph_ood_distance_d%d_max" % self.neighbor_rank: maximum,
            "graph_ood_percentile_group_equal": percentile,
            "graph_applicability_score": 1.0 - percentile,
            "graph_ood_warning_95": int(percentile >= 0.95),
            "graph_ood_warning_99": int(percentile >= 0.99),
            "graph_edge_count": int(len(edge_scores)),
            "graph_reference_unit": "maximum training-row d%d within each equally weighted training group"
            % self.neighbor_rank,
        }

    def to_dict(self) -> Dict[str, Any]:
        self._require_fitted()
        return {
            "class": self.__class__.__name__,
            "serialization_version": self.SERIALIZATION_VERSION,
            "fit_scope": "training fold only; labels and query/test rows are not accepted by fit",
            "feature_blocks": self.feature_blocks,
            "raw_feature_count": len(self.feature_columns),
            "encoded_feature_count": len(self.featurizer_.output_columns),
            "retained_encoded_feature_count": len(self.retained_encoded_columns_),
            "constant_encoded_feature_count": len(self.constant_encoded_columns_),
            "encoded_block_counts": self.encoded_block_counts_,
            "block_balance_factors": self.block_balance_factors_,
            "block_balance_rule": "standardize training fold then divide each retained column by sqrt(block width)",
            "pca_variance_target": self.variance_retained,
            "pca_component_count": int(self.pca_.n_components_),
            "pca_explained_variance_ratio_sum": float(np.sum(self.pca_.explained_variance_ratio_)),
            "distance": "Euclidean PCA distance to fifth-nearest distinct training group (group distance=min row distance)",
            "neighbor_rank": self.neighbor_rank,
            "training_row_count": int(len(self.training_groups_)),
            "training_group_count": int(len(self.training_group_indices_)),
            "edge_reference_ecdf": "each training group has equal total weight",
            "graph_reference_ecdf": "one within-group maximum per training group",
            "warning_threshold_95": float(self.warning_threshold_95_),
            "warning_threshold_99": float(self.warning_threshold_99_),
            "graph_warning_threshold_95": float(self.graph_warning_threshold_95_),
            "graph_warning_threshold_99": float(self.graph_warning_threshold_99_),
            "training_signature_sha256": self.training_signature_sha256_,
        }

    def save(self, path: Path) -> None:
        self._require_fitted()
        joblib.dump(self, str(Path(path)))

    @classmethod
    def load(cls, path: Path) -> "FourBlockPCADistanceOOD":
        loaded = joblib.load(str(Path(path)))
        if not isinstance(loaded, cls):
            raise TypeError("Serialized object is not %s" % cls.__name__)
        loaded._require_fitted()
        return loaded
