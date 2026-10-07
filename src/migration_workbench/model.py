"""Inference through the frozen paper model and its original feature functions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from .paths import MODEL_DIR, PROJECT, ensure_science_path, require_inputs


PAPER_MODEL_SHA256 = "7f2cdf24e8baf50094c9eee85d5aabef357e1695572278cd624b47f0d1d88ba3"


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


class FrozenBarrierModel:
    def __init__(self):
        require_inputs()
        ensure_science_path()
        import joblib
        from revision_core.element_properties import NEUTRAL_PROPERTY_COLUMNS

        self.pipeline_path = MODEL_DIR / "selected_barrier_model_pipeline.joblib"
        actual = sha256_file(self.pipeline_path)
        if actual != PAPER_MODEL_SHA256:
            raise ValueError("论文冻结模型哈希不一致：" + actual)
        self.pipeline = joblib.load(str(self.pipeline_path))
        self.ood = joblib.load(
            str(MODEL_DIR / "revision_uncertainty_network" / "full_strict_X_only_diversity_embedding.joblib")
        )
        self.config = json.loads((PROJECT / "config" / "revision_model_config.yaml").read_text(encoding="utf-8"))
        manifest = json.loads((MODEL_DIR / "element_properties_manifest.json").read_text(encoding="utf-8"))
        self.all_symbols = list(manifest["used_elements"])
        self.property_lookup = {}
        with (MODEL_DIR / "element_properties_snapshot.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                self.property_lookup[row["symbol"]] = {
                    key: float(row[key]) if row[key] else math.nan for key in NEUTRAL_PROPERTY_COLUMNS
                }
        if set(self.all_symbols) != set(self.property_lookup):
            raise ValueError("元素属性快照与训练元素清单不一致")
        self.model_version = actual

    def feature_row(self, structure, migrating_species, path):
        ensure_science_path()
        from revision_core.barrier_features import _composition_block, edge_feature_blocks

        elements = [str(value) for value in structure["elements"]]
        unknown = sorted(set(elements + [migrating_species]) - set(self.all_symbols))
        if unknown:
            raise ValueError("模型训练元素表中没有这些元素，不能推断：" + ", ".join(unknown))
        moving_index = int(path["moving_atom_index"])
        if not 0 <= moving_index < len(elements) or elements[moving_index] != migrating_species:
            raise ValueError("路径 moving_atom_index 必须指向初始结构中的迁移离子")
        start = path["start_frac_coords"]
        end = path["end_frac_coords"]
        shift = path["image_shift"]
        composition = _composition_block(
            elements, migrating_species, self.all_symbols, self.property_lookup, structure["lattice_mat"]
        )
        blocks = edge_feature_blocks(
            structure, migrating_species, start, end, shift,
            moving_index, self.all_symbols, self.property_lookup, self.config,
        )
        row = dict(composition)
        for values in blocks.values():
            row.update(values)
        missing = sorted(set(self.pipeline.feature_columns) - set(row))
        if missing:
            raise RuntimeError("描述符缺失：" + ", ".join(missing[:8]))
        return row

    def predict(self, structure, migrating_species, paths, formula_species_group):
        import pandas as pd
        from revision_core.uncertainty_calibration import ensemble_disagreement

        rows = [self.feature_row(structure, migrating_species, path) for path in paths]
        frame = pd.DataFrame(rows)
        predictions = self.pipeline.predict(frame)
        spread = ensemble_disagreement(self.pipeline, frame)
        groups = [formula_species_group] * len(frame)
        ood = self.ood.score(frame, query_groups=groups).to_dict("records")
        graph_ood = self.ood.score_graph(frame, query_groups=groups)
        result = []
        for index, path in enumerate(paths):
            row = dict(path)
            row["predicted_barrier_eV"] = float(predictions[index])
            row["ensemble_disagreement_eV"] = float(spread[index])
            row.update(ood[index])
            row["model_sha256"] = self.model_version
            row["prediction_role"] = "full_model_prospective"
            row["uncertainty_note"] = "树间分歧不是校准预测区间；新边覆盖率未验证"
            result.append(row)
        return result, graph_ood
