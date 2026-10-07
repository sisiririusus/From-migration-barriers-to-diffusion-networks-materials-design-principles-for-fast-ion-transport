"""Integration checks for scientific parity and real user input paths."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from migration_workbench.examples import catalog, load_case
from migration_workbench.inputs import load_project, load_graph_csv
from migration_workbench.model import FrozenBarrierModel
from migration_workbench.network import periodic_analysis
from migration_workbench.paths import PROJECT, ROOT
from migration_workbench.runner import analyze_project


class WorkbenchIntegrationTest(unittest.TestCase):
    def test_paper_case_inventory_and_exact_na23(self):
        cases = catalog()
        self.assertEqual(len(cases), 8)
        base = [case for case in cases if case["family"] == "base_periodic"]
        comparison = [case for case in cases if case["family"] == "composition_comparison"]
        self.assertEqual((sum(case["n_nodes"] for case in base), sum(case["n_edges"] for case in base)), (64, 713))
        self.assertEqual(sum(case["n_edges"] for case in comparison), 139)
        actual = load_case("Na23Se8Cl8_Na", recompute=True)
        self.assertTrue(actual["computed_now"])
        self.assertEqual(actual["readout"]["winding_vector_rank"], 3)
        self.assertEqual(len(actual["readout"]["critical_periodic_edges"]), 6)
        self.assertAlmostEqual(actual["readout"]["Eperc_a"], 1.1829268333333336, places=10)
        na10 = load_case("Na10Mn2Al2P9O24_Na")
        self.assertEqual(len(na10["recommendations"]), 90)
        recommended_ids = {row["edge_uid"] for row in na10["recommendations"]}
        self.assertIn("edge_e1e29378acca730eb28240", recommended_ids)
        self.assertNotIn("edge_01523e5585583c263853a1", recommended_ids)
        self.assertTrue(all(row["is_labeled"] == "0" for row in na10["recommendations"]))
        ca = load_case("Ca7Cu16S16_Ca", recompute=True)
        self.assertEqual(ca["readout"]["winding_vector_rank"], 2)
        self.assertIsNone(ca["readout"]["Eperc_c"])

    def test_frozen_model_reproduces_713_candidate_predictions(self):
        model = FrozenBarrierModel()
        frame = pd.read_csv(str(PROJECT / "tables/revision_source_data/revision_periodic_candidate_local_features.csv"))
        saved = pd.read_csv(str(PROJECT / "new_data/revision_models/revision_periodic_candidate_predictions.csv"))
        predicted = pd.DataFrame({"edge_uid": frame["edge_uid"].astype(str), "prediction": model.pipeline.predict(frame)})
        compared = predicted.merge(saved[["edge_uid", "predicted_barrier_eV"]], on="edge_uid", validate="one_to_one")
        self.assertEqual(len(compared), 713)
        self.assertLess(float(np.max(np.abs(compared["prediction"] - compared["predicted_barrier_eV"]))), 1e-8)

    def test_user_path_invokes_model_and_ood(self):
        project = load_project(ROOT / "examples/user_path_project.json")
        result = analyze_project(project)
        edge = result["edges"][0]
        self.assertAlmostEqual(edge["predicted_barrier_eV"], 0.52805, places=5)
        self.assertIsInstance(edge["applicability_score"], float)
        self.assertEqual(result["formula_species_group"], "NaV4P4O20|Na")
        self.assertEqual(edge["query_group_excluded_from_reference"], 1)
        self.assertEqual(edge["prediction_role"], "full_model_prospective")
        self.assertEqual(result["readout"]["winding_vector_rank"], 0)
        self.assertIsNone(result["readout"]["Eperc_a"])

    def test_user_periodic_graph_and_occupied_target_rejection(self):
        project = load_project(ROOT / "examples/user_network_graph.json")
        result = analyze_project(project)
        self.assertEqual(result["readout"]["winding_vector_rank"], 2)
        self.assertEqual(result["readout"]["Eperc_a"], 0.2)
        self.assertEqual(result["readout"]["Eperc_b"], 0.3)
        self.assertIsNone(result["readout"]["Eperc_c"])
        path_example = json.loads((ROOT / "examples/user_path_project.json").read_text(encoding="utf-8"))
        path_example["nodes"][1]["frac_coords"] = path_example["structure"]["coords"][1]
        with self.assertRaisesRegex(ValueError, "终点被同种离子占据|起点"):
            # V is not a migrating Na ion here, so the imported site remains
            # geometrically legal. Make it occupied by a second Na instead.
            path_example["structure"]["elements"][1] = "Na"
            from migration_workbench.inputs import validate_nodes, _validate_edges, validate_structure
            structure = validate_structure(path_example["structure"], "Na")
            nodes = validate_nodes(path_example["nodes"])
            _validate_edges(path_example["edges"], nodes, lattice=structure["lattice_mat"], structure=structure, species="Na", predict=True)

    def test_csv_import_rejects_cross_material_graph_mix(self):
        with tempfile.TemporaryDirectory() as folder:
            nodes = Path(folder) / "nodes.csv"
            edges = Path(folder) / "edges.csv"
            nodes.write_text(
                "graph_id,node_id,frac_coords\nA,n0,\"[0,0,0]\"\nB,n1,\"[0,0,0]\"\n",
                encoding="utf-8",
            )
            edges.write_text(
                "graph_id,edge_uid,node_i,node_j,image_shift_a,image_shift_b,image_shift_c,barrier_eV\n"
                "A,e0,n0,n0,1,0,0,0.2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "多个 graph_id"):
                load_graph_csv(nodes, edges)


if __name__ == "__main__":
    unittest.main()
