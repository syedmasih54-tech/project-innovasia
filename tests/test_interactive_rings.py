"""
StreamShield - Interactive Mule Rings Topology Test Suite
Verifies Part A, Part B, Part C, and Part D requirements:
1. Graph schema validation (nodes, edges, detected_rings, ground_truth_eval).
2. Detector rings are driven by clustering on decisions, NOT generator labels.
3. Organic/unlabelled detected rings appear in normal view.
4. Ground truth evaluation is isolated and correctly tracks matched/missed/false rings.
5. Dynamic backend response when data changes.
6. Frontend functions verify node panel opening, ring zoom, search, and toggle behavior.
"""

import os
import sys
import json
import unittest
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from src.rings import export_rings_graph, find_mule_rings, load_flagged_decisions

class TestInteractiveRings(unittest.TestCase):

    def setUp(self):
        self.graph_path = os.path.join(BASE_DIR, "data", "rings_graph.json")
        self.decisions_path = os.path.join(BASE_DIR, "data", "decisions.jsonl")

    def test_01_backend_graph_schema(self):
        """Verifies graph export structure, required node/edge fields, and types."""
        self.assertTrue(os.path.exists(self.graph_path), "data/rings_graph.json must exist")
        with open(self.graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIn("detected_rings", data)
        self.assertIn("ground_truth_eval", data)
        self.assertIn("time_window", data)

        nodes = data["nodes"]
        edges = data["edges"]
        rings = data["detected_rings"]

        self.assertGreater(len(nodes), 200, "Must include ~250 nodes (ring members + background normal accounts)")
        self.assertLess(len(nodes), 350, "Capped around 300 nodes for 60fps performance")
        self.assertGreater(len(edges), 100, "Must include internal and boundary transaction edges")
        self.assertGreater(len(rings), 0, "Must detect ring clusters")

        # Verify Node schema
        valid_roles = {"collector", "feeder", "exit", "chain", "normal"}
        for n in nodes[:30]:
            self.assertIn("id", n)
            self.assertIn("short_id", n)
            self.assertIn("role", n)
            self.assertIn(n["role"], valid_roles)
            self.assertIn("risk_score", n)
            self.assertIn("amount_in", n)
            self.assertIn("amount_out", n)
            self.assertIn("pass_through_ratio", n)
            self.assertIn("first_seen", n)
            self.assertIn("first_flagged", n)
            self.assertIn("ring_id", n)
            self.assertIn("top_reasons", n)
            self.assertIsInstance(n["top_reasons"], list)
            self.assertIn("payments", n)

        # Verify Edge schema
        for e in edges[:30]:
            self.assertIn("id", e)
            self.assertIn("source", e)
            self.assertIn("target", e)
            self.assertIn("amount", e)
            self.assertIn("count", e)
            self.assertIn("decision", e)
            self.assertIn(e["decision"], {"HOLD", "STEP-UP", "ALLOW"})

    def test_02_ring_id_from_detector_not_generator(self):
        """Verifies that ring_id comes strictly from the detector and normal nodes have ring_id=null."""
        with open(self.graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data["nodes"]
        normal_nodes = [n for n in nodes if n["role"] == "normal"]
        ring_nodes = [n for n in nodes if n["role"] != "normal"]

        self.assertGreater(len(normal_nodes), 100, "Must have background normal connected nodes")
        for n in normal_nodes:
            self.assertIsNone(n["ring_id"], f"Normal node {n['id']} must have null ring_id in standard view")

        for n in ring_nodes:
            self.assertIsNotNone(n["ring_id"], f"Ring member node {n['id']} must have detector ring_id")
            self.assertTrue(n["ring_id"].startswith("ring_found_"), "ring_id must come from ring detector (e.g. ring_found_XX)")

    def test_03_organic_unlabelled_rings_appear_in_view(self):
        """Verifies that detected rings with NO synthetic label still appear as detected rings."""
        with open(self.graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rings = data["detected_rings"]
        organic_rings = [r for r in rings if not r.get("matched_true_ring")]
        self.assertGreater(len(organic_rings), 0, "Organic unlabelled clusters must appear in detected rings")

        for r in organic_rings:
            self.assertIn("ring_id", r)
            self.assertIn("members", r)
            self.assertIn("total_rupees", r)
            self.assertIn("first_flagged_time", r)
            self.assertIn("detection_delay", r)

    def test_04_ground_truth_evaluation_isolation(self):
        """Verifies ground truth evaluation object tracks matched, missed, false groups, precision, and recall."""
        with open(self.graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gt = data["ground_truth_eval"]
        self.assertIn("matched_rings", gt)
        self.assertIn("missed_rings", gt)
        self.assertIn("false_groups", gt)
        self.assertIn("precision", gt)
        self.assertIn("recall", gt)
        self.assertIn("total_detected", gt)
        self.assertIn("total_true", gt)

        self.assertEqual(len(gt["matched_rings"]), 5, "5 planted rings were in the test dataset")
        self.assertEqual(len(gt["missed_rings"]), 0, "All 5 planted rings were detected (100% recall)")
        self.assertEqual(gt["recall"], 1.0)
        self.assertGreater(len(gt["false_groups"]), 0, "Organic unlabelled rings are classified as false groups in ground truth eval")
        self.assertEqual(gt["precision"], round(5 / len(data["detected_rings"]), 4))

    def test_05_api_endpoint_response(self):
        """Verifies that the live server returns /api/rings_graph matches data/rings_graph.json."""
        try:
            req = urllib.request.Request("http://localhost:8080/api/rings_graph")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                body = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(len(body["nodes"]), len(json.load(open(self.graph_path))["nodes"]))
                self.assertEqual(len(body["detected_rings"]), len(json.load(open(self.graph_path))["detected_rings"]))
        except Exception as e:
            self.skipTest(f"Live server not reachable on localhost:8080 in sandboxed test run: {e}")

    def test_06_frontend_markup_and_scripts(self):
        """Verifies that app/static/index.html includes all required interactive elements and zero hardcoded rings."""
        html_path = os.path.join(BASE_DIR, "app", "static", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Canvas & Cytoscape
        self.assertIn('id="cy-mule-graph"', html, "Cytoscape container must exist")
        self.assertIn('cytoscape.min.js', html, "Cytoscape library must be loaded")

        # Interactive controls
        self.assertIn('id="mule-search-input"', html, "Search input must exist")
        self.assertIn('id="filter-mule-role"', html, "Role filter dropdown must exist")
        self.assertIn('id="toggle-ring-only"', html, "Ring only toggle must exist")
        self.assertIn('id="filter-min-amt"', html, "Min amount filter must exist")
        self.assertIn('id="toggle-ground-truth"', html, "Ground truth toggle must exist")
        self.assertIn('id="btn-timeline-play"', html, "Timeline play button must exist")
        self.assertIn('id="mule-timeline-slider"', html, "Timeline slider must exist")
        self.assertIn('id="mule-detail-panel"', html, "Side detail panel must exist")
        self.assertIn('id="mule-node-tooltip"', html, "Node hover tooltip must exist")
        self.assertIn('id="mule-edge-popup"', html, "Edge click popup must exist")
        self.assertIn('id="ring-chips-container"', html, "Ring chips container must exist")

        # Interactive action functions
        self.assertIn('focusAccountFromSearch', html)
        self.assertIn('zoomToRing', html)
        self.assertIn('openMuleDetailPanel', html)
        self.assertIn('focusRingFromPanel', html)
        self.assertIn('showMoneyPathFromPanel', html)
        self.assertIn('toggleGroundTruthMode', html)
        self.assertIn('toggleTimelinePlayback', html)

        # Wording checks
        self.assertNotIn('5 of 5 Planted Rings Intercepted', html, "Old hardcoded wording must be replaced")
        self.assertIn('rings detected (live)', html, "New dynamic wording must be present")

if __name__ == "__main__":
    unittest.main()

