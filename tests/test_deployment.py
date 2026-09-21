"""The deployed demo must ship its evaluated model, map and readable analysis."""
import json
import os
import unittest
import warnings
from unittest.mock import patch

from sklearn.exceptions import InconsistentVersionWarning
from web.app import app


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_and_prediction_with_shipped_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            self.assertEqual(self.client.get("/healthz").json, {"status": "ok"})
            sites = self.client.get("/api/sites").json
            self.assertEqual(len(sites), 60)
            result = self.client.post("/api/predict", json={"features": sites[0]["features"]})
        self.assertEqual(result.status_code, 200)
        self.assertGreaterEqual(result.json["probability"], 0)
        self.assertLessEqual(result.json["probability"], 1)
        self.assertEqual(result.json["warnings"], [])

    def test_model_evidence_and_map_are_packaged_and_match(self):
        metrics = self.client.get("/api/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertTrue(metrics.json)
        grid = self.client.get("/api/susceptibility")
        self.assertEqual(grid.status_code, 200)
        self.assertFalse(grid.json["stale"], grid.json.get("stale_reason"))
        self.assertEqual(self.client.get("/api/boundary").status_code, 200)

    def test_page_notebook_and_static_files_are_packaged(self):
        for path in ["/", "/notebook", "/static/app.css", "/static/theme.css",
                     "/static/favicon.svg", "/static/walk_data.js"]:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                response.close()

    def test_portfolio_link_can_be_set_without_rebuilding(self):
        url = "https://portfolio.example/research.html#tos"
        with patch.dict(os.environ, {"PORTFOLIO_TOS_URL": url}):
            response = self.client.get("/deployment-config.js")
            self.assertIn(json.dumps(url), response.text)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        with patch.dict(os.environ, {"PORTFOLIO_TOS_URL": ""}):
            self.assertIn("window.RANGAMATI_CONFIG, {}", self.client.get("/deployment-config.js").text)

    def test_outside_study_area_never_calls_elevation_service(self):
        with patch("web.app.terrain.sample_point") as sample:
            response = self.client.post("/api/sample", json={"lat": 0, "lon": 0})
        self.assertEqual(response.status_code, 422)
        self.assertTrue(response.json["refused"])
        sample.assert_not_called()


if __name__ == "__main__":
    unittest.main()
