import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "wps_parquet_llm_summary_cli.py"
    spec = importlib.util.spec_from_file_location("wps_parquet_llm_summary_cli", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load wps_parquet_llm_summary_cli module")
    spec.loader.exec_module(module)
    return module


class TestParquetLlmSummaryCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_build_summary_payloads_contains_expected_sections(self):
        frame = pd.DataFrame(
            {
                "ts": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:01Z", "2025-01-01T00:00:02Z"],
                "advertiser_addr": ["aa", "aa", "bb"],
                "rssi": [-60, -61, -59],
            }
        )
        payloads = self.module.build_summary_payloads(
            frame, table_name="ble_adv_events", source_path="sample.parquet"
        )

        self.assertIn("capture_summary", payloads)
        self.assertIn("table_catalog", payloads)
        self.assertEqual(payloads["capture_summary"]["row_count"], 3)
        self.assertEqual(payloads["capture_summary"]["time_column"], "ts")
        self.assertIn("advertiser_addr", payloads["capture_summary"]["top_entities"])

        table = payloads["table_catalog"]["tables"][0]
        self.assertEqual(table["table"], "ble_adv_events")
        self.assertEqual(len(table["columns"]), 3)

    def test_main_writes_files_when_output_dir_is_provided(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            parquet_path = tmp / "input.parquet"
            out_dir = tmp / "out"
            pd.DataFrame({"ts": ["2025-01-01T00:00:00Z"], "id": ["x"], "value": [1]}).to_parquet(parquet_path)

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "wps_parquet_llm_summary_cli.py",
                    str(parquet_path),
                    "--output-dir",
                    str(out_dir),
                    "--no-human-readable",
                ]
                self.module.main()
            finally:
                sys.argv = old_argv

            self.assertTrue((out_dir / "capture_summary.json").exists())
            self.assertTrue((out_dir / "table_catalog.json").exists())

    def test_human_summary_includes_object_composition(self):
        frame = pd.DataFrame(
            {
                "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:01Z"],
                "payload": [{"a": 1, "b": 2}, {"a": 3}],
                "tags": [["x", "y"], ["z"]],
            }
        )

        payloads = self.module.build_summary_payloads(
            frame, table_name="test", source_path="sample.parquet"
        )
        text = self.module._human_readable(payloads["capture_summary"], payloads["table_catalog"])
        self.assertIn("object composition", text)
        self.assertIn("dict keys", text)
        self.assertIn("element types", text)


if __name__ == "__main__":
    unittest.main()
