import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile


def load_wps_module():
    dummy_serial = types.ModuleType("serial")
    dummy_serial.Serial = object
    dummy_wpshelper = types.ModuleType("wpshelper")
    for name in [
        "wps_open",
        "wps_configure",
        "wps_start_record",
        "wps_stop_record",
        "wps_analyze_capture",
        "wps_save_capture",
        "wps_update_matter_keys",
        "wps_close",
    ]:
        setattr(dummy_wpshelper, name, lambda *args, **kwargs: None)

    sys.modules.setdefault("serial", dummy_serial)
    sys.modules.setdefault("wpshelper", dummy_wpshelper)

    module_path = Path(__file__).resolve().parents[1] / "wps_matter_key_update_from_log.py"
    spec = importlib.util.spec_from_file_location("wps_matter_key_update_from_log", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load wps_matter_key_update_from_log module")
    sys.modules["wps_matter_key_update_from_log"] = module
    spec.loader.exec_module(module)
    return module


class TestWpsMatterKeyUpdateFromLog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_wps_module()

    def test_process_line_extracts_key(self):
        line = "[00:00:28.851] <inf> chip: AES_CCM_encrypt key = 0x48e8b550bfc8523e"
        result = self.module.process_line(line)
        self.assertEqual(result, {"key": "48e8b550bfc8523e"})

    def test_process_line_extracts_source_node_id(self):
        line = "[00:00:28.852] <inf> chip: AES_CCM_decrypt nonce = 0x000102030405060708090a0b0c0d0e0f"
        result = self.module.process_line(line)
        self.assertEqual(result, {"source_node_id": "0f0e0d0c0b0a0908"})

    def test_process_line_ignores_unmatched(self):
        self.assertEqual(self.module.process_line("no match here"), {})

    def test_update_file_appends_json(self):
        mapping = {"0xabc": ["0x123"]}
        with NamedTemporaryFile(mode="r+", delete=False) as handle:
            path = handle.name
        self.module.update_file(path, mapping)
        with open(path, "r", encoding="utf-8") as handle:
            line = handle.readline().strip()
        _, json_part = line.split(" ", 1)
        self.assertEqual(json.loads(json_part), mapping)


if __name__ == "__main__":
    unittest.main()
