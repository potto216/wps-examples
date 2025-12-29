import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_pcapng_module():
    dummy_scapy = types.ModuleType("scapy")
    dummy_scapy_all = types.ModuleType("scapy.all")
    dummy_layers = types.ModuleType("scapy.layers")
    dummy_layers_bluetooth = types.ModuleType("scapy.layers.bluetooth")

    class DummyPacket:
        pass

    dummy_scapy_all.Packet = DummyPacket

    def dummy_rdpcap(_path):
        return []

    dummy_scapy_all.rdpcap = dummy_rdpcap

    class DummyLayer:
        pass

    for name in [
        "BTLE",
        "BTLEAdvertisingHdr",
        "BTLEAdvDirectInd",
        "BTLEAdvExtInd",
        "BTLEAdvInd",
        "BTLEAdvNonconnInd",
        "BTLEAdvScanInd",
        "BTLEConnectReq",
        "BTLEScanReq",
        "BTLEScanRsp",
    ]:
        setattr(dummy_layers_bluetooth, name, type(name, (DummyLayer,), {}))

    sys.modules.setdefault("scapy", dummy_scapy)
    sys.modules.setdefault("scapy.all", dummy_scapy_all)
    sys.modules.setdefault("scapy.layers", dummy_layers)
    sys.modules.setdefault("scapy.layers.bluetooth", dummy_layers_bluetooth)

    module_path = Path(__file__).resolve().parents[1] / "analysis" / "pcapng_analysis.py"
    spec = importlib.util.spec_from_file_location("pcapng_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load pcapng_analysis module")
    sys.modules["pcapng_analysis"] = module
    spec.loader.exec_module(module)
    return module


class TestPcapngAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_pcapng_module()
        cls.analyzer = cls.module.PcapngAnalyzer("/tmp/does_not_exist.pcapng")

    def test_describe_numeric_empty(self):
        summary = self.analyzer._describe_numeric([])
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["min"])
        self.assertIsNone(summary["max"])
        self.assertIsNone(summary["mean"])
        self.assertIsNone(summary["median"])

    def test_describe_numeric_values(self):
        summary = self.analyzer._describe_numeric([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 4.0)
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["median"], 2.5)

    def test_percentile_and_iqr_bounds(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(self.analyzer._percentile(values, 50), 3.0)
        lower, upper = self.analyzer._iqr_bounds(values)
        self.assertLess(lower, min(values))
        self.assertGreater(upper, max(values))

    def test_parse_adv_data(self):
        payload = bytes([2, 0x01, 0x06, 3, 0x19, 0x00, 0x01])
        parsed = self.analyzer._parse_adv_data(payload)
        self.assertEqual(parsed[0]["type"], "0x01")
        self.assertEqual(parsed[0]["value"], "06")
        self.assertEqual(parsed[1]["type"], "0x19")
        self.assertEqual(parsed[1]["value"], "0001")

    def test_payload_entropy(self):
        entropy = self.analyzer._payload_entropy(["00", "ff"])
        self.assertAlmostEqual(entropy, 1.0, places=6)

    def test_check_hopping_detects_violation(self):
        record = self.module.PacketRecord
        records = [
            record(timestamp=0.0, channel=37, rssi=None, packet_type="adv_ind", primary_address="aa", addresses={}, adv_data_hex=None),
            record(timestamp=0.005, channel=37, rssi=None, packet_type="adv_ind", primary_address="aa", addresses={}, adv_data_hex=None),
            record(timestamp=0.009, channel=37, rssi=None, packet_type="adv_ind", primary_address="aa", addresses={}, adv_data_hex=None),
        ]
        violations = self.analyzer._check_hopping("aa", records)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["packets"], 3)

    def test_address_churn_windowing(self):
        record = self.module.PacketRecord
        records = [
            record(timestamp=0.1, channel=37, rssi=None, packet_type="adv_ind", primary_address="aa", addresses={}, adv_data_hex=None),
            record(timestamp=0.2, channel=37, rssi=None, packet_type="adv_ind", primary_address="bb", addresses={}, adv_data_hex=None),
        ]
        churn = self.analyzer._address_churn(records)
        self.assertEqual(churn["total_windows"], 1)
        self.assertEqual(churn["average_new_addresses"], 2)


if __name__ == "__main__":
    unittest.main()
