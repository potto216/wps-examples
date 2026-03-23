import unittest

import pandas as pd
from scapy.all import Ether  # type: ignore
from scapy.layers.dot11 import Dot11, RadioTap  # type: ignore

from lib.io.pcapng_parquet import PcapngPacketDataFrameConverter


class TestPcapngParquetTimestamp(unittest.TestCase):
    def test_packet_to_row_emits_utc_timestamp_and_unix_seconds(self):
        pkt = Ether()
        pkt.time = 1700000000.5

        converter = PcapngPacketDataFrameConverter("dummy.pcapng")
        row = converter._packet_to_row(0, pkt).to_dict()

        self.assertIn("timestamp", row)
        self.assertIn("timestamp_unix_s", row)
        self.assertAlmostEqual(float(row["timestamp_unix_s"]), 1700000000.5)

        ts = row["timestamp"]
        self.assertIsInstance(ts, pd.Timestamp)
        self.assertTrue(ts.tz is not None)
        self.assertEqual(str(ts.tz), "UTC")
        self.assertEqual(ts.value, pd.Timestamp("2023-11-14T22:13:20.5Z").value)

    def test_packet_to_row_extracts_wifi_addresses_channel_and_rssi(self):
        pkt = RadioTap(ChannelFrequency=2412, dBm_AntSignal=-47) / Dot11(
            addr1="11:22:33:44:55:66",
            addr2="aa:bb:cc:dd:ee:ff",
            addr3="77:88:99:aa:bb:cc",
        )

        converter = PcapngPacketDataFrameConverter("dummy.pcapng")
        row = converter._packet_to_row(0, pkt).to_dict()

        self.assertEqual(row["channel"], 1)
        self.assertEqual(row["rssi"], -47)
        self.assertEqual(row["src"], "aa:bb:cc:dd:ee:ff")
        self.assertEqual(row["dst"], "11:22:33:44:55:66")


if __name__ == "__main__":
    unittest.main()
