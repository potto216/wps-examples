import unittest

import pandas as pd
from scapy.all import Ether  # type: ignore

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


if __name__ == "__main__":
    unittest.main()
