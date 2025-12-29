# PCAPNG Bluetooth LE Analysis API

The `analysis/pcapng_analysis.py` module provides a JSON-centric API for Bluetooth LE
pcapng captures. It is intended to be reused by notebooks such as
`show_le_pcapng.ipynb` and analysis scripts under `analysis/`.

## Key Features

- **Packet type analysis** (advertising, scan, connect, extended advertising)
- **Advertising payload parsing** into AD structure elements
- **Address logging** (counts + first/last seen)
- **Outlier detection** for inter-arrival times
- **Bluetooth specification heuristics**
  - Advertising channel compliance (37/38/39)
  - Minimum recommended advertising interval (20 ms)
  - Frequency-hopping checks within a 10 ms event window
- **Creative analysis**
  - Payload entropy estimation
  - Address churn over rolling 1 s windows

## Usage

```python
from analysis.pcapng_analysis import PcapngAnalyzer

analyzer = PcapngAnalyzer("/path/to/capture.pcapng")
report = analyzer.analyze()
print(report["summary"])

# Save a JSON report
analyzer.export_json("/tmp/ble_analysis.json")
```

## JSON Output Schema (High-Level)

```json
{
  "summary": {
    "pcap_path": "...",
    "total_packets": 0,
    "channels": [37, 38, 39],
    "unique_addresses": 0
  },
  "packet_types": {
    "counts": {"adv_ind": 123, "scan_rsp": 42},
    "total": 165
  },
  "advertising": {
    "total_advertising_packets": 0,
    "channel_counts": {"37": 12},
    "adv_data_length_stats": {"count": 0, "min": null, "max": null, "mean": null, "median": null},
    "adv_data_types": {"0x01": 10, "0xff": 5},
    "sample_adv_payloads": []
  },
  "addresses": {
    "unique_addresses": 0,
    "addresses": {
      "AA:BB:CC:DD:EE:FF": {"count": 5, "first_seen": 0.0, "last_seen": 1.2}
    }
  },
  "timing": {
    "inter_arrival_stats": {"count": 0, "min": null, "max": null, "mean": null, "median": null},
    "outliers": []
  },
  "violations": {
    "advertising_channel_violations": [],
    "advertising_interval_violations": [],
    "frequency_hopping_violations": []
  },
  "creative": {
    "payload_entropy_bits": 0.0,
    "address_churn": {"total_windows": 0, "average_new_addresses": 0, "windows": []}
  }
}
```

## Notes on Heuristics

- **Minimum advertising interval** uses a 20 ms threshold for the same address.
- **Frequency hopping** is evaluated within a 10 ms window. If three or more
  packets for the same address stay on a single channel, the window is flagged.
- These checks are intentionally conservative and should be tuned per capture
  type (e.g., extended advertising vs. legacy advertising).
