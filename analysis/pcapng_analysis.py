"""PCAPNG Bluetooth LE analysis helpers.

This module provides a JSON-serializable analysis payload for Bluetooth LE
pcapng captures using Scapy. It focuses on advertising packets, address
logging, and timing/frequency-hopping heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scapy.all import Packet, rdpcap
from scapy.fields import Field as ScapyField

# Scapy 2.7 Bluetooth LE layers.
from scapy.layers.bluetooth4LE import (  # type: ignore
    BTLE,
    BTLE_ADV,
    BTLE_ADV_DIRECT_IND,
    BTLE_ADV_IND,
    BTLE_ADV_NONCONN_IND,
    BTLE_ADV_SCAN_IND,
    BTLE_CONNECT_REQ,
    BTLE_PPI,
    BTLE_RF,
    BTLE_SCAN_REQ,
    BTLE_SCAN_RSP,
)

ADV_CHANNELS = {37, 38, 39}
ADV_TYPE_LAYERS: Tuple[Tuple[type, str], ...] = (
    (BTLE_ADV_IND, "adv_ind"),
    (BTLE_ADV_DIRECT_IND, "adv_direct_ind"),
    (BTLE_ADV_NONCONN_IND, "adv_nonconn_ind"),
    (BTLE_ADV_SCAN_IND, "adv_scan_ind"),
    (BTLE_SCAN_REQ, "scan_req"),
    (BTLE_SCAN_RSP, "scan_rsp"),
    (BTLE_CONNECT_REQ, "connect_req"),
)

ADDRESS_FIELDS = ("AdvA", "ScanA", "InitA", "AAS")
ADV_DATA_FIELDS = ("AdvData", "ScanRspData", "Data")


def _format_bdaddr(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, bytes):
        if len(value) == 6:
            return ":".join(f"{b:02x}" for b in value)
        return value.hex()

    if isinstance(value, str):
        # Guard against accidentally stringifying Scapy Field descriptors.
        if value.startswith("<") and "Field" in value and value.endswith(">"):
            return None
        return value

    try:
        as_bytes = bytes(value)
    except Exception:
        return str(value)

    if len(as_bytes) == 6:
        return ":".join(f"{b:02x}" for b in as_bytes)
    return as_bytes.hex()


@dataclass
class PacketRecord:
    timestamp: float
    channel: Optional[int]
    rssi: Optional[int]
    packet_type: str
    primary_address: Optional[str]
    addresses: Dict[str, str]
    adv_data_hex: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "channel": self.channel,
            "rssi": self.rssi,
            "packet_type": self.packet_type,
            "primary_address": self.primary_address,
            "addresses": self.addresses,
            "adv_data_hex": self.adv_data_hex,
        }


class PcapngAnalyzer:
    """Analyze Bluetooth LE packets from a pcapng file."""

    def __init__(self, pcap_path: str):
        self.pcap_path = pcap_path

    def analyze(self) -> Dict[str, Any]:
        packets = rdpcap(self.pcap_path)
        records = self._extract_records(packets)
        summary = self._summary(records)
        packet_types = self._packet_types(records)
        advertising = self._advertising_analysis(records)
        addresses = self._address_log(records)
        timing = self._timing_analysis(records)
        violations = self._violation_analysis(records)
        creative = self._creative_analysis(records)

        return {
            "summary": summary,
            "packet_types": packet_types,
            "advertising": advertising,
            "addresses": addresses,
            "timing": timing,
            "violations": violations,
            "creative": creative,
        }

    def export_json(self, output_path: str) -> str:
        import json

        payload = self.analyze()
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return output_path

    def _extract_records(self, packets: Iterable[Packet]) -> List[PacketRecord]:
        if BTLE is None:
            raise RuntimeError(
                "Scapy BTLE layers are unavailable. Install/upgrade Scapy and ensure "
                "Bluetooth LE layers are present (e.g. scapy.layers.bluetooth4LE)."
            )
        records: List[PacketRecord] = []
        for packet in packets:
            if not packet.haslayer(BTLE):
                continue
            record = self._packet_to_record(packet)
            if record is not None:
                records.append(record)
        return records

    def _packet_to_record(self, packet: Packet) -> Optional[PacketRecord]:
        timestamp = float(packet.time)
        channel = None
        rssi = None
        packet_type = "unknown"
        # Channel/RSSI metadata is typically stored in capture metadata layers.
        if packet.haslayer(BTLE_PPI):
            ppi = packet[BTLE_PPI]
            chan = getattr(ppi, "btle_channel", None)
            channel = int(chan) if chan is not None else None
            rssi_value = getattr(ppi, "rssi_avg", None)
            rssi = int(rssi_value) if rssi_value is not None else None
        elif packet.haslayer(BTLE_RF):
            rf = packet[BTLE_RF]
            chan = getattr(rf, "rf_channel", None)
            channel = int(chan) if chan is not None else None
            rssi_value = getattr(rf, "signal", None)
            rssi = int(rssi_value) if rssi_value is not None else None
        for layer, name in ADV_TYPE_LAYERS:
            if packet.haslayer(layer):
                packet_type = name
                break

        addresses = self._extract_addresses(packet)
        primary_address = (
            addresses.get("AdvA")
            or addresses.get("ScanA")
            or addresses.get("InitA")
            or addresses.get("AAS")
            or next(iter(addresses.values()), None)
        )
        adv_data = self._extract_adv_data(packet)
        adv_data_hex = adv_data.hex() if adv_data else None

        return PacketRecord(
            timestamp=timestamp,
            channel=channel,
            rssi=rssi,
            packet_type=packet_type,
            primary_address=primary_address,
            addresses=addresses,
            adv_data_hex=adv_data_hex,
        )

    def _extract_addresses(self, packet: Packet) -> Dict[str, str]:
        addresses: Dict[str, str] = {}
        for layer_cls in packet.layers():
            layer = packet.getlayer(layer_cls)
            if layer is None:
                continue
            for field in ADDRESS_FIELDS:
                if field in getattr(layer, "fields", {}):
                    value = layer.fields.get(field)
                else:
                    value = getattr(layer, field, None)

                if value is None or isinstance(value, ScapyField):
                    continue

                text = _format_bdaddr(value)
                if text:
                    addresses[field] = text
        return addresses

    def _extract_adv_data(self, packet: Packet) -> Optional[bytes]:
        for layer in packet.layers():
            for field in ADV_DATA_FIELDS:
                if hasattr(layer, field):
                    value = getattr(layer, field)
                    if value:
                        if isinstance(value, bytes):
                            return value
                        try:
                            return bytes(value)
                        except TypeError:
                            return None
        return None

    def _summary(self, records: List[PacketRecord]) -> Dict[str, Any]:
        total = len(records)
        channels = sorted({record.channel for record in records if record.channel})
        addresses = {record.primary_address for record in records if record.primary_address}
        return {
            "pcap_path": self.pcap_path,
            "total_packets": total,
            "channels": channels,
            "unique_addresses": len(addresses),
        }

    def _packet_types(self, records: List[PacketRecord]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for record in records:
            counts[record.packet_type] = counts.get(record.packet_type, 0) + 1
        return {
            "counts": dict(sorted(counts.items(), key=lambda item: item[0])),
            "total": len(records),
        }

    def _advertising_analysis(self, records: List[PacketRecord]) -> Dict[str, Any]:
        advertising_records = [
            record
            for record in records
            if record.packet_type.startswith("adv_") or record.packet_type in {"scan_rsp", "scan_req"}
        ]
        channel_counts: Dict[str, int] = {}
        data_lengths: List[int] = []
        adv_data_types: Dict[str, int] = {}
        sample_data: List[Dict[str, Any]] = []

        for record in advertising_records:
            if record.channel is not None:
                channel_counts[str(record.channel)] = channel_counts.get(str(record.channel), 0) + 1
            if record.adv_data_hex:
                raw_bytes = bytes.fromhex(record.adv_data_hex)
                data_lengths.append(len(raw_bytes))
                elements = self._parse_adv_data(raw_bytes)
                for element in elements:
                    adv_type = element["type"]
                    adv_data_types[adv_type] = adv_data_types.get(adv_type, 0) + 1
                if len(sample_data) < 10:
                    sample_data.append(
                        {
                            "timestamp": record.timestamp,
                            "channel": record.channel,
                            "primary_address": record.primary_address,
                            "elements": elements,
                        }
                    )

        return {
            "total_advertising_packets": len(advertising_records),
            "channel_counts": channel_counts,
            "adv_data_length_stats": self._describe_numeric(data_lengths),
            "adv_data_types": dict(sorted(adv_data_types.items(), key=lambda item: item[0])),
            "sample_adv_payloads": sample_data,
        }

    def _address_log(self, records: List[PacketRecord]) -> Dict[str, Any]:
        addresses: Dict[str, Dict[str, Any]] = {}
        for record in records:
            if not record.primary_address:
                continue
            entry = addresses.setdefault(
                record.primary_address,
                {"count": 0, "first_seen": record.timestamp, "last_seen": record.timestamp},
            )
            entry["count"] += 1
            entry["first_seen"] = min(entry["first_seen"], record.timestamp)
            entry["last_seen"] = max(entry["last_seen"], record.timestamp)
        return {
            "unique_addresses": len(addresses),
            "addresses": addresses,
        }

    def _timing_analysis(self, records: List[PacketRecord]) -> Dict[str, Any]:
        by_address = self._group_by_address(records)
        inter_arrival_all: List[float] = []
        outliers: List[Dict[str, Any]] = []
        for address, items in by_address.items():
            times = sorted(item.timestamp for item in items)
            deltas = [t2 - t1 for t1, t2 in zip(times, times[1:]) if t2 > t1]
            inter_arrival_all.extend(deltas)
            if len(deltas) < 4:
                continue
            lower, upper = self._iqr_bounds(deltas)
            for time, delta in zip(times[1:], deltas):
                if delta < lower or delta > upper:
                    outliers.append(
                        {
                            "address": address,
                            "timestamp": time,
                            "inter_arrival": delta,
                            "lower_bound": lower,
                            "upper_bound": upper,
                        }
                    )
        return {
            "inter_arrival_stats": self._describe_numeric(inter_arrival_all),
            "outliers": outliers,
        }

    def _violation_analysis(self, records: List[PacketRecord]) -> Dict[str, Any]:
        min_adv_interval_s = 0.02
        channel_violations: List[Dict[str, Any]] = []
        interval_violations: List[Dict[str, Any]] = []
        hop_violations: List[Dict[str, Any]] = []

        for record in records:
            if record.packet_type.startswith("adv_") and record.channel not in ADV_CHANNELS:
                channel_violations.append(
                    {
                        "timestamp": record.timestamp,
                        "channel": record.channel,
                        "packet_type": record.packet_type,
                    }
                )

        by_address = self._group_by_address(records)
        for address, items in by_address.items():
            items_sorted = sorted(items, key=lambda item: item.timestamp)
            for prev, current in zip(items_sorted, items_sorted[1:]):
                delta = current.timestamp - prev.timestamp
                if delta < min_adv_interval_s:
                    interval_violations.append(
                        {
                            "address": address,
                            "timestamp": current.timestamp,
                            "inter_arrival": delta,
                            "minimum_recommended_interval": min_adv_interval_s,
                        }
                    )
            hop_violations.extend(self._check_hopping(address, items_sorted))

        return {
            "advertising_channel_violations": channel_violations,
            "advertising_interval_violations": interval_violations,
            "frequency_hopping_violations": hop_violations,
        }

    def _creative_analysis(self, records: List[PacketRecord]) -> Dict[str, Any]:
        payloads = [record.adv_data_hex for record in records if record.adv_data_hex]
        entropy = self._payload_entropy(payloads)
        churn = self._address_churn(records)
        return {
            "payload_entropy_bits": entropy,
            "address_churn": churn,
        }

    def _group_by_address(self, records: List[PacketRecord]) -> Dict[str, List[PacketRecord]]:
        grouped: Dict[str, List[PacketRecord]] = {}
        for record in records:
            address = record.primary_address or "unknown"
            grouped.setdefault(address, []).append(record)
        return grouped

    def _describe_numeric(self, values: List[float]) -> Dict[str, Any]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
        values_sorted = sorted(values)
        count = len(values_sorted)
        mid = count // 2
        if count % 2 == 0:
            median = (values_sorted[mid - 1] + values_sorted[mid]) / 2
        else:
            median = values_sorted[mid]
        mean = sum(values_sorted) / count
        return {
            "count": count,
            "min": values_sorted[0],
            "max": values_sorted[-1],
            "mean": mean,
            "median": median,
        }

    def _iqr_bounds(self, values: List[float]) -> Tuple[float, float]:
        values_sorted = sorted(values)
        q1 = self._percentile(values_sorted, 25)
        q3 = self._percentile(values_sorted, 75)
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    def _percentile(self, values: List[float], percentile: int) -> float:
        if not values:
            return 0.0
        k = (len(values) - 1) * (percentile / 100.0)
        f = int(k)
        c = min(f + 1, len(values) - 1)
        if f == c:
            return values[f]
        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        return d0 + d1

    def _parse_adv_data(self, data: bytes) -> List[Dict[str, str]]:
        elements: List[Dict[str, str]] = []
        index = 0
        while index < len(data):
            length = data[index]
            if length == 0:
                break
            end = index + 1 + length
            element = data[index + 1 : end]
            if not element:
                break
            ad_type = element[0]
            ad_value = element[1:]
            elements.append(
                {
                    "length": str(length),
                    "type": f"0x{ad_type:02x}",
                    "value": ad_value.hex(),
                }
            )
            index = end
        return elements

    def _check_hopping(
        self, address: str, records: List[PacketRecord]
    ) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        if not records:
            return violations

        window_start = records[0].timestamp
        channels: List[int] = []
        window = []

        for record in records:
            if record.channel is None:
                continue
            if record.timestamp - window_start <= 0.01:
                window.append(record)
                channels.append(record.channel)
            else:
                violations.extend(self._evaluate_channel_window(address, window, channels))
                window = [record]
                channels = [record.channel]
                window_start = record.timestamp

        violations.extend(self._evaluate_channel_window(address, window, channels))
        return violations

    def _evaluate_channel_window(
        self, address: str, window: List[PacketRecord], channels: List[int]
    ) -> List[Dict[str, Any]]:
        if not window:
            return []
        unique_channels = set(channels)
        if len(unique_channels) <= 1 and len(window) >= 3:
            return [
                {
                    "address": address,
                    "start_time": window[0].timestamp,
                    "end_time": window[-1].timestamp,
                    "channels": sorted(unique_channels),
                    "packets": len(window),
                }
            ]
        return []

    def _payload_entropy(self, payloads: List[str]) -> float:
        if not payloads:
            return 0.0
        counts = [0] * 256
        total = 0
        for payload_hex in payloads:
            payload = bytes.fromhex(payload_hex)
            for byte in payload:
                counts[byte] += 1
                total += 1
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts:
            if count == 0:
                continue
            probability = count / total
            entropy -= probability * self._log2(probability)
        return entropy

    def _log2(self, value: float) -> float:
        import math

        return math.log(value, 2)

    def _address_churn(self, records: List[PacketRecord]) -> Dict[str, Any]:
        if not records:
            return {"total_windows": 0, "average_new_addresses": 0}
        sorted_records = sorted(records, key=lambda record: record.timestamp)
        window_size = 1.0
        windows: List[Dict[str, Any]] = []
        start = sorted_records[0].timestamp
        current_addresses: set[str] = set()
        total_new = 0
        current_index = 0

        while current_index < len(sorted_records):
            window_end = start + window_size
            new_addresses: set[str] = set()
            while (
                current_index < len(sorted_records)
                and sorted_records[current_index].timestamp <= window_end
            ):
                address = sorted_records[current_index].primary_address
                if address and address not in current_addresses:
                    new_addresses.add(address)
                current_index += 1
            current_addresses |= new_addresses
            total_new += len(new_addresses)
            windows.append(
                {
                    "window_start": start,
                    "window_end": window_end,
                    "new_addresses": len(new_addresses),
                }
            )
            start = window_end

        return {
            "total_windows": len(windows),
            "average_new_addresses": total_new / len(windows) if windows else 0,
            "windows": windows,
        }
