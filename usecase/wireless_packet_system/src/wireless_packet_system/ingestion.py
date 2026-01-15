from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Iterable, Optional

import ijson
from dateutil import parser as date_parser

from wireless_packet_system.gps import GPSSeries
from wireless_packet_system.models import PacketRecord


def _parse_epoch(value: str) -> datetime:
    epoch = float(value)
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _get_nested(packet: Dict[str, Any], path: Iterable[str]) -> Optional[str]:
    current: Any = packet
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, list):
        return current[0] if current else None
    return current


def _extract_bluetooth(packet: Dict[str, Any]) -> Dict[str, Any]:
    layers = packet.get("_source", {}).get("layers", {})
    bluetooth: Dict[str, Any] = {}
    bluetooth["device_address"] = _get_nested(layers, ["btle", "btle.advertising_address"])
    bluetooth["rssi_dbm"] = _get_nested(layers, ["btle_rf", "btle_rf.signal_dbm"])
    bluetooth["channel"] = _get_nested(layers, ["btle_rf", "btle_rf.channel"])
    return bluetooth


def _normalize_packet(
    packet: Dict[str, Any],
    session_id: str,
    protocol: str,
    gps_series: Optional[GPSSeries],
) -> PacketRecord:
    layers = packet.get("_source", {}).get("layers", {})
    epoch = _get_nested(layers, ["frame", "frame.time_epoch"]) or _get_nested(layers, ["frame", "frame.time"])
    if epoch is None:
        raise ValueError("Packet missing frame.time_epoch")

    timestamp = _parse_epoch(epoch) if epoch.replace(".", "", 1).isdigit() else date_parser.parse(epoch)
    timestamp = timestamp.astimezone(timezone.utc)

    bluetooth_fields = _extract_bluetooth(packet)
    device_address = bluetooth_fields.get("device_address")
    rssi_dbm = _coerce_float(bluetooth_fields.get("rssi_dbm"))
    channel = _coerce_int(bluetooth_fields.get("channel"))

    latitude = longitude = altitude = None
    if gps_series is not None:
        point = gps_series.interpolate(timestamp.timestamp())
        latitude = point.latitude
        longitude = point.longitude
        altitude = point.altitude

    return PacketRecord(
        timestamp=timestamp,
        session_id=session_id,
        protocol=protocol,
        device_address=device_address,
        rssi_dbm=rssi_dbm,
        channel=channel,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        payload=packet,
    )


def _coerce_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _coerce_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def load_packets(file_path: str) -> Generator[Dict[str, Any], None, None]:
    with open(file_path, "rb") as handle:
        parser = ijson.items(handle, "item")
        for packet in parser:
            yield packet


def normalize_packets(
    file_path: str,
    session_id: str,
    protocol: str,
    gps_series: Optional[GPSSeries],
) -> Generator[PacketRecord, None, None]:
    for packet in load_packets(file_path):
        yield _normalize_packet(packet, session_id=session_id, protocol=protocol, gps_series=gps_series)


def json_lines_packets(file_path: str) -> Generator[Dict[str, Any], None, None]:
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
