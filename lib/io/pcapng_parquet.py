"""Convert pcapng packets to pandas DataFrame rows and save as parquet."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from scapy.all import Packet, rdpcap

# Optional Bluetooth metadata extraction when present.
try:
    from scapy.layers.bluetooth4LE import BTLE_PPI, BTLE_RF  # type: ignore
except Exception:  # pragma: no cover - depends on scapy build
    BTLE_PPI = None
    BTLE_RF = None


logger = logging.getLogger(__name__)


def _safe_repr(value: Any, limit: int = 200) -> str:
    try:
        rendered = repr(value)
    except Exception as exc:  # pragma: no cover - defensive logging helper
        rendered = f"<repr failed: {exc}>"
    if len(rendered) > limit:
        return f"{rendered[: limit - 3]}..."
    return rendered


def _packet_layer_debug_info(packet: Packet) -> List[Dict[str, object]]:
    layer_debug_info: List[Dict[str, object]] = []
    current_layer = packet
    visited_layer_ids = set()

    while current_layer is not None and id(current_layer) not in visited_layer_ids:
        visited_layer_ids.add(id(current_layer))
        field_names = list(getattr(current_layer, "fields", {}).keys())
        field_values = {name: _safe_repr(getattr(current_layer, name, None)) for name in field_names}
        layer_debug_info.append(
            {
                "layer": current_layer.__class__.__name__,
                "field_names": field_names,
                "field_values": field_values,
            }
        )
        payload = getattr(current_layer, "payload", None)
        if payload is None or payload is current_layer:
            break
        payload_name = payload.__class__.__name__
        if payload_name == "NoPayload":
            break
        current_layer = payload

    return layer_debug_info


@dataclass
class PacketRow:
    packet_index: int
    timestamp: Optional[pd.Timestamp]
    timestamp_unix_s: Optional[float]
    captured_length: int
    wire_length: int
    highest_layer: str
    layers: str
    summary: str
    channel: Optional[int]
    rssi: Optional[int]
    src: Optional[str]
    dst: Optional[str]
    raw_hex: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "packet_index": self.packet_index,
            "timestamp": self.timestamp,
            "timestamp_unix_s": self.timestamp_unix_s,
            "captured_length": self.captured_length,
            "wire_length": self.wire_length,
            "highest_layer": self.highest_layer,
            "layers": self.layers,
            "summary": self.summary,
            "channel": self.channel,
            "rssi": self.rssi,
            "src": self.src,
            "dst": self.dst,
            "raw_hex": self.raw_hex,
        }


class PcapngPacketDataFrameConverter:
    """Create a one-row-per-packet DataFrame from a pcapng file."""

    def __init__(self, pcapng_path: str):
        self.pcapng_path = pcapng_path

    def to_dataframe(self) -> pd.DataFrame:
        logger.debug("Reading pcapng file %s", self.pcapng_path)
        packets = rdpcap(self.pcapng_path)
        logger.debug("Loaded %d packets from %s", len(packets), self.pcapng_path)
        rows = [self._packet_to_row(i, packet).to_dict() for i, packet in enumerate(packets)]
        df = pd.DataFrame(rows)
        logger.debug(
            "Constructed DataFrame for %s with shape=%s columns=%s",
            self.pcapng_path,
            df.shape,
            list(df.columns),
        )
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            logger.debug("Normalized timestamp column to UTC datetime for %s", self.pcapng_path)
        return df

    def _packet_to_row(self, packet_index: int, packet: Packet) -> PacketRow:
        logger.debug("Processing packet_index=%d summary=%s", packet_index, packet.summary())
        layer_names = [layer.__name__ for layer in packet.layers()]
        layer_debug_info = _packet_layer_debug_info(packet)
        logger.debug(
            "Packet %d layers=%s layer_field_details=%s",
            packet_index,
            layer_names,
            layer_debug_info,
        )

        raw_time = getattr(packet, "time", None)
        ts_unix_s: Optional[float]
        if raw_time is None:
            logger.debug("Packet %d has no packet.time value", packet_index)
            ts_unix_s = None
        else:
            try:
                ts_unix_s = float(raw_time)
                logger.debug(
                    "Packet %d timestamp raw=%s parsed_unix_s=%s",
                    packet_index,
                    _safe_repr(raw_time),
                    ts_unix_s,
                )
            except Exception as exc:
                logger.debug(
                    "Packet %d could not parse packet.time raw=%s error=%s",
                    packet_index,
                    _safe_repr(raw_time),
                    exc,
                )
                ts_unix_s = None

        ts: Optional[pd.Timestamp]
        if ts_unix_s is None:
            ts = None
        else:
            # Match analysis/plot_bluetooth_gps_map.py: interpret scapy packet.time as seconds since Unix epoch.
            ts = pd.to_datetime(ts_unix_s, unit="s", utc=True, errors="coerce")
            if ts is pd.NaT or pd.isna(ts):
                logger.debug("Packet %d timestamp conversion returned NaT", packet_index)
                ts = None
            else:
                logger.debug("Packet %d timestamp=%s", packet_index, ts)

        channel = None
        rssi = None
        channel_source = None
        if BTLE_PPI is not None and packet.haslayer(BTLE_PPI):
            ppi = packet[BTLE_PPI]
            chan = getattr(ppi, "btle_channel", None)
            channel = int(chan) if chan is not None else None
            signal = getattr(ppi, "rssi_avg", None)
            rssi = int(signal) if signal is not None else None
            channel_source = "BTLE_PPI"
        elif BTLE_RF is not None and packet.haslayer(BTLE_RF):
            rf = packet[BTLE_RF]
            chan = getattr(rf, "rf_channel", None)
            channel = int(chan) if chan is not None else None
            signal = getattr(rf, "signal", None)
            rssi = int(signal) if signal is not None else None
            channel_source = "BTLE_RF"

        logger.debug(
            "Packet %d extracted channel=%s rssi=%s channel_source=%s",
            packet_index,
            channel,
            rssi,
            channel_source,
        )

        src = getattr(packet, "src", None)
        dst = getattr(packet, "dst", None)
        logger.debug(
            "Packet %d addressing src=%s dst=%s highest_layer=%s",
            packet_index,
            _safe_repr(src),
            _safe_repr(dst),
            getattr(packet, "lastlayer", lambda: "unknown")(),
        )

        raw_bytes = bytes(packet)
        row = PacketRow(
            packet_index=packet_index,
            timestamp=ts,
            timestamp_unix_s=ts_unix_s,
            captured_length=len(raw_bytes),
            wire_length=int(getattr(packet, "wirelen", len(raw_bytes))),
            highest_layer=str(getattr(packet, "lastlayer", lambda: "unknown")()),
            layers="|".join(layer_names),
            summary=str(packet.summary()),
            channel=channel,
            rssi=rssi,
            src=str(src) if src is not None else None,
            dst=str(dst) if dst is not None else None,
            raw_hex=raw_bytes.hex(),
        )
        logger.debug("Packet %d row=%s", packet_index, row.to_dict())
        return row


def parquet_path_for(pcapng_path: str) -> str:
    base, _ = os.path.splitext(pcapng_path)
    return f"{base}.parquet"


def iter_pcapng_files(path: str, recursive: bool) -> Iterable[str]:
    if os.path.isfile(path):
        if path.lower().endswith(".pcapng"):
            yield path
        return

    if recursive:
        for root, _, files in os.walk(path):
            for name in files:
                if name.lower().endswith(".pcapng"):
                    yield os.path.join(root, name)
        return

    for name in os.listdir(path):
        if name.lower().endswith(".pcapng"):
            yield os.path.join(path, name)


def convert_pcapng_path_to_parquet(pcapng_path: str, parquet_path: Optional[str] = None) -> str:
    output_path = parquet_path or parquet_path_for(pcapng_path)
    converter = PcapngPacketDataFrameConverter(pcapng_path)
    dataframe = converter.to_dataframe()
    dataframe.to_parquet(output_path, engine="pyarrow", index=False)
    return output_path
