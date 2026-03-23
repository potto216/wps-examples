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

try:
    from scapy.layers.dot11 import Dot11, RadioTap  # type: ignore
except Exception:  # pragma: no cover - depends on scapy build
    Dot11 = None
    RadioTap = None


def _coerce_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _wifi_channel_from_frequency(frequency_mhz: Optional[int]) -> Optional[int]:
    if frequency_mhz is None:
        return None
    if frequency_mhz == 2484:
        return 14
    if 2412 <= frequency_mhz <= 2472:
        return ((frequency_mhz - 2412) // 5) + 1
    if 5000 <= frequency_mhz <= 5895:
        return (frequency_mhz - 5000) // 5
    if 5955 <= frequency_mhz <= 7115:
        return (frequency_mhz - 5950) // 5
    return None


def _get_packet_field(packet: Packet, field_name: str) -> object:
    try:
        value = packet.getfieldval(field_name)
    except Exception:
        value = None
    if value is not None:
        return value
    return getattr(packet, "fields", {}).get(field_name)


def _extract_channel_and_rssi(packet: Packet) -> tuple[Optional[int], Optional[int]]:
    channel = None
    rssi = None
    layer_presence = {
        "BTLE_PPI": BTLE_PPI is not None and packet.haslayer(BTLE_PPI),
        "BTLE_RF": BTLE_RF is not None and packet.haslayer(BTLE_RF),
        "RadioTap": RadioTap is not None and packet.haslayer(RadioTap),
        "Dot11": Dot11 is not None and packet.haslayer(Dot11),
    }
    logger.debug("Channel/RSSI layer presence: %s", layer_presence)

    if BTLE_PPI is not None and packet.haslayer(BTLE_PPI):
        ppi = packet[BTLE_PPI]
        channel = _coerce_optional_int(getattr(ppi, "btle_channel", None))
        rssi = _coerce_optional_int(getattr(ppi, "rssi_avg", None))
        logger.debug("Using BTLE_PPI metadata for channel/rssi extraction")
        return channel, rssi

    if BTLE_RF is not None and packet.haslayer(BTLE_RF):
        rf = packet[BTLE_RF]
        channel = _coerce_optional_int(getattr(rf, "rf_channel", None))
        rssi = _coerce_optional_int(getattr(rf, "signal", None))
        logger.debug("Using BTLE_RF metadata for channel/rssi extraction")
        return channel, rssi

    if RadioTap is not None and packet.haslayer(RadioTap):
        radiotap = packet[RadioTap]
        logger.debug("Using RadioTap metadata for channel/rssi extraction")
        channel = _coerce_optional_int(_get_packet_field(radiotap, "ChannelPlusNumber"))
        if channel is not None and channel <= 0:
            channel = None
        if channel is None:
            channel = _wifi_channel_from_frequency(
                _coerce_optional_int(_get_packet_field(radiotap, "ChannelFrequency"))
            )
        if channel is None:
            channel = _wifi_channel_from_frequency(
                _coerce_optional_int(_get_packet_field(radiotap, "ChannelPlusFrequency"))
            )
        rssi = _coerce_optional_int(_get_packet_field(radiotap, "dBm_AntSignal"))

    if not any(layer_presence.values()):
        logger.debug("No BTLE_PPI, BTLE_RF, RadioTap, or Dot11 layer found for channel/rssi extraction")

    return channel, rssi


def _extract_src_dst(packet: Packet) -> tuple[Optional[str], Optional[str]]:
    src = getattr(packet, "src", None)
    dst = getattr(packet, "dst", None)
    if src is not None or dst is not None:
        return (str(src) if src is not None else None, str(dst) if dst is not None else None)

    if Dot11 is not None and packet.haslayer(Dot11):
        dot11 = packet[Dot11]
        return dot11.addr2, dot11.addr1

    return None, None


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

        channel, rssi = _extract_channel_and_rssi(packet)
        src, dst = _extract_src_dst(packet)

        packet_bytes = bytes(packet)
        wire_length = _coerce_optional_int(getattr(packet, "wirelen", None))

        logger.debug(
            "Packet %d extracted channel=%s rssi=%s",
            packet_index,
            channel,
            rssi,
        )

        logger.debug(
            "Packet %d addressing src=%s dst=%s highest_layer=%s",
            packet_index,
            _safe_repr(src),
            _safe_repr(dst),
            getattr(packet, "lastlayer", lambda: "unknown")(),
        )

        row = PacketRow(
            packet_index=packet_index,
            timestamp=ts,
            timestamp_unix_s=ts_unix_s,
            captured_length=len(packet_bytes),
            wire_length=wire_length if wire_length is not None else len(packet_bytes),
            highest_layer=str(getattr(packet, "lastlayer", lambda: "unknown")()),
            layers="|".join(layer_names),
            summary=str(packet.summary()),
            channel=channel,
            rssi=rssi,
            src=src,
            dst=dst,
            raw_hex=packet_bytes.hex(),
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
