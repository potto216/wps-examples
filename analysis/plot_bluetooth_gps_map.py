"""Plot Bluetooth packet locations on a GPS track map."""

from __future__ import annotations

import argparse
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
from scapy.all import Packet, rdpcap
from scapy.layers.bluetooth import (
    BTLE,
    HCI_ACL_Hdr,
    HCI_Command_Hdr,
    HCI_Event_Hdr,
    HCI_Hdr,
    L2CAP_Hdr,
)

from analysis.gps import GpsColumns, interpolate_gps_events, load_gpx, load_kml


def _load_gps_data(path: str, gps_type: str) -> pd.DataFrame:
    if gps_type == "gpx":
        gps_df = load_gpx(path)
    elif gps_type == "kml":
        gps_df = load_kml(path)
    else:
        raise ValueError(f"Unsupported GPS file type: {gps_type}")

    if GpsColumns.timestamp not in gps_df.columns:
        raise ValueError(
            "GPS data is missing timestamp information; use a GPX file with track timestamps."
        )

    gps_df[GpsColumns.timestamp] = pd.to_datetime(
        gps_df[GpsColumns.timestamp], utc=True, errors="coerce"
    )
    gps_df = gps_df.dropna(subset=[GpsColumns.timestamp, GpsColumns.latitude, GpsColumns.longitude])
    if gps_df.empty:
        raise ValueError("GPS data contains no usable timestamped fixes.")
    return gps_df


def _classify_packet(packet: Packet) -> Optional[str]:
    if packet.haslayer(BTLE):
        return "le"
    if packet.haslayer((HCI_ACL_Hdr, HCI_Command_Hdr, HCI_Event_Hdr, HCI_Hdr, L2CAP_Hdr)):
        return "br_edr"
    return None


def _load_packet_events(pcap_path: str) -> pd.DataFrame:
    packets = rdpcap(pcap_path)
    rows = []
    for packet in packets:
        packet_type = _classify_packet(packet)
        if packet_type is None:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(float(packet.time), unit="s", utc=True),
                "packet_type": packet_type,
            }
        )

    return pd.DataFrame(rows)


def plot_packets_on_map(gps_df: pd.DataFrame, packet_df: pd.DataFrame) -> None:
    if packet_df.empty:
        raise ValueError("No Bluetooth packets were found in the capture.")

    events_with_locations = interpolate_gps_events(
        gps_df,
        packet_df,
        gps_time_col=GpsColumns.timestamp,
        event_time_col="timestamp",
        lat_col=GpsColumns.latitude,
        lon_col=GpsColumns.longitude,
    )
    events_with_locations = events_with_locations.dropna(
        subset=[GpsColumns.latitude, GpsColumns.longitude]
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(
        gps_df[GpsColumns.longitude],
        gps_df[GpsColumns.latitude],
        color="gray",
        linewidth=1.5,
        label="GPS Track",
    )

    le_events = events_with_locations[events_with_locations["packet_type"] == "le"]
    br_events = events_with_locations[events_with_locations["packet_type"] == "br_edr"]

    if not le_events.empty:
        ax.scatter(
            le_events[GpsColumns.longitude],
            le_events[GpsColumns.latitude],
            color="tab:blue",
            s=24,
            label="Bluetooth LE",
        )

    if not br_events.empty:
        ax.scatter(
            br_events[GpsColumns.longitude],
            br_events[GpsColumns.latitude],
            color="tab:orange",
            s=24,
            label="Bluetooth BR/EDR",
        )

    ax.set_title("Bluetooth Packet Locations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show(block=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Bluetooth packet locations on a GPS track, interpolating packet timestamps."
        )
    )
    parser.add_argument("--pcapng", required=True, help="Path to the Bluetooth pcapng file.")
    parser.add_argument("--gps", required=True, help="Path to the GPS file.")
    parser.add_argument(
        "--gps-type",
        default="gpx",
        choices=["gpx", "kml"],
        help="GPS file type (default: gpx).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    gps_df = _load_gps_data(args.gps, args.gps_type)
    packet_df = _load_packet_events(args.pcapng)
    plot_packets_on_map(gps_df, packet_df)


if __name__ == "__main__":
    main()
