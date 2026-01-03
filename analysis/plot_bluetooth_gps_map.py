"""Plot Bluetooth packet locations on a GPS track map."""

from __future__ import annotations

import argparse
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from scapy.all import Packet, rdpcap
from scapy.layers.bluetooth import HCI_ACL_Hdr, HCI_Command_Hdr, HCI_Event_Hdr, HCI_Hdr, L2CAP_Hdr

# Scapy's Bluetooth LE layers move across versions.
try:
    # Scapy 2.7 Bluetooth LE layers.
    from scapy.layers.bluetooth4LE import (  # type: ignore
        BTLE,
        BTLE_ADV_DIRECT_IND,
        BTLE_ADV_IND,
        BTLE_ADV_NONCONN_IND,
        BTLE_ADV_SCAN_IND,
        BTLE_CONNECT_REQ,
        BTLE_SCAN_REQ,
        BTLE_SCAN_RSP,
    )
except ImportError:  # pragma: no cover
    try:
        # Some older Scapy installs expose BTLE here, but not all ADV_* layers.
        from scapy.layers.bluetooth import BTLE  # type: ignore
    except ImportError:  # pragma: no cover
        BTLE = None  # type: ignore[assignment]

    BTLE_ADV_DIRECT_IND = None  # type: ignore[assignment]
    BTLE_ADV_IND = None  # type: ignore[assignment]
    BTLE_ADV_NONCONN_IND = None  # type: ignore[assignment]
    BTLE_ADV_SCAN_IND = None  # type: ignore[assignment]
    BTLE_CONNECT_REQ = None  # type: ignore[assignment]
    BTLE_SCAN_REQ = None  # type: ignore[assignment]
    BTLE_SCAN_RSP = None  # type: ignore[assignment]


LE_ADV_TYPE_LAYERS: tuple[tuple[object, str], ...] = tuple(
    (layer, name)
    for layer, name in (
        (BTLE_ADV_IND, "adv_ind"),
        (BTLE_ADV_DIRECT_IND, "adv_direct_ind"),
        (BTLE_ADV_NONCONN_IND, "adv_nonconn_ind"),
        (BTLE_ADV_SCAN_IND, "adv_scan_ind"),
        (BTLE_SCAN_REQ, "scan_req"),
        (BTLE_SCAN_RSP, "scan_rsp"),
        (BTLE_CONNECT_REQ, "connect_req"),
    )
    if layer is not None
)

from gps import GpsColumns, interpolate_gps_events, load_gpx, load_kml


def _fmt_ts(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "(none)"

    # Always display in UTC, explicitly labeled.
    if getattr(ts, "tzinfo", None) is None:
        ts_utc = ts.tz_localize("UTC")
    else:
        ts_utc = ts.tz_convert("UTC")
    # Use ISO-8601 with 'Z' suffix.
    text = ts_utc.isoformat().replace("+00:00", "Z")
    return f"{text} (UTC)"


def _fmt_range(ts_min: Optional[pd.Timestamp], ts_max: Optional[pd.Timestamp]) -> str:
    return f"{_fmt_ts(ts_min)} .. {_fmt_ts(ts_max)}"


def _parse_datetime_arg(value: Optional[str], *, is_stop: bool) -> tuple[Optional[pd.Timestamp], bool]:
    """Parse a CLI datetime string into a timezone-aware UTC Timestamp.

    Accepted formats:
    - Date only: YYYY-MM-DD
    - Datetime: ISO-8601 or common forms like YYYY-MM-DD HH:MM[:SS]

    Notes:
    - If no timezone is provided, the value is interpreted as UTC.
    - For date-only stop values, the range end is treated as exclusive at next midnight UTC.
      Example: --stop 2026-01-02 means "< 2026-01-03T00:00:00Z".
    """

    if value is None:
        return None, False

    text = value.strip()
    is_date_only = len(text) == 10 and text[4] == "-" and text[7] == "-"
    if is_date_only:
        ts = pd.to_datetime(text, utc=True, errors="raise")
        if is_stop:
            return ts + pd.Timedelta(days=1), True
        return ts, True

    ts = pd.to_datetime(text, utc=True, errors="raise")
    return ts, False


def _parse_time_offset(value: Optional[str]) -> pd.Timedelta:
    """Parse a time offset string into a Timedelta.

    The offset is *added* to packet timestamps.

    Accepted formats:
    - "+HH:MM" or "-HH:MM" (recommended)
    - integer hours like "+5" or "-4"

    Example: if the capture timestamps are local time in UTC-05:00 but should align to UTC,
    use --pcap-time-offset +05:00.
    """

    if value is None:
        return pd.Timedelta(0)

    text = value.strip()
    if not text:
        return pd.Timedelta(0)

    # Hours only
    if ":" not in text:
        try:
            hours = int(text)
        except ValueError as exc:
            raise ValueError(f"Invalid --pcap-time-offset '{value}'. Use +HH:MM or -HH:MM.") from exc
        return pd.Timedelta(hours=hours)

    # +HH:MM / -HH:MM
    sign = 1
    if text[0] == "+":
        text = text[1:]
    elif text[0] == "-":
        sign = -1
        text = text[1:]

    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid --pcap-time-offset '{value}'. Use +HH:MM or -HH:MM.")
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid --pcap-time-offset '{value}'. Use +HH:MM or -HH:MM.") from exc

    if hours < 0 or minutes < 0 or minutes >= 60:
        raise ValueError(f"Invalid --pcap-time-offset '{value}'. Use +HH:MM or -HH:MM.")

    return sign * pd.Timedelta(hours=hours, minutes=minutes)


def _apply_time_range_filter(
    df: pd.DataFrame,
    *,
    time_col: str,
    start: Optional[pd.Timestamp],
    stop: Optional[pd.Timestamp],
    stop_is_exclusive: bool,
) -> pd.DataFrame:
    if start is None and stop is None:
        return df

    filtered = df
    if start is not None:
        filtered = filtered[filtered[time_col] >= start]
    if stop is not None:
        if stop_is_exclusive:
            filtered = filtered[filtered[time_col] < stop]
        else:
            filtered = filtered[filtered[time_col] <= stop]
    return filtered


def _load_gps_data(path: str, gps_type: str, *, verbose: bool = False) -> pd.DataFrame:
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

    if verbose:
        ts_min = gps_df[GpsColumns.timestamp].min()
        ts_max = gps_df[GpsColumns.timestamp].max()
        print(f"GPS fixes loaded: {len(gps_df)}")
        print(f"GPS time range: {_fmt_range(ts_min, ts_max)}")
    return gps_df


def _classify_packet(packet: Packet, *, detailed_le_adv_types: bool) -> Optional[str]:
    if BTLE is not None and packet.haslayer(BTLE):
        if detailed_le_adv_types and LE_ADV_TYPE_LAYERS:
            for layer, name in LE_ADV_TYPE_LAYERS:
                # NOTE: typing: Scapy's haslayer() accepts either a class or a name.
                if packet.haslayer(layer):
                    return name
            return "le_other"
        return "le"
    if packet.haslayer((HCI_ACL_Hdr, HCI_Command_Hdr, HCI_Event_Hdr, HCI_Hdr, L2CAP_Hdr)):
        return "br_edr"
    return None


def _load_packet_events(
    pcap_path: str,
    *,
    time_offset: pd.Timedelta = pd.Timedelta(0),
    detailed_le_adv_types: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    packets = rdpcap(pcap_path)
    total_packets = len(packets)

    pcap_ts_min: Optional[pd.Timestamp] = None
    pcap_ts_max: Optional[pd.Timestamp] = None
    if total_packets:
        # Use *all* packet timestamps for the capture range (even if classification drops most).
        pcap_ts_min = pd.to_datetime(float(packets[0].time), unit="s", utc=True) + time_offset
        pcap_ts_max = pcap_ts_min
        for packet in packets[1:]:
            ts = pd.to_datetime(float(packet.time), unit="s", utc=True) + time_offset
            if ts < pcap_ts_min:
                pcap_ts_min = ts
            if ts > pcap_ts_max:
                pcap_ts_max = ts

    rows = []
    for packet in packets:
        packet_type = _classify_packet(packet, detailed_le_adv_types=detailed_le_adv_types)
        if packet_type is None:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(float(packet.time), unit="s", utc=True) + time_offset,
                "packet_type": packet_type,
            }
        )

    packet_df = pd.DataFrame(rows)
    packet_df.attrs["pcap_time_min"] = pcap_ts_min
    packet_df.attrs["pcap_time_max"] = pcap_ts_max

    if verbose:
        if not packet_df.empty:
            if detailed_le_adv_types:
                counts = packet_df["packet_type"].value_counts().to_dict()
            else:
                counts = packet_df["packet_type"].value_counts().to_dict()
        else:
            counts = {}

        le_count = int(counts.get("le", 0))
        br_count = int(counts.get("br_edr", 0))
        print(f"Packets read from capture: {total_packets}")
        if time_offset != pd.Timedelta(0):
            # Timedelta string is clear enough for logs (e.g., '0 days 05:00:00').
            print(f"PCAP time offset applied: {time_offset}")
        if pcap_ts_min is not None and pcap_ts_max is not None:
            print(f"PCAP time range (all packets): {_fmt_range(pcap_ts_min, pcap_ts_max)}")
        if detailed_le_adv_types:
            non_br = len(packet_df) - br_count
            print(
                f"Bluetooth packets classified: {len(packet_df)} (LE detailed={non_br}, BR/EDR={br_count})"
            )
            if counts:
                ordered = dict(sorted(counts.items(), key=lambda kv: kv[0]))
                print(f"Packet type counts: {ordered}")
        else:
            print(f"Bluetooth packets classified: {len(packet_df)} (LE={le_count}, BR/EDR={br_count})")
        if not packet_df.empty:
            ts_min = packet_df["timestamp"].min()
            ts_max = packet_df["timestamp"].max()
            print(f"Packet time range: {_fmt_range(ts_min, ts_max)}")

    return packet_df


def _type_color_map(types: list[str]) -> dict[str, object]:
    if not types:
        return {}
    cmap_name = "tab10" if len(types) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    return {t: cmap(i % cmap.N) for i, t in enumerate(sorted(types))}


def plot_packets_on_map(gps_df: pd.DataFrame, packet_df: pd.DataFrame, *, verbose: bool = False) -> None:
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

    if verbose:
        matched = len(events_with_locations)
        print(f"Packets with interpolated GPS locations: {matched} / {len(packet_df)}")
        if matched:
            le_matched = int((events_with_locations["packet_type"] == "le").sum())
            br_matched = int((events_with_locations["packet_type"] == "br_edr").sum())
            print(f"Matched by type: LE={le_matched}, BR/EDR={br_matched}")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(
        gps_df[GpsColumns.longitude],
        gps_df[GpsColumns.latitude],
        color="gray",
        linewidth=1.5,
        label="GPS Track",
    )

    types_present = sorted(events_with_locations["packet_type"].dropna().unique().tolist())
    if types_present == ["br_edr", "le"] or types_present == ["le", "br_edr"]:
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
    else:
        color_map = _type_color_map(types_present)
        for pkt_type in types_present:
            subset = events_with_locations[events_with_locations["packet_type"] == pkt_type]
            if subset.empty:
                continue
            label = pkt_type if pkt_type != "br_edr" else "br_edr"
            ax.scatter(
                subset[GpsColumns.longitude],
                subset[GpsColumns.latitude],
                color=color_map.get(pkt_type, "tab:blue"),
                s=24,
                label=label,
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
    parser.add_argument(
        "--pcap-time-offset",
        help=(
            "Time offset to add to packet timestamps before filtering/interpolation. "
            "Formats: +HH:MM or -HH:MM (recommended) or integer hours like +5/-4. "
            "Example: if the pcap timestamps are local time UTC-05:00 but you want UTC, use +05:00."
        ),
    )
    parser.add_argument("--gps", required=True, help="Path to the GPS file.")
    parser.add_argument(
        "--gps-type",
        default="gpx",
        choices=["gpx", "kml"],
        help="GPS file type (default: gpx).",
    )
    parser.add_argument(
        "--start",
        help=(
            "Start time filter (UTC if no timezone). Formats: YYYY-MM-DD or "
            "YYYY-MM-DD HH:MM[:SS] or ISO-8601 (e.g. 2026-01-02T13:45:00Z)."
        ),
    )
    parser.add_argument(
        "--stop",
        help=(
            "Stop time filter (UTC if no timezone). Formats: YYYY-MM-DD or "
            "YYYY-MM-DD HH:MM[:SS] or ISO-8601. Date-only stop is exclusive at next midnight "
            "(e.g. --stop 2026-01-02 means < 2026-01-03T00:00:00Z)."
        ),
    )
    parser.add_argument(
        "--use-pcap-range",
        action="store_true",
        help=(
            "Use the pcapng timestamp range (min/max) as the start/stop filter. "
            "Overrides --start/--stop."
        ),
    )
    parser.add_argument(
        "--color-by-packet-type",
        action="store_true",
        help=(
            "Plot Bluetooth packet locations with a different color per packet type. "
            "For LE packets, this groups by Scapy 2.7 BTLE_ADV_* / SCAN_* / CONNECT_REQ layers "
            "when available; otherwise it falls back to a single LE category."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print debugging information about loaded data.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pcap_time_offset = _parse_time_offset(args.pcap_time_offset)

    gps_df = _load_gps_data(args.gps, args.gps_type, verbose=args.verbose)
    gps_time_min_loaded = gps_df[GpsColumns.timestamp].min() if not gps_df.empty else None
    gps_time_max_loaded = gps_df[GpsColumns.timestamp].max() if not gps_df.empty else None

    packet_df = _load_packet_events(
        args.pcapng,
        time_offset=pcap_time_offset,
        detailed_le_adv_types=bool(args.color_by_packet_type),
        verbose=args.verbose,
    )
    pkt_time_min_loaded = packet_df["timestamp"].min() if not packet_df.empty else None
    pkt_time_max_loaded = packet_df["timestamp"].max() if not packet_df.empty else None

    # Decide which time window to use.
    if args.use_pcap_range:
        start_ts = packet_df.attrs.get("pcap_time_min")
        stop_ts = packet_df.attrs.get("pcap_time_max")
        stop_is_exclusive = False
        if start_ts is None or stop_ts is None:
            raise ValueError("Could not determine a time range from the pcapng file.")
    else:
        start_ts, _ = _parse_datetime_arg(args.start, is_stop=False)
        stop_ts, stop_is_exclusive = _parse_datetime_arg(args.stop, is_stop=True)

    if args.verbose and (start_ts is not None or stop_ts is not None):
        source = "pcapng" if args.use_pcap_range else "command line"
        start_desc = _fmt_ts(start_ts)
        if stop_ts is None:
            stop_desc = "(none)"
        else:
            stop_desc = _fmt_ts(stop_ts) + (" (exclusive)" if stop_is_exclusive else "")
        print(f"Time filter ({source}): start={start_desc}, stop={stop_desc}")

    if start_ts is not None or stop_ts is not None:
        gps_before = len(gps_df)
        pkt_before = len(packet_df)
        gps_df = _apply_time_range_filter(
            gps_df,
            time_col=GpsColumns.timestamp,
            start=start_ts,
            stop=stop_ts,
            stop_is_exclusive=stop_is_exclusive,
        )
        packet_df = _apply_time_range_filter(
            packet_df,
            time_col="timestamp",
            start=start_ts,
            stop=stop_ts,
            stop_is_exclusive=stop_is_exclusive,
        )
        if args.verbose:
            print(f"GPS fixes after time filter: {len(gps_df)} / {gps_before}")
            print(f"Packets after time filter: {len(packet_df)} / {pkt_before}")

        if gps_df.empty:
            stop_note = " (exclusive)" if (stop_ts is not None and stop_is_exclusive) else ""
            raise ValueError(
                "Time filtering removed all GPS fixes, so GPS interpolation cannot run. "
                f"Loaded GPS range: {_fmt_range(gps_time_min_loaded, gps_time_max_loaded)} (n={gps_before}). "
                f"Loaded packet range: {_fmt_range(pkt_time_min_loaded, pkt_time_max_loaded)} (n={pkt_before}). "
                f"Filter window: start={_fmt_ts(start_ts)}, stop={_fmt_ts(stop_ts)}{stop_note}. "
                "This typically means the GPS and PCAP timestamps are in different timezones or timebases."
            )

    plot_packets_on_map(gps_df, packet_df, verbose=args.verbose)


if __name__ == "__main__":
    main()
