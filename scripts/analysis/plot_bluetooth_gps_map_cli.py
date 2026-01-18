#!/usr/bin/env python3
"""Command-line interface for plotting Bluetooth packet locations on a GPS map."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Allow running this script directly from any working directory.
# (The repository root contains the top-level `analysis/` package.)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis import plot_bluetooth_gps_map as plot_map


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Bluetooth packet locations on a GPS track, interpolating packet timestamps."
        )
    )
    parser.add_argument("--pcapng", required=True, help="Path to the Bluetooth pcapng file.")
    parser.add_argument(
        "--config",
        default=str(plot_map.DEFAULT_CONFIG_PATH),
        help=(
            "Optional JSON config file for plot styling. "
            f"Default: {plot_map.DEFAULT_CONFIG_PATH} if present."
        ),
    )
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
        "--no-packets",
        action="store_true",
        help=(
            "Do not plot packet scatter points on the map (still uses packet timestamps for the density line)."
        ),
    )
    parser.add_argument(
        "--density-line",
        action="store_true",
        help=(
            "Scale the GPS track line width by Bluetooth packet density. "
            "Uses density_* settings from the JSON config."
        ),
    )
    parser.add_argument(
        "--basemap",
        default="none",
        choices=["none", "osm"],
        help=(
            "Optional basemap background. 'osm' uses OpenStreetMap tiles and caches downloads "
            "for faster repeat plots. Default: none."
        ),
    )
    parser.add_argument(
        "--basemap-zoom",
        type=int,
        help=(
            "Basemap zoom level (0-19). If omitted, a zoom is chosen automatically based on the plot area."
        ),
    )
    parser.add_argument(
        "--basemap-cache-dir",
        default=str(Path(".cache") / "basemaps"),
        help=(
            "Directory to cache basemap tiles and stitched images (default: .cache/basemaps)."
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

    config_path = Path(args.config)
    config = plot_map._load_plot_config(
        config_path,
        allow_missing=config_path == plot_map.DEFAULT_CONFIG_PATH,
    )
    track_style = plot_map._parse_track_style(config)
    density_config = plot_map._parse_density_config(config, enabled=bool(args.density_line))
    if args.verbose:
        if config_path.exists():
            print(f"Plot config loaded: {config_path}")
        elif config_path == plot_map.DEFAULT_CONFIG_PATH:
            print(f"Plot config not found (using defaults): {config_path}")

        # Raw config values (with defaults applied by the parse helpers).
        density_packet_types = config.get("density_packet_types")
        if isinstance(density_packet_types, list):
            density_packet_types_str = ", ".join(str(x) for x in density_packet_types)
        elif density_packet_types is None:
            density_packet_types_str = "(all types)"
        else:
            density_packet_types_str = str(density_packet_types)

        print(
            "Plot config parameters: "
            f"line_color={track_style.color}, "
            f"line_base_width={track_style.base_linewidth}, "
            f"line_width_scale={track_style.width_scale}"
        )
        print(
            "Density config parameters: "
            f"density_window_seconds={density_config.window.total_seconds():g}, "
            f"density_min_linewidth={density_config.min_linewidth:g}, "
            f"density_max_linewidth={density_config.max_linewidth:g}, "
            f"density_packet_types={density_packet_types_str}"
        )

    pcap_time_offset = plot_map._parse_time_offset(args.pcap_time_offset)

    gps_df = plot_map._load_gps_data(args.gps, args.gps_type, verbose=args.verbose)
    gps_time_min_loaded = gps_df[plot_map.GpsColumns.timestamp].min() if not gps_df.empty else None
    gps_time_max_loaded = gps_df[plot_map.GpsColumns.timestamp].max() if not gps_df.empty else None

    packet_df = plot_map._load_packet_events(
        args.pcapng,
        time_offset=pcap_time_offset,
        detailed_le_adv_types=bool(args.color_by_packet_type),
        verbose=args.verbose,
    )

    packet_df.attrs["use_basemap"] = args.basemap != "none"
    packet_df.attrs["basemap_provider"] = args.basemap
    packet_df.attrs["basemap_zoom"] = args.basemap_zoom
    packet_df.attrs["basemap_cache_dir"] = args.basemap_cache_dir
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
        start_ts, _ = plot_map._parse_datetime_arg(args.start, is_stop=False)
        stop_ts, stop_is_exclusive = plot_map._parse_datetime_arg(args.stop, is_stop=True)

    if args.verbose and (start_ts is not None or stop_ts is not None):
        source = "pcapng" if args.use_pcap_range else "command line"
        start_desc = plot_map._fmt_ts(start_ts)
        if stop_ts is None:
            stop_desc = "(none)"
        else:
            stop_desc = plot_map._fmt_ts(stop_ts) + (" (exclusive)" if stop_is_exclusive else "")
        print(f"Time filter ({source}): start={start_desc}, stop={stop_desc}")

    if start_ts is not None or stop_ts is not None:
        gps_before = len(gps_df)
        pkt_before = len(packet_df)
        gps_df = plot_map._apply_time_range_filter(
            gps_df,
            time_col=plot_map.GpsColumns.timestamp,
            start=start_ts,
            stop=stop_ts,
            stop_is_exclusive=stop_is_exclusive,
        )
        packet_df = plot_map._apply_time_range_filter(
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
                f"Loaded GPS range: {plot_map._fmt_range(gps_time_min_loaded, gps_time_max_loaded)} "
                f"(n={gps_before}). "
                f"Loaded packet range: {plot_map._fmt_range(pkt_time_min_loaded, pkt_time_max_loaded)} "
                f"(n={pkt_before}). "
                f"Filter window: start={plot_map._fmt_ts(start_ts)}, "
                f"stop={plot_map._fmt_ts(stop_ts)}{stop_note}. "
                "This typically means the GPS and PCAP timestamps are in different timezones or timebases."
            )

    # Build both figures first so they open as separate windows.
    # Always run the packet-count plot with default args (5s interval, all types).
    plot_map.plot_bluetooth_packet_counts_over_time(
        packet_df,
        interval_seconds=5.0,
        packet_types=None,
        verbose=args.verbose,
        show=False,
    )
    plot_map.plot_packets_on_map(
        gps_df,
        packet_df,
        track_style=track_style,
        density_config=density_config,
        show_packets=not bool(args.no_packets),
        verbose=args.verbose,
        show=False,
    )
    plt.show(block=True)


if __name__ == "__main__":
    main()
