"""Plot Bluetooth packet locations on a GPS track map."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
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


WEB_MERCATOR_RADIUS_M = 6378137.0
OSM_TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
DEFAULT_CONFIG_PATH = Path("plot_bluetooth_gps_map.json")


@dataclass(frozen=True)
class TrackStyle:
    color: str
    base_linewidth: float
    width_scale: float


@dataclass(frozen=True)
class DensityLineConfig:
    enabled: bool
    window: pd.Timedelta
    max_linewidth: float
    min_linewidth: float
    packet_types: Optional[list[str]]


def _lonlat_to_web_mercator(lon: float, lat: float) -> tuple[float, float]:
    lat_clamped = max(min(lat, 85.05112878), -85.05112878)
    x = WEB_MERCATOR_RADIUS_M * math.radians(lon)
    y = WEB_MERCATOR_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(lat_clamped) / 2.0))
    return x, y


def _tile_xy_from_lonlat(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat_clamped = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat_clamped)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _lonlat_from_tile_xy(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2.0**zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def _tile_bounds_mercator(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    lon_left, lat_top = _lonlat_from_tile_xy(x, y, z)
    lon_right, lat_bottom = _lonlat_from_tile_xy(x + 1, y + 1, z)
    x0, y0 = _lonlat_to_web_mercator(lon_left, lat_bottom)
    x1, y1 = _lonlat_to_web_mercator(lon_right, lat_top)
    return x0, y0, x1, y1


def _basemap_cache_key(
    *,
    provider: str,
    zoom: int,
    bounds_mercator: tuple[float, float, float, float],
) -> str:
    x0, y0, x1, y1 = bounds_mercator
    payload = f"{provider}|{zoom}|{x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _download_osm_tile(*, z: int, x: int, y: int, tile_path: Path):
    # Pillow is typically present via matplotlib, but keep it optional.
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: Pillow. Install with 'pip install pillow'.") from exc

    tile_path.parent.mkdir(parents=True, exist_ok=True)
    if tile_path.exists():
        return Image.open(tile_path).convert("RGBA")

    url = OSM_TILE_URL_TEMPLATE.format(z=z, x=x, y=y)
    req = Request(
        url,
        headers={
            "User-Agent": "wps-examples/plot_bluetooth_gps_map",
            "Accept": "image/png",
        },
    )
    with urlopen(req, timeout=30) as resp:
        data = resp.read()

    tile_path.write_bytes(data)
    return Image.open(tile_path).convert("RGBA")


def _choose_zoom_for_bounds(bounds_mercator: tuple[float, float, float, float]) -> int:
    x0, y0, x1, y1 = bounds_mercator
    width_m = max(1.0, abs(x1 - x0))
    height_m = max(1.0, abs(y1 - y0))
    max_dim_m = max(width_m, height_m)

    target_px = 1024.0
    initial_res = 156543.03392804062
    z = int(math.floor(math.log2((initial_res * target_px) / max_dim_m)))
    return max(0, min(19, z))


def _load_or_build_osm_basemap(
    *,
    bounds_mercator: tuple[float, float, float, float],
    zoom: Optional[int],
    cache_dir: Path,
):
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: Pillow. Install with 'pip install pillow'.") from exc

    z = int(zoom) if zoom is not None else _choose_zoom_for_bounds(bounds_mercator)
    provider = "osm"
    key = _basemap_cache_key(provider=provider, zoom=z, bounds_mercator=bounds_mercator)

    stitched_dir = cache_dir / "stitched"
    tiles_dir = cache_dir / "tiles"
    stitched_png = stitched_dir / f"{key}.png"
    stitched_json = stitched_dir / f"{key}.json"

    if stitched_png.exists() and stitched_json.exists():
        meta = json.loads(stitched_json.read_text(encoding="utf-8"))
        extent = tuple(meta["extent_mercator"])  # type: ignore[assignment]
        return Image.open(stitched_png).convert("RGBA"), extent, z

    # Convert desired bounds to lon/lat to compute required tile indices.
    # X is linear with lon in Web Mercator; for Y we use the inverse tile formula.
    x0, y0, x1, y1 = bounds_mercator
    lon_left = math.degrees(x0 / WEB_MERCATOR_RADIUS_M)
    lon_right = math.degrees(x1 / WEB_MERCATOR_RADIUS_M)

    def mercator_y_to_lat(y_m: float) -> float:
        return math.degrees(2.0 * math.atan(math.exp(y_m / WEB_MERCATOR_RADIUS_M)) - math.pi / 2.0)

    lat_bottom = mercator_y_to_lat(y0)
    lat_top = mercator_y_to_lat(y1)

    x_min_f, y_max_f = _tile_xy_from_lonlat(lon_left, lat_bottom, z)
    x_max_f, y_min_f = _tile_xy_from_lonlat(lon_right, lat_top, z)

    x_min = int(math.floor(min(x_min_f, x_max_f)))
    x_max = int(math.floor(max(x_min_f, x_max_f)))
    y_min = int(math.floor(min(y_min_f, y_max_f)))
    y_max = int(math.floor(max(y_min_f, y_max_f)))

    n = 2**z
    x_min = max(0, min(n - 1, x_min))
    x_max = max(0, min(n - 1, x_max))
    y_min = max(0, min(n - 1, y_min))
    y_max = max(0, min(n - 1, y_max))

    # Download/cache tiles and stitch.
    tile_size = 256
    width_tiles = x_max - x_min + 1
    height_tiles = y_max - y_min + 1
    stitched = Image.new("RGBA", (width_tiles * tile_size, height_tiles * tile_size))

    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            tile_path = tiles_dir / str(z) / str(tx) / f"{ty}.png"
            tile_img = _download_osm_tile(z=z, x=tx, y=ty, tile_path=tile_path)
            stitched.paste(tile_img, ((tx - x_min) * tile_size, (ty - y_min) * tile_size))

    # Compute stitched image extent in Web Mercator.
    left, _, _, top = _tile_bounds_mercator(z, x_min, y_min)
    _, bottom, right, _ = _tile_bounds_mercator(z, x_max, y_max)
    extent_merc = (left, right, bottom, top)

    stitched_dir.mkdir(parents=True, exist_ok=True)
    stitched.save(stitched_png)
    stitched_json.write_text(
        json.dumps(
            {
                "provider": provider,
                "zoom": z,
                "bounds_mercator": list(bounds_mercator),
                "extent_mercator": list(extent_merc),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return stitched, extent_merc, z


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


def _load_plot_config(path: Path, *, allow_missing: bool) -> dict[str, object]:
    if not path.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        config_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {path}") from exc

    if not isinstance(config_data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return config_data


def _parse_track_style(config: dict[str, object]) -> TrackStyle:
    color = str(config.get("line_color", "gray"))
    base_linewidth = float(config.get("line_base_width", 1.5))
    width_scale = float(config.get("line_width_scale", 1.0))
    return TrackStyle(color=color, base_linewidth=base_linewidth, width_scale=width_scale)


def _parse_density_config(config: dict[str, object], *, enabled: bool) -> DensityLineConfig:
    window_seconds = float(config.get("density_window_seconds", 30.0))
    if window_seconds <= 0:
        raise ValueError("density_window_seconds must be > 0")
    max_linewidth = float(config.get("density_max_linewidth", 6.0))
    min_linewidth = float(config.get("density_min_linewidth", 0.5))
    if max_linewidth <= 0 or min_linewidth <= 0:
        raise ValueError("density_max_linewidth and density_min_linewidth must be > 0")
    if min_linewidth > max_linewidth:
        raise ValueError("density_min_linewidth must be <= density_max_linewidth")
    packet_types = config.get("density_packet_types")
    if packet_types is not None:
        if not isinstance(packet_types, list):
            raise ValueError("density_packet_types must be a list of packet type strings")
        packet_types = [str(item) for item in packet_types]
    return DensityLineConfig(
        enabled=enabled,
        window=pd.Timedelta(seconds=window_seconds),
        max_linewidth=max_linewidth,
        min_linewidth=min_linewidth,
        packet_types=packet_types,
    )


def _count_packets_in_window(
    gps_times: pd.Series,
    packet_times: pd.Series,
    window: pd.Timedelta,
) -> np.ndarray:
    if packet_times.empty:
        return np.zeros(len(gps_times), dtype=float)

    gps_ns = gps_times.to_numpy(dtype="datetime64[ns]").astype("int64")
    packet_ns = packet_times.sort_values().to_numpy(dtype="datetime64[ns]").astype("int64")
    half_window = int(window.value / 2)
    left = gps_ns - half_window
    right = gps_ns + half_window
    left_idx = np.searchsorted(packet_ns, left, side="left")
    right_idx = np.searchsorted(packet_ns, right, side="right")
    return (right_idx - left_idx).astype(float)


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


def plot_packets_on_map(
    gps_df: pd.DataFrame,
    packet_df: pd.DataFrame,
    *,
    track_style: TrackStyle,
    density_config: DensityLineConfig,
    verbose: bool = False,
    show: bool = True,
) -> None:
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

    use_basemap = bool(packet_df.attrs.get("use_basemap"))
    basemap_provider = str(packet_df.attrs.get("basemap_provider") or "none")
    basemap_zoom = packet_df.attrs.get("basemap_zoom")
    basemap_cache_dir = Path(str(packet_df.attrs.get("basemap_cache_dir") or ".cache/basemaps"))

    if use_basemap and basemap_provider == "osm":
        lon_min = float(gps_df[GpsColumns.longitude].min())
        lon_max = float(gps_df[GpsColumns.longitude].max())
        lat_min = float(gps_df[GpsColumns.latitude].min())
        lat_max = float(gps_df[GpsColumns.latitude].max())

        x0, y0 = _lonlat_to_web_mercator(lon_min, lat_min)
        x1, y1 = _lonlat_to_web_mercator(lon_max, lat_max)
        left = min(x0, x1)
        right = max(x0, x1)
        bottom = min(y0, y1)
        top = max(y0, y1)
        pad_x = max(10.0, (right - left) * 0.05)
        pad_y = max(10.0, (top - bottom) * 0.05)
        bounds_merc = (left - pad_x, bottom - pad_y, right + pad_x, top + pad_y)

        basemap_img, extent_merc, z = _load_or_build_osm_basemap(
            bounds_mercator=bounds_merc,
            zoom=int(basemap_zoom) if basemap_zoom is not None else None,
            cache_dir=basemap_cache_dir,
        )
        if verbose:
            print(f"Basemap: OpenStreetMap tiles (zoom={z})")
            print(f"Basemap cache dir: {basemap_cache_dir}")

        ax.imshow(basemap_img, extent=extent_merc, origin="upper")

        # Plot data in Web Mercator to align with tiles.
        gps_x = []
        gps_y = []
        for lon, lat in zip(gps_df[GpsColumns.longitude].tolist(), gps_df[GpsColumns.latitude].tolist()):
            x, y = _lonlat_to_web_mercator(float(lon), float(lat))
            gps_x.append(x)
            gps_y.append(y)
        track_x = np.array(gps_x, dtype=float)
        track_y = np.array(gps_y, dtype=float)
    else:
        track_x = gps_df[GpsColumns.longitude].to_numpy(dtype=float)
        track_y = gps_df[GpsColumns.latitude].to_numpy(dtype=float)

    track_label = "GPS Track"
    if density_config.enabled:
        density_packet_df = packet_df
        if density_config.packet_types:
            density_packet_df = density_packet_df[
                density_packet_df["packet_type"].isin(density_config.packet_types)
            ]
        gps_times = gps_df[GpsColumns.timestamp]
        packet_times = density_packet_df["timestamp"] if not density_packet_df.empty else gps_times.iloc[:0]
        counts = _count_packets_in_window(gps_times, packet_times, density_config.window)
        if len(counts) > 1:
            segment_counts = (counts[:-1] + counts[1:]) / 2.0
        else:
            segment_counts = counts
        # Color the track by packet density (keep linewidth constant).
        # This makes density changes visible even when raw counts are high/low across the capture.
        cmap = plt.get_cmap("viridis")
        if len(segment_counts):
            vmin = float(np.min(segment_counts))
            vmax = float(np.max(segment_counts))
        else:
            vmin = 0.0
            vmax = 0.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax) if vmax > vmin else None
        if norm is None:
            segment_colors = np.tile(np.array(cmap(0.0)), (len(segment_counts), 1))
        else:
            segment_colors = cmap(norm(segment_counts))

        line_width = track_style.base_linewidth * track_style.width_scale
        if len(track_x) >= 2:
            points = np.column_stack([track_x, track_y])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=segment_colors,
                    linewidths=line_width,
                )
            )
            track_label = "GPS Track (density color)"
            ax.plot(
                [],
                [],
                color=cmap(1.0),
                linewidth=line_width,
                label=track_label,
            )
        else:
            ax.plot(
                track_x,
                track_y,
                color=track_style.color,
                linewidth=line_width,
                label=track_label,
            )
        if verbose:
            packet_type_desc = (
                ", ".join(density_config.packet_types)
                if density_config.packet_types
                else "all types"
            )
            max_count = int(np.max(segment_counts)) if len(segment_counts) else 0
            min_count = int(np.min(segment_counts)) if len(segment_counts) else 0
            print(
                "Density line enabled: "
                f"window={density_config.window}, types={packet_type_desc}, "
                f"count_range=[{min_count}, {max_count}], "
                f"mode=color (cmap=viridis, linewidth={line_width:g})"
            )
    else:
        ax.plot(
            track_x,
            track_y,
            color=track_style.color,
            linewidth=track_style.base_linewidth * track_style.width_scale,
            label=track_label,
        )

    types_present = sorted(events_with_locations["packet_type"].dropna().unique().tolist())
    if types_present == ["br_edr", "le"] or types_present == ["le", "br_edr"]:
        le_events = events_with_locations[events_with_locations["packet_type"] == "le"]
        br_events = events_with_locations[events_with_locations["packet_type"] == "br_edr"]

        if not le_events.empty:
            if use_basemap and basemap_provider == "osm":
                xs, ys = zip(
                    *(
                        _lonlat_to_web_mercator(float(lon), float(lat))
                        for lon, lat in zip(
                            le_events[GpsColumns.longitude].tolist(),
                            le_events[GpsColumns.latitude].tolist(),
                        )
                    )
                )
                ax.scatter(xs, ys, color="tab:blue", s=24, label="Bluetooth LE")
            else:
                ax.scatter(
                    le_events[GpsColumns.longitude],
                    le_events[GpsColumns.latitude],
                    color="tab:blue",
                    s=24,
                    label="Bluetooth LE",
                )

        if not br_events.empty:
            if use_basemap and basemap_provider == "osm":
                xs, ys = zip(
                    *(
                        _lonlat_to_web_mercator(float(lon), float(lat))
                        for lon, lat in zip(
                            br_events[GpsColumns.longitude].tolist(),
                            br_events[GpsColumns.latitude].tolist(),
                        )
                    )
                )
                ax.scatter(xs, ys, color="tab:orange", s=24, label="Bluetooth BR/EDR")
            else:
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
            if use_basemap and basemap_provider == "osm":
                xs, ys = zip(
                    *(
                        _lonlat_to_web_mercator(float(lon), float(lat))
                        for lon, lat in zip(
                            subset[GpsColumns.longitude].tolist(),
                            subset[GpsColumns.latitude].tolist(),
                        )
                    )
                )
                ax.scatter(
                    xs,
                    ys,
                    color=color_map.get(pkt_type, "tab:blue"),
                    s=24,
                    label=label,
                )
            else:
                ax.scatter(
                    subset[GpsColumns.longitude],
                    subset[GpsColumns.latitude],
                    color=color_map.get(pkt_type, "tab:blue"),
                    s=24,
                    label=label,
                )

    ax.set_title("Bluetooth Packet Locations")
    if use_basemap and basemap_provider == "osm":
        ax.set_xlabel("Web Mercator X (m)")
        ax.set_ylabel("Web Mercator Y (m)")
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if show:
        plt.show(block=True)


def plot_bluetooth_packet_counts_over_time(
    packet_df: pd.DataFrame,
    *,
    interval_seconds: float = 5.0,
    packet_types: Optional[list[str]] = None,
    verbose: bool = False,
    show: bool = True,
) -> None:
    """Plot bluetooth packet density over time as counts per fixed interval.

    - X axis: time (readable datetime)
    - Y axis: number of packets in each interval
    - Title: always includes interval length
    - Filtering: optional packet type allow-list
    """

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")

    if packet_df.empty:
        raise ValueError("No Bluetooth packets were found in the capture.")

    filtered = packet_df
    if packet_types:
        filtered = filtered[filtered["packet_type"].isin(packet_types)]

    if filtered.empty:
        raise ValueError("No Bluetooth packets matched the requested packet type filter(s).")

    ts = pd.to_datetime(filtered["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        raise ValueError("Packet timestamps could not be parsed.")

    # Bin timestamps into fixed-width intervals.
    interval = pd.Timedelta(seconds=float(interval_seconds))
    bins = ts.dt.floor(interval)
    counts = bins.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(counts.index.to_pydatetime(), counts.to_numpy(dtype=int), linestyle="-", marker="o", markersize=3)

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Packets / interval")

    interval_label = f"{int(interval_seconds)}s" if float(interval_seconds).is_integer() else f"{interval_seconds:g}s"
    filter_desc = "all types" if not packet_types else ", ".join(packet_types)
    ax.set_title(f"Bluetooth packets per {interval_label} interval ({filter_desc})")

    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    plt.tight_layout()

    if verbose:
        print(
            "Packet count plot: "
            f"interval={interval}, points={len(counts)}, total_packets={len(filtered)}"
        )

    if show:
        plt.show(block=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Bluetooth packet locations on a GPS track, interpolating packet timestamps."
        )
    )
    parser.add_argument("--pcapng", required=True, help="Path to the Bluetooth pcapng file.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "Optional JSON config file for plot styling. "
            f"Default: {DEFAULT_CONFIG_PATH} if present."
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
    config = _load_plot_config(
        config_path,
        allow_missing=config_path == DEFAULT_CONFIG_PATH,
    )
    track_style = _parse_track_style(config)
    density_config = _parse_density_config(config, enabled=bool(args.density_line))
    if args.verbose:
        if config_path.exists():
            print(f"Plot config loaded: {config_path}")
        elif config_path == DEFAULT_CONFIG_PATH:
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

    # Build both figures first so they open as separate windows.
    # Always run the packet-count plot with default args (5s interval, all types).
    plot_bluetooth_packet_counts_over_time(
        packet_df,
        interval_seconds=5.0,
        packet_types=None,
        verbose=args.verbose,
        show=False,
    )
    plot_packets_on_map(
        gps_df,
        packet_df,
        track_style=track_style,
        density_config=density_config,
        verbose=args.verbose,
        show=False,
    )
    plt.show(block=True)


if __name__ == "__main__":
    main()
