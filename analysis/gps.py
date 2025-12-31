"""GPS data helpers for WPS examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclass(frozen=True)
class GpsColumns:
    """Column names used by the GPS helpers."""

    latitude: str = "lat"
    longitude: str = "lon"
    altitude: str = "alt"
    timestamp: str = "timestamp"
    speed: str = "speed"
    satellites: str = "sat"


def _load_xml(source: str | bytes | Iterable[bytes]) -> ET.Element:
    if isinstance(source, (str, bytes)):
        tree = ET.parse(source)
        return tree.getroot()
    return ET.fromstring(b"".join(source))


def load_kml(path: str) -> pd.DataFrame:
    """Load GPS coordinates from a KML file.

    Args:
        path: Path to the KML file.

    Returns:
        DataFrame with columns: lon, lat, alt.
    """
    root = _load_xml(path)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    coordinates_blocks = root.findall(".//kml:LineString/kml:coordinates", ns)

    rows: list[dict[str, float]] = []
    for block in coordinates_blocks:
        if block.text is None:
            continue
        for entry in block.text.strip().split():
            parts = entry.split(",")
            if len(parts) < 2:
                continue
            lon = float(parts[0])
            lat = float(parts[1])
            alt = float(parts[2]) if len(parts) > 2 and parts[2] else np.nan
            rows.append({"lon": lon, "lat": lat, "alt": alt})

    return pd.DataFrame(rows)


def load_gpx(path: str) -> pd.DataFrame:
    """Load GPS trackpoints from a GPX file.

    Args:
        path: Path to the GPX file.

    Returns:
        DataFrame with columns: lat, lon, ele, timestamp, speed, sat.
    """
    root = _load_xml(path)
    ns = {"gpx": "http://www.topografix.com/GPX/1/0"}

    rows: list[dict[str, object]] = []
    for trkpt in root.findall(".//gpx:trkpt", ns):
        lat = trkpt.attrib.get("lat")
        lon = trkpt.attrib.get("lon")
        ele = trkpt.findtext("gpx:ele", default=None, namespaces=ns)
        time_text = trkpt.findtext("gpx:time", default=None, namespaces=ns)
        speed = trkpt.findtext("gpx:speed", default=None, namespaces=ns)
        sat = trkpt.findtext("gpx:sat", default=None, namespaces=ns)

        rows.append(
            {
                "lat": float(lat) if lat is not None else np.nan,
                "lon": float(lon) if lon is not None else np.nan,
                "ele": float(ele) if ele is not None else np.nan,
                "timestamp": pd.to_datetime(time_text, utc=True) if time_text else pd.NaT,
                "speed": float(speed) if speed is not None else np.nan,
                "sat": float(sat) if sat is not None else np.nan,
            }
        )

    return pd.DataFrame(rows)


def interpolate_gps_events(
    gps_df: pd.DataFrame,
    events_df: pd.DataFrame,
    gps_time_col: str = "timestamp",
    event_time_col: str = "timestamp",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Interpolate GPS coordinates for event timestamps.

    Args:
        gps_df: DataFrame with GPS fixes.
        events_df: DataFrame with event timestamps.
        gps_time_col: Column in gps_df with timestamps.
        event_time_col: Column in events_df with timestamps.
        lat_col: Latitude column in gps_df.
        lon_col: Longitude column in gps_df.

    Returns:
        Events DataFrame copy with interpolated lat/lon columns added.
    """
    if gps_time_col not in gps_df.columns:
        raise ValueError(f"gps_df missing required column: {gps_time_col}")
    if event_time_col not in events_df.columns:
        raise ValueError(f"events_df missing required column: {event_time_col}")

    gps_sorted = (
        gps_df[[gps_time_col, lat_col, lon_col]]
        .dropna()
        .sort_values(gps_time_col)
        .drop_duplicates(subset=gps_time_col)
    )
    if gps_sorted.empty:
        raise ValueError("gps_df must contain at least one valid GPS fix with timestamps")

    gps_times = pd.to_datetime(gps_sorted[gps_time_col], utc=True)
    gps_ns = gps_times.view("int64").to_numpy()
    lat_values = gps_sorted[lat_col].to_numpy()
    lon_values = gps_sorted[lon_col].to_numpy()

    events = events_df.copy()
    event_times = pd.to_datetime(events[event_time_col], utc=True)
    event_ns = event_times.view("int64").to_numpy()

    lat_interp = np.interp(event_ns, gps_ns, lat_values)
    lon_interp = np.interp(event_ns, gps_ns, lon_values)

    out_of_range = (event_ns < gps_ns.min()) | (event_ns > gps_ns.max()) | event_times.isna()
    lat_interp[out_of_range] = np.nan
    lon_interp[out_of_range] = np.nan

    events[lat_col] = lat_interp
    events[lon_col] = lon_interp
    return events


def plot_gps_path_map(
    gps_df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    title: str = "GPS Path",
) -> go.Figure:
    """Plot GPS path on an OpenStreetMap tile layer."""
    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat=gps_df[lat_col],
            lon=gps_df[lon_col],
            mode="lines+markers",
            marker={"size": 6},
            name="GPS Path",
        )
    )
    fig.update_layout(
        mapbox={"style": "open-street-map", "zoom": 12},
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        title=title,
    )
    return fig


def plot_gps_with_events_map(
    gps_df: pd.DataFrame,
    events_df: pd.DataFrame,
    event_label_col: Optional[str] = None,
    gps_time_col: str = "timestamp",
    event_time_col: str = "timestamp",
    lat_col: str = "lat",
    lon_col: str = "lon",
    title: str = "GPS Path with Events",
) -> go.Figure:
    """Plot GPS path with interpolated event markers on a map."""
    events_with_locations = interpolate_gps_events(
        gps_df,
        events_df,
        gps_time_col=gps_time_col,
        event_time_col=event_time_col,
        lat_col=lat_col,
        lon_col=lon_col,
    )

    fig = plot_gps_path_map(gps_df, lat_col=lat_col, lon_col=lon_col, title=title)

    text = (
        events_with_locations[event_label_col]
        if event_label_col and event_label_col in events_with_locations.columns
        else events_with_locations[event_time_col].astype(str)
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=events_with_locations[lat_col],
            lon=events_with_locations[lon_col],
            mode="markers",
            marker={"size": 10, "color": "red"},
            name="Events",
            text=text,
        )
    )
    return fig
