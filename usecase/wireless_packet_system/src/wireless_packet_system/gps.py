from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

import numpy as np
from dateutil import parser as date_parser
import gpxpy
from fastkml import kml

from wireless_packet_system.models import GPSPoint


@dataclass(frozen=True)
class GPSSeries:
    timestamps: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    altitudes: np.ndarray

    def interpolate(self, epoch_seconds: float) -> GPSPoint:
        latitude = float(np.interp(epoch_seconds, self.timestamps, self.latitudes))
        longitude = float(np.interp(epoch_seconds, self.timestamps, self.longitudes))
        altitude = float(np.interp(epoch_seconds, self.timestamps, self.altitudes))
        timestamp = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        return GPSPoint(timestamp=timestamp, latitude=latitude, longitude=longitude, altitude=altitude)


def _parse_datetime(value: str) -> datetime:
    parsed = date_parser.parse(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _gps_points_to_series(points: Sequence[GPSPoint]) -> GPSSeries:
    timestamps = np.array([p.timestamp.timestamp() for p in points], dtype=float)
    latitudes = np.array([p.latitude for p in points], dtype=float)
    longitudes = np.array([p.longitude for p in points], dtype=float)
    altitudes = np.array([p.altitude if p.altitude is not None else 0.0 for p in points], dtype=float)
    return GPSSeries(timestamps=timestamps, latitudes=latitudes, longitudes=longitudes, altitudes=altitudes)


def load_gpx(file_path: str) -> GPSSeries:
    with open(file_path, "r", encoding="utf-8") as handle:
        gpx = gpxpy.parse(handle)

    points: List[GPSPoint] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time is None:
                    continue
                timestamp = point.time
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                else:
                    timestamp = timestamp.astimezone(timezone.utc)
                points.append(
                    GPSPoint(
                        timestamp=timestamp,
                        latitude=point.latitude,
                        longitude=point.longitude,
                        altitude=point.elevation,
                    )
                )

    if not points:
        raise ValueError("No GPS points found in GPX file.")

    return _gps_points_to_series(points)


def _extract_kml_points(feature: kml.KML) -> Iterable[GPSPoint]:
    for document in feature.features():
        for folder in document.features():
            for placemark in folder.features():
                geom = placemark.geometry
                if geom is None:
                    continue
                if geom.geom_type == "LineString":
                    coords = list(geom.coords)
                else:
                    coords = []
                times = placemark.times or []
                for idx, coord in enumerate(coords):
                    if idx >= len(times):
                        continue
                    lon, lat, *alt = coord
                    altitude = alt[0] if alt else None
                    timestamp = _parse_datetime(times[idx])
                    yield GPSPoint(timestamp=timestamp, latitude=lat, longitude=lon, altitude=altitude)


def load_kml(file_path: str) -> GPSSeries:
    with open(file_path, "rb") as handle:
        raw = handle.read()

    feature = kml.KML()
    feature.from_string(raw)
    points = list(_extract_kml_points(feature))

    if not points:
        raise ValueError("No GPS points found in KML file.")

    return _gps_points_to_series(points)


def load_gps(file_path: str) -> GPSSeries:
    if file_path.lower().endswith(".gpx"):
        return load_gpx(file_path)
    if file_path.lower().endswith(".kml"):
        return load_kml(file_path)
    raise ValueError("Unsupported GPS file type. Use .gpx or .kml.")
