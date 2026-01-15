from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GPSPoint:
    timestamp: datetime
    latitude: float
    longitude: float
    altitude: Optional[float]


@dataclass(frozen=True)
class PacketRecord:
    timestamp: datetime
    session_id: str
    protocol: str
    device_address: Optional[str]
    rssi_dbm: Optional[float]
    channel: Optional[int]
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    payload: Dict[str, Any]
