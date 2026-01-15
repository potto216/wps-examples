from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import psycopg
from psycopg import sql
from psycopg import extras
from psycopg.rows import dict_row

from wireless_packet_system.models import PacketRecord


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str

    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.name} "
            f"user={self.user} password={self.password}"
        )


def connect(config: DatabaseConfig) -> psycopg.Connection:
    return psycopg.connect(config.dsn(), row_factory=dict_row)


def insert_session(
    connection: psycopg.Connection,
    session_id: str,
    protocol: str,
    capture_device: Optional[str],
    capture_start: Optional[str],
    capture_end: Optional[str],
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO packet_sessions (session_id, capture_start, capture_end, protocol, capture_device)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id)
            DO UPDATE SET capture_start = EXCLUDED.capture_start,
                          capture_end = EXCLUDED.capture_end,
                          protocol = EXCLUDED.protocol,
                          capture_device = EXCLUDED.capture_device
            """,
            (session_id, capture_start, capture_end, protocol, capture_device),
        )
    connection.commit()


def _packet_values(packet: PacketRecord) -> Tuple:
    return (
        packet.timestamp,
        packet.session_id,
        packet.protocol,
        packet.device_address,
        packet.rssi_dbm,
        packet.channel,
        packet.latitude,
        packet.longitude,
        packet.altitude,
        psycopg.types.json.Jsonb(packet.payload),
    )


def bulk_insert_packets(connection: psycopg.Connection, packets: Iterable[PacketRecord], batch_size: int = 1000) -> int:
    rows: List[Tuple] = []
    inserted = 0

    insert_sql = sql.SQL(
        """
        INSERT INTO bluetooth_packets (
            time, session_id, protocol, device_address, rssi_dbm, channel,
            latitude, longitude, altitude, packet_json
        ) VALUES %s
        """
    )

    with connection.cursor() as cursor:
        for packet in packets:
            rows.append(_packet_values(packet))
            if len(rows) >= batch_size:
                extras.execute_values(cursor, insert_sql, rows)
                inserted += len(rows)
                rows = []
        if rows:
            extras.execute_values(cursor, insert_sql, rows)
            inserted += len(rows)

    connection.commit()
    return inserted


def fetch_packet_counts(
    connection: psycopg.Connection,
    start_time: str,
    end_time: str,
    bucket_seconds: int,
) -> List[dict]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                time_bucket(%s * interval '1 second', time) AS bucket,
                COUNT(*) AS packet_count
            FROM bluetooth_packets
            WHERE time >= %s AND time < %s
            GROUP BY bucket
            ORDER BY bucket
            """,
            (bucket_seconds, start_time, end_time),
        )
        return list(cursor.fetchall())
