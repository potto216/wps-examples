CREATE TABLE IF NOT EXISTS packet_sessions (
    session_id TEXT PRIMARY KEY,
    capture_start TIMESTAMPTZ,
    capture_end TIMESTAMPTZ,
    protocol TEXT NOT NULL,
    capture_device TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS bluetooth_packets (
    time TIMESTAMPTZ NOT NULL,
    session_id TEXT NOT NULL REFERENCES packet_sessions(session_id),
    protocol TEXT NOT NULL,
    device_address TEXT,
    rssi_dbm DOUBLE PRECISION,
    channel INTEGER,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    altitude DOUBLE PRECISION,
    packet_json JSONB
);

SELECT create_hypertable('bluetooth_packets', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS bluetooth_packets_device_time_idx
    ON bluetooth_packets (device_address, time DESC);

CREATE INDEX IF NOT EXISTS bluetooth_packets_location_idx
    ON bluetooth_packets (latitude, longitude);
