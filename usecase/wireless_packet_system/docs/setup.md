# Setup Guide (Ubuntu)

This guide covers installation of Python dependencies and Postgres + TimescaleDB for the wireless packet analysis system.

## 1) Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential libpq-dev
```

## 2) Install PostgreSQL + TimescaleDB

### 2.1 Install PostgreSQL

```bash
sudo apt-get install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql
```

### 2.2 Add the TimescaleDB repository

```bash
sudo apt-get install -y gnupg lsb-release wget

wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/timescaledb-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/timescaledb-archive-keyring.gpg] https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/timescaledb.list

sudo apt-get update
sudo apt-get install -y timescaledb-2-postgresql-14
```

> **Note:** If you use a different PostgreSQL major version, install the matching TimescaleDB package. For example, for PostgreSQL 15, install `timescaledb-2-postgresql-15`.

### 2.3 Configure TimescaleDB

```bash
sudo timescaledb-tune --quiet --yes
sudo systemctl restart postgresql
```

### 2.4 Enable the extension

```bash
sudo -u postgres psql -c "CREATE DATABASE packetdb;"
sudo -u postgres psql -d packetdb -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
```

## 3) Create a database user

```bash
sudo -u postgres psql -c "CREATE USER packet_user WITH PASSWORD 'packet_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE packetdb TO packet_user;"
```

## 4) Initialize the schema

```bash
psql -h localhost -U packet_user -d packetdb -f configs/schema.sql
```

## 5) Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 6) Configure the system

Copy and edit the example config:

```bash
cp examples/example_config.yaml examples/local_config.yaml
```

Update the values in `examples/local_config.yaml` to match your environment (database host/user/password, capture file paths, etc.).

## 7) Run ingestion

```bash
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python -m wireless_packet_system.cli ingest \
  --config examples/local_config.yaml \
  --packets /data/captures/capture.json \
  --gps /data/gps/track.gpx \
  --session-id 2024-06-01
```

## 8) Run analysis

```bash
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python -m wireless_packet_system.cli analyze \
  --config examples/local_config.yaml \
  --start "2024-06-01T00:00:00Z" \
  --end "2024-06-02T00:00:00Z"
```
