# Setup Guide (Ubuntu)

This guide covers installation of Python dependencies and Postgres + TimescaleDB for the wireless packet analysis system.

## 1) Install system dependencies

```bash
sudo apt update
sudo apt install -y build-essential libpq-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
# uv drops the binary in ~/.local/bin by default; ensure it’s on PATH:
uv python install
uv venv
```

## 2) Install PostgreSQL + TimescaleDB 
Instructions from https://www.tigerdata.com/docs/self-hosted/latest/install/installation-linux

### 2.1 Install PostgreSQL

```bash
sudo apt install -y gnupg postgresql-common apt-transport-https lsb-release wget
# Automated repository configuration:
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
sudo apt install postgresql-18
```

### 2.2 Add the TimescaleDB repository (https://github.com/timescale/timescaledb)

```bash
echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main" | sudo tee /etc/apt/sources.list.d/timescaledb.list

wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/timescaledb.gpg

sudo apt update

sudo apt install timescaledb-2-postgresql-18 postgresql-client-18
```

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

### 2.5 Troubleshooting (service is "active (exited)" / socket missing)

On Ubuntu/Debian, `postgresql.service` is a meta-unit that often shows **"active (exited)"**. The real server is a per-cluster unit (for example `postgresql@18-main`).

If `psql` fails with `No such file or directory` for `/var/run/postgresql/.s.PGSQL.5432`, check cluster status:

```bash
pg_lsclusters
```

Start the cluster you intend to use:

```bash
sudo systemctl start postgresql@18-main
sudo systemctl status postgresql@18-main --no-pager
sudo -u postgres psql
```

#### Fixing a major-version mismatch (example: cluster 18 points at a 16 data directory)

If `pg_lsclusters` shows something like:

```text
18  main  5432  down  postgres  /var/lib/postgresql/16/main
```

that means PostgreSQL 18 is configured to use a PostgreSQL 16 data directory. **A newer major version cannot start on an older major version data directory**; you must either create a fresh 18 cluster or upgrade/migrate the 16 data.

1) Inspect the error log to confirm:

```bash
sudo tail -n 200 /var/log/postgresql/postgresql-18-main.log
```

2) Choose one path:

- **Fresh install (OK to lose old DB data)**
  - Safest approach is to backup/move the old directory out of the way first so nothing gets deleted accidentally:

    ```bash
    sudo systemctl stop postgresql || true
    sudo mv /var/lib/postgresql/16/main \
      /var/lib/postgresql/16/main.bak-$(date +%F)
    ```

  - Then re-create a clean 18 cluster (exact commands vary by distro tooling; if your system uses `postgresql-common`, these usually work):

    ```bash
    sudo pg_dropcluster --stop 18 main
    sudo pg_createcluster 18 main --start
    ```

- **Keep existing DB data (upgrade/migrate 16 > 18)**
  - Recommended on Debian/Ubuntu is `pg_upgradecluster`:

    ```bash
    sudo apt install -y postgresql-16 postgresql-18
    sudo pg_upgradecluster 16 main
    ```

  - If you no longer have a registered `16 main` cluster (only the data directory remains), you’ll need to restore the PostgreSQL 16 cluster configuration or migrate from a backup/dump; do **not** point PostgreSQL 18 at the old directory.

## 3) Create a database user

```bash
sudo -u postgres psql -c "CREATE USER packet_user WITH PASSWORD 'password';"
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
