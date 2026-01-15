# Wireless Packet Analysis System

A modular Python project for ingesting Wireshark JSON packet captures, enriching them with GPS coordinates, storing them in TimescaleDB, and running batch analytics (including anomaly detection). The system is designed so that protocol-specific parsing (Bluetooth today, Zigbee/802.15.4 and Wi-Fi later) can be swapped without rewriting the pipeline.

## Project layout

```
usecase/wireless_packet_system/
├── configs/
│   └── schema.sql
├── docs/
│   └── setup.md
├── examples/
│   ├── example_config.yaml
│   └── sample_ingest.sh
├── src/
│   └── wireless_packet_system/
│       ├── __init__.py
│       ├── analysis.py
│       ├── cli.py
│       ├── db.py
│       ├── gps.py
│       ├── ingestion.py
│       ├── llm_orchestrator.py
│       └── models.py
└── requirements.txt
```

## Quick start

1. Follow the setup instructions to install Python dependencies and configure Postgres + TimescaleDB: [docs/setup.md](docs/setup.md)
2. Create a database and run the schema:

```bash
psql -h localhost -U packet_user -d packetdb -f configs/schema.sql
```

3. Update `examples/example_config.yaml` with your database credentials and file paths.
4. Ensure the package is on your path (from the project root):

```bash
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
```

5. Ingest a JSON capture and GPS track:

```bash
python -m wireless_packet_system.cli ingest \
  --config examples/example_config.yaml \
  --packets path/to/capture.json \
  --gps path/to/track.gpx \
  --session-id 2024-06-01
```

6. Run a simple anomaly analysis:

```bash
python -m wireless_packet_system.cli analyze \
  --config examples/example_config.yaml \
  --start "2024-06-01T00:00:00Z" \
  --end "2024-06-02T00:00:00Z"
```

## Notes

- Ingestion expects Wireshark/TShark JSON with `frame.time_epoch`. Other protocols can be added by implementing a parser in `ingestion.py`.
- The database stores a normalized set of columns plus a JSONB payload for protocol-specific fields.
- LLM orchestration is deliberately conservative: it generates SQL and analysis steps, but you control execution.
