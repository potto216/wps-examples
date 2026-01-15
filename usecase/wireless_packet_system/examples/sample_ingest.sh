#!/usr/bin/env bash
set -euo pipefail

python -m wireless_packet_system.cli ingest \
  --config examples/example_config.yaml \
  --packets /data/captures/capture.json \
  --gps /data/gps/track.gpx \
  --session-id 2024-06-01
