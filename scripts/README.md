# Scripts

This directory contains standalone scripts that automate common WPS workflows.

## Available scripts

* `capture/wps_capture_cli.py`: start a WPS capture and stop on keypress.
* `capture/wps_cfa_to_cfax_cli.py`: convert `.cfa` captures to `.cfax`.
* `capture/wps_cfax_to_pcapng_cli.py`: export `.cfax` captures to `.pcapng`.
* `capture/wps_pcapng_to_parquet_cli.py`: convert `.pcapng` captures to packet-level parquet DataFrames.
* `analysis/wps_pcapng_analyze_cli.py`: analyze a Bluetooth LE `.pcapng` file into JSON.
* `analysis/wps_parquet_llm_summary_cli.py`: generate LLM-friendly capture summary and table catalog JSON from a parquet file.
* `matter/wps_matter_key_update_from_log.py`: parse Matter logs and update WPS keys during capture.
* `matter/wps_matter_key_update_from_logr2.py`: legacy Matter key updater with inline defaults.

## Prerequisites

* Python 3.8+
* The `wpshelper` submodule initialized (`git submodule update --init --recursive`).
* Teledyne LeCroy WPS installed for capture workflows.
* For capture automation, the WPS automation server (`FTSAutoServer.exe`) must be available.

## Capture: `capture/wps_capture_cli.py`

Starts a WPS capture, waits for a keypress, then stops and saves the capture.

**Key options**

* `--log-file` (required): where to write logs.
* `--data-path`: directory where `.cfax` files are saved.
* `--equipment`: `X240`, `X500`, or `X600`.
* `--le`, `--bredr`: enable/disable technologies.
* `--capture-technology`: provide a full `capturetechnology=...` string.
* `--auto-server-path`: path to `FTSAutoServer.exe` if not in the default WPS install.
* `--tcp-ip`, `--tcp-port`: automation server address (defaults to `127.0.0.1:22901`).

**Example**

```bash
python scripts/capture/wps_capture_cli.py \
  --equipment X500 \
  --log-file capture.log \
  --data-path "C:\\Users\\Public\\Documents\\Teledyne LeCroy Wireless\\My Capture Files" \
  --le \
  --bredr
```

Windows PowerShell:

```powershell
python scripts/capture/wps_capture_cli.py `
  --tcp-ip 127.0.0.1 `
  --equipment X240 `
  --log-file capture.log `
  --data-path "C:\\Users\\Public\\Documents\\Teledyne LeCroy Wireless\\My Capture Files" `
  --le `
  --bredr
```

To provide a custom capture technology string:

```bash
python scripts/capture/wps_capture_cli.py \
  --log-file capture.log \
  --capture-technology "capturetechnology=bredr-off|le-on|2m-on|spectrum-off|wifi-off|wpan-off" \
  --data-path "C:\\Users\\Public\\Documents\\Teledyne LeCroy Wireless\\My Capture Files"
```

Windows PowerShell:

```powershell
python scripts/capture/wps_capture_cli.py `
  --log-file capture.log `
  --capture-technology "capturetechnology=bredr-off|le-on|2m-on|spectrum-off|wifi-off|wpan-off" `
  --data-path "C:\\Users\\Public\\Documents\\Teledyne LeCroy Wireless\\My Capture Files"
```

## Matter keys: `matter/wps_matter_key_update_from_log.py`

Reads Matter log data (from serial or a file), extracts session keys and source node IDs, and updates WPS with those keys while capturing.

**Key options**

* `--serial_port`: serial port to read (default: `COM1`).
* `--baud_rate`: baud rate (default: `115200`).
* `--input_file`: read log data from a file instead of serial.
* `--update_file_name`: file that receives the JSON mapping updates.
* `--output_file_name`: raw log output file.

**Example (serial input)**

```bash
python scripts/matter/wps_matter_key_update_from_log.py \
  --serial_port COM4 \
  --baud_rate 115200 \
  --update_file_name matter_keys.jsonl \
  --output_file_name matter_serial.log
```

Windows PowerShell:

```powershell
python scripts/matter/wps_matter_key_update_from_log.py `
  --serial_port COM4 `
  --baud_rate 115200 `
  --update_file_name matter_keys.jsonl `
  --output_file_name matter_serial.log
```

**Example (file input)**

```bash
python scripts/matter/wps_matter_key_update_from_log.py \
  --input_file sample_matter.log \
  --update_file_name matter_keys.jsonl \
  --output_file_name matter_parsed.log
```

Windows PowerShell:

```powershell
python scripts/matter/wps_matter_key_update_from_log.py `
  --input_file sample_matter.log `
  --update_file_name matter_keys.jsonl `
  --output_file_name matter_parsed.log
```

## Convert CFA to CFAX: `capture/wps_cfa_to_cfax_cli.py`

Convert `.cfa` captures to `.cfax` using WPS automation.

**Example**

Windows PowerShell:

```powershell
python .\scripts\capture\wps_cfa_to_cfax_cli.py `
   --recursive `
   --skip-existing `
   C:\Users\potto\projects\tmp
```

## Convert CFAX to PCAPNG: `capture/wps_cfax_to_pcapng_cli.py`

Export `.cfax` captures to `.pcapng` with optional technology filters.

**Example**

```powershell
python scripts/capture/wps_cfax_to_pcapng_cli.py "D:\captures" --recursive --skip-existing --technology-filter LE
```

## Convert PCAPNG to parquet: `capture/wps_pcapng_to_parquet_cli.py`

Convert `.pcapng` captures into `.parquet` files using pandas with the `pyarrow` engine.
Each parquet row corresponds to one captured packet.

**Example**

```powershell
python scripts/capture/wps_pcapng_to_parquet_cli.py "D:\captures" --recursive --skip-existing
```

```bash
python scripts/capture/wps_pcapng_to_parquet_cli.py ~/data/captures/ --recursive --skip-existing
```
## Legacy Matter key updater: `matter/wps_matter_key_update_from_logr2.py`

Legacy variant that uses inline defaults for WPS paths, device configuration, and capture settings.

## Notes

* Defaults for WPS paths and automation server IPs differ by script; check the constants near the top of each file.

## PCAPNG analysis: `analysis/wps_pcapng_analyze_cli.py`

Analyze a Bluetooth LE `.pcapng` file using Scapy and emit a JSON report.

**Example**

```powershell
python scripts/analysis/wps_pcapng_analyze_cli.py "D:\path\capture.pcapng" -o report.json
```

## Parquet LLM summary: `analysis/wps_parquet_llm_summary_cli.py`

Build `CaptureSummary` and `TableCatalog` style artifacts from one parquet file, as described in the LLM-guided analyst design.

**Example**

```bash
python scripts/analysis/wps_parquet_llm_summary_cli.py data/sample.parquet --output-dir out/llm_summary
```

Use `--no-human-readable` to suppress stdout output, and omit `--output-dir` if you only want the terminal summary.
