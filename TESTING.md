# Testing

This repository uses Python `unittest` tests under `tests/`.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activate the virtual environment:

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux (bash/zsh):

```bash
source .venv/bin/activate
```

## Run all tests

Windows (recommended; avoids the Microsoft Store `python` alias):

```powershell
py -m unittest discover -s tests
# or:
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

macOS/Linux:

```bash
python -m unittest discover -s tests
```

## Run a single test module

Windows:

```powershell
py -m unittest tests.test_pcapng_analysis
py -m unittest tests.test_wps_matter_key_update_from_log
```

macOS/Linux:

```bash
python -m unittest tests.test_pcapng_analysis
```

```bash
python -m unittest tests.test_wps_matter_key_update_from_log
```

## Notes

* The tests stub out optional dependencies (for example, `scapy` and `wpshelper`) so you can run them without hardware or WPS installed.
* If you are developing against live captures or WPS hardware, keep those environment-specific steps separate from the unit tests.
