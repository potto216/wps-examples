# Testing

This repository uses Python `unittest` tests under `tests/`.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run all tests

```bash
python -m unittest discover -s tests
```

## Run a single test module

```bash
python -m unittest tests.test_pcapng_analysis
```

```bash
python -m unittest tests.test_wps_matter_key_update_from_log
```

## Notes

* The tests stub out optional dependencies (for example, `scapy` and `wpshelper`) so you can run them without hardware or WPS installed.
* If you are developing against live captures or WPS hardware, keep those environment-specific steps separate from the unit tests.
