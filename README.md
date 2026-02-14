# Overview

This repository is a collection of examples for automation tasks using [Teledyne LeCroy's Frontline Wireless Protocol Suite](https://www.teledynelecroy.com/support/softwaredownload/psgdocuments.aspx?standardid=2&mseries=671). This suite is used for capturing wireless technology data as well as logic and wired serial protocols.

# Use Cases

- [Optimizing Bluetooth Audio Broadcasts with Python](usecase/README.md)


# Examples

Notebooks live under `notebooks/` grouped by workflow (capture, analysis, audio, spectrum).

- **`notebooks/capture/test_automation_le.ipynb`**  
  Capture Bluetooth LE traffic using the WPS helper functions available in [wpshelper](https://github.com/potto216/wpshelper).

- **`notebooks/capture/test_automation_le_single_step.ipynb`**  
  A step-by-step example demonstrating how to capture Bluetooth LE traffic without using the WPS helper functions.

- **`analysis/pcapng_analysis.py`**  
  JSON-centric Bluetooth LE pcapng analysis helpers that can be reused in notebooks such as
  `notebooks/analysis/show_le_pcapng.ipynb` and in scripts under `analysis/`. See `analysis/README.md` for usage
  and schema details.

## Scripts

Standalone automation scripts are available under `scripts/`:

* `scripts/capture/wps_capture_cli.py`: start a WPS capture and stop on keypress.
* `scripts/capture/wps_cfa_to_cfax_cli.py`: convert `.cfa` captures to `.cfax`.
* `scripts/capture/wps_cfax_to_pcapng_cli.py`: export `.cfax` captures to `.pcapng`.
* `scripts/analysis/wps_pcapng_analyze_cli.py`: analyze a Bluetooth LE `.pcapng` file into JSON.
* `scripts/matter/wps_matter_key_update_from_log.py`: parse Matter logs and update WPS keys during capture.
* `scripts/matter/wps_matter_key_update_from_logr2.py`: legacy Matter key updater with inline defaults.

See `scripts/README.md` for details and usage examples.

## Shared Python code (`lib/`)

The repository-level `lib/` directory is the home for shared Python helpers used by scripts and notebooks.
See [`lib/README.md`](lib/README.md) for naming, layout, and dependency-management recommendations.

# Setup
## Python

## Repository 
Clone the repository with
```
git clone --recursive https://github.com/potto216/wps-examples.git
```
To include the submodules. If you have problems the following may be helpful

After cloning the repository, follow these steps to pull the wpshelper submodule to ensure it matches the exact version tracked:
```
git submodule update --init --recursive
Submodule 'wpshelper' (git@github.com:potto216/wpshelper.git) registered for path 'wpshelper'
Cloning into '/home/user/bluetooth/wps/wps_bluetooth_examples/wpshelper'...
Submodule path 'wpshelper': checked out 'a846e7f14a876227d6198bbca53e598942c73f3c'
```

After initialization to update wpshelper to the latest commit used by main use
```
git submodule update --init --recursive
```

If you need to use the latest version of wpshelper from its remote repository rather than the specifically pinned versions (for example you are going to make a contribution) then use:
```
# use sync if making changes to .gitmodules
git submodule sync
git submodule update --recursive --remote
```
## Notes
To determine the `scapy` version and see if the needed bluetooth classes are used run:

```
python -c "import scapy; import scapy.layers.bluetooth4LE as b; print('scapy', scapy.__version__); names=[n for n in dir(b) if n.startswith('BTLE')]; print('BTLE symbols sample:', sorted(names)[:80])"

python.exe -c "import scapy.layers.bluetooth4LE as b; print('BTLE_ADV fields', [f.name for f in getattr(b.BTLE_ADV,'fields_desc',[])][:20]); print('BTLE_RF fields', [f.name for f in getattr(b.BTLE_RF,'fields_desc',[])][:30]); print('BTLE_PPI fields', [f.name for f in getattr(b.BTLE_PPI,'fields_desc',[])][:30])"
```

# Contributing

Contributions are welcome! If you have ideas or improvements, feel free to open an issue or submit a pull request.

# License

This project is open-source. See the [LICENSE](LICENSE.md) file for more details.
