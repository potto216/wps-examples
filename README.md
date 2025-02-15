# Overview

This repository is a collection of examples for automation tasks using [Teledyne LeCroy's Frontline Wireless Protocol Suite](https://www.teledynelecroy.com/support/softwaredownload/psgdocuments.aspx?standardid=2&mseries=671). This powerful suite is used for capturing wireless technology data as well as logic and wired serial protocols.

# Use Cases

- [Optimizing Bluetooth Audio Broadcasts with Python](usecase/README.md)


# Examples

- **`test_automation_le.ipynb`**  
  Capture Bluetooth LE traffic using the WPS helper functions available in [wpshelper](https://github.com/potto216/wpshelper).

- **`test_automation_le_single_step.ipynb`**  
  A step-by-step example demonstrating how to capture Bluetooth LE traffic without using the WPS helper functions.


# Setup

After cloning the repository, follow these steps to pull and configure the submodule:

```
git submodule update --init --recursive
Submodule 'wpshelper' (git@github.com:potto216/wpshelper.git) registered for path 'wpshelper'
Cloning into '/home/user/bluetooth/wps/wps_bluetooth_examples/wpshelper'...
Submodule path 'wpshelper': checked out 'a846e7f14a876227d6198bbca53e598942c73f3c'
```

To update wpshelper to the latest commit on main use

```
git submodule update --init --recursive
```

## Contributing

Contributions are welcome! If you have ideas or improvements, feel free to open an issue or submit a pull request.

## License

This project is open-source. See the [LICENSE](LICENSE.md) file for more details.
