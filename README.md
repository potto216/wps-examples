# Overview
This is a collection of examples for automation tasks using [Teledyne LeCroy's Frontline Wireless Protocol Suite](https://www.teledynelecroy.com/support/softwaredownload/psgdocuments.aspx?standardid=2&mseries=671) software that is used for wireless technology capture as well as logic and wired serial protocols.

* `test_automation_le.ipynb` - Capture Bluetooth LE traffic using the WPS helper functions in [wpshelper](https://github.com/potto216/wpshelper)
* `test_automation_le_single_step.ipynb` - Step by step example capturing Bluetooth LE traffic without the WPS helper functions 

# Setup
After cloning the repository pull the submodule with:
```
git submodule update --init --recursive
Submodule 'wpshelper' (git@github.com:potto216/wpshelper.git) registered for path 'wpshelper'
Cloning into '/home/user/bluetooth/wps/wps_bluetooth_examples/wpshelper'...
Submodule path 'wpshelper': checked out 'a846e7f14a876227d6198bbca53e598942c73f3c'
```

To update wpshelper to the latest commit on main use
```
git submodule foreach git pull origin main
```