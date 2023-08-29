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
Initialize the Submodule(s)

This sets up the necessary information about the submodule in .git/config.
```
git submodule init
```

Submodule initialization and update steps with:
```
git submodule update --init --recursive
```

If the wpshelper submodule is set to use the git protocol (e.g., git://github.com/user/repo.git) and you want to switch it to use the https protocol (e.g., https://github.com/user/repo.git), you'll need to modify the URL configuration for that submodule.

Here's how you can achieve this:

Edit the .gitmodules File

Open the .gitmodules file located in the root of your repository in a text editor. Find the section related to the submodule and modify its url property.

Change from:

[submodule "path/to/submodule"]
    path = path/to/submodule
    url = git://github.com/user/repo.git
To:

[submodule "path/to/submodule"]
    path = path/to/submodule
    url = https://github.com/user/repo.git
Save and close the .gitmodules file.

Update the Configuration of the Submodule

Run the following command to propagate the changes from .gitmodules to the git configuration:

```
git submodule sync
```

Update the Submodule. Now, you can pull the changes for the submodule using the https protocol:

```
git submodule update --recursive --remote
```
After these steps, the submodule should be set to use the https protocol for all future operations.