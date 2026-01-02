#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from typing import Optional

module_path = os.path.abspath(os.path.join("./wpshelper"))
if module_path not in sys.path:
    sys.path.append(module_path)

from wpshelper import (
    wps_open,
    wps_configure,
    wps_start_record,
    wps_stop_record,
    wps_analyze_capture,
    wps_save_capture,
    wps_close,
    wps_find_installations,
)

# Default to localhost because FTSAutoServer.exe is started on this machine.
# Override with --tcp-ip if your automation server is remote or bound to a
# specific interface.
TCP_IP = "127.0.0.1"
TCP_PORT = 22901
MAX_TO_READ = 1000
SLEEP_TIME = 2
MAX_WAIT_TIME = 60

DEFAULT_WPS_BASE_DIR = r"C:\Program Files (x86)\Teledyne LeCroy Wireless"
DEFAULT_DATA_PATH = r"C:\Users\Public\Documents\Teledyne LeCroy Wireless\My Capture Files"

LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}


def build_capture_technology(args: argparse.Namespace) -> str:
    if args.capture_technology:
        return args.capture_technology

    le_state = "le-on" if args.le else "le-off"
    bredr_state = "bredr-on" if args.bredr else "bredr-off"
    spectrum_state = "spectrum-on" if args.spectrum else "spectrum-off"
    spectrum_value = spectrum_state
    if args.spectrum and args.spectrum_interval:
        spectrum_value = f"{spectrum_state};interval={args.spectrum_interval}"

    return (
        "capturetechnology="
        f"{bredr_state}|{le_state}|2m-on|"
        f"{spectrum_value}|wifi-off|wpan-off"
    )


def normalize_filename_piece(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def capture_technology_label(capture_technology: str) -> str:
    if capture_technology.startswith("capturetechnology="):
        capture_technology = capture_technology.split("=", 1)[1]
    entries = []
    for segment in capture_technology.split("|"):
        segment = segment.split(";", 1)[0]
        if segment.endswith("-on"):
            entries.append(segment.replace("-on", ""))
    if not entries:
        entries = ["capture"]
    return normalize_filename_piece("_".join(entries))


def make_capture_filename(
    prefix: str,
    equipment: str,
    capture_technology: str,
    data_path: str,
    extension: str,
) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tech_label = capture_technology_label(capture_technology)
    pieces = [normalize_filename_piece(prefix), equipment.lower(), tech_label, timestamp]
    pieces = [piece for piece in pieces if piece]
    filename = "_".join(pieces) + extension
    return os.path.join(data_path, filename)


def start_server(auto_server_path: str, logger: logging.Logger) -> Optional[subprocess.Popen]:
    if not os.path.exists(auto_server_path):
        raise FileNotFoundError(
            f"Auto server executable not found at {auto_server_path}. "
            "Set --auto-server-path to the correct location."
        )
    logger.info("Starting automation server: %s", auto_server_path)
    process = subprocess.Popen([auto_server_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    return process


def wait_for_keypress(stop_key: str, logger: logging.Logger) -> None:
    logger.info("Waiting for key '%s' to stop capture.", stop_key)
    stop_key = stop_key.lower()
    try:
        import msvcrt

        while True:
            if msvcrt.kbhit():
                char = msvcrt.getwch().lower()
                if char == stop_key:
                    return
            time.sleep(0.1)
    except ImportError:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if ready:
                    char = sys.stdin.read(1).lower()
                    if char == stop_key:
                        return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def request_console_focus() -> None:
    if os.name != "nt":
        return
    import ctypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    hwnd = kernel32.GetConsoleWindow()
    if not hwnd:
        raise RuntimeError("Console window handle not found.")
    if hasattr(user32, "AllowSetForegroundWindow"):
        user32.AllowSetForegroundWindow(0xFFFFFFFF)
    user32.ShowWindow(hwnd, 5)
    if not user32.SetForegroundWindow(hwnd):
        raise RuntimeError("Failed to set console window foreground.")


def configure_logging(log_level: str, log_file: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("wps_capture_cli")


def validate_args(args: argparse.Namespace) -> None:
    if args.log_level not in LOG_LEVELS:
        raise ValueError(f"Invalid log level '{args.log_level}'. Expected one of {sorted(LOG_LEVELS)}.")
    if not args.log_file:
        raise ValueError("--log-file is required.")
    if args.stop_key and len(args.stop_key) != 1:
        raise ValueError("--stop-key must be a single character.")
    if args.capture_technology and not args.capture_technology.startswith("capturetechnology="):
        raise ValueError("--capture-technology must start with 'capturetechnology='.")
    if args.spectrum_interval is not None and args.spectrum_interval <= 0:
        raise ValueError("--spectrum-interval must be a positive integer.")
    if not (args.le or args.bredr or args.capture_technology):
        raise ValueError("Enable at least one of --le or --bredr (or provide --capture-technology).")
    if not os.path.isdir(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")
    if not args.tcp_ip or not str(args.tcp_ip).strip():
        raise ValueError("--tcp-ip must be a non-empty string.")
    if not (1 <= args.tcp_port <= 65535):
        raise ValueError("--tcp-port must be in range 1..65535.")
    if args.sleep_time <= 0:
        raise ValueError("--sleep-time must be a positive number of seconds.")
    if args.max_wait_time <= 0:
        raise ValueError("--max-wait-time must be a positive number of seconds.")


def main() -> None:
    wps_installations = wps_find_installations(base_dir=DEFAULT_WPS_BASE_DIR, show_log=False)
    latest_install = (wps_installations or {}).get("latest") or {}
    
    # Keep the constant in sync for any downstream code that references it.
    global DEFAULT_WPS_PATH
    DEFAULT_WPS_PATH = r"C:\Program Files (x86)\Teledyne LeCroy Wireless\Wireless Protocol Suite 4.50"
    default_wps_path = latest_install.get("path") or DEFAULT_WPS_PATH

    DEFAULT_WPS_PATH = default_wps_path

    parser = argparse.ArgumentParser(
        description="Start a WPS capture for Bluetooth LE and BR/EDR and stop on keypress."
    )
    parser.add_argument("--equipment", choices=["X240", "X500", "X600"], default="X240")
    parser.add_argument("--prefix", default="", help="Prefix for the capture filename.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Directory to store capture files.")
    parser.add_argument("--capture-technology", help="Full capturetechnology string to pass to WPS.")
    parser.add_argument("--le", dest="le", action="store_true", default=True, help="Enable LE capture.")
    parser.add_argument("--no-le", dest="le", action="store_false", help="Disable LE capture.")
    parser.add_argument("--bredr", dest="bredr", action="store_true", default=True, help="Enable BR/EDR capture.")
    parser.add_argument("--no-bredr", dest="bredr", action="store_false", help="Disable BR/EDR capture.")
    parser.add_argument("--spectrum", action="store_true", help="Enable spectrum capture.")
    parser.add_argument("--spectrum-interval", type=int, help="Spectrum interval in ms when enabled.")
    parser.add_argument("--log-level", default="info", help="Log level (debug, info, warning, error, critical).")
    parser.add_argument("--log-file", required=True, help="Path to the log file.")
    parser.add_argument("--stop-key", default="q", help="Key to stop the capture.")
    parser.add_argument("--auto-server-path", help="Path to FTSAutoServer.exe.")
    parser.add_argument(
        "--wps-path",
        default=DEFAULT_WPS_PATH,
        help="Base path for WPS install (defaults to latest detected installation when available).",
    )
    parser.add_argument("--extension", default=".cfax", help="Capture file extension (default: .cfax).")
    parser.add_argument(
        "--tcp-ip",
        default=TCP_IP,
        help="Automation server IP address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=TCP_PORT,
        help=f"Automation server port (default: {TCP_PORT}).",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=SLEEP_TIME,
        help=f"Socket timeout / poll interval in seconds (default: {SLEEP_TIME}).",
    )
    parser.add_argument(
        "--max-wait-time",
        type=float,
        default=MAX_WAIT_TIME,
        help=f"Max seconds to wait for WPS initialization (default: {MAX_WAIT_TIME}).",
    )

    args = parser.parse_args()
    args.log_level = args.log_level.lower()

    validate_args(args)
    logger = configure_logging(args.log_level, args.log_file)

    logger.info("WPS installations scan (base_dir=%s):\n%s", DEFAULT_WPS_BASE_DIR, json.dumps(wps_installations, indent=2, sort_keys=True))
    logger.info("Default WPS path selected: %s", DEFAULT_WPS_PATH)

    capture_technology = build_capture_technology(args)
    capture_file = make_capture_filename(
        args.prefix, args.equipment, capture_technology, args.data_path, args.extension
    )

    wps_executable_path = os.path.join(args.wps_path, "Executables", "Core")
    auto_server_path = args.auto_server_path or os.path.join(wps_executable_path, "FTSAutoServer.exe")

    effective_config = {
        "args": vars(args),
        "derived": {
            "capture_technology": capture_technology,
            "capture_file": capture_file,
            "wps_executable_path": wps_executable_path,
            "auto_server_path": auto_server_path,
            "max_to_read": MAX_TO_READ,
            "wps_installations": wps_installations,
        },
    }
    logger.info("Effective configuration:\n%s", json.dumps(effective_config, indent=2, sort_keys=True))
    logger.info("Capture file will be saved to %s", capture_file)

    server_process = None
    try:
        server_process = start_server(auto_server_path, logger)
        wps_handle = wps_open(
            tcp_ip=args.tcp_ip,
            tcp_port=args.tcp_port,
            max_to_read=MAX_TO_READ,
            wps_executable_path=wps_executable_path,
            personality_key=args.equipment,
            sleep_time=args.sleep_time,
            max_wait_time=args.max_wait_time,
        )
        if args.equipment.upper() == "X240":
            logger.warning(
                "X240 capture technology may need to be configured on the device before automation."
            )
        wps_configure(wps_handle, args.equipment, capture_technology, show_log=True)
        wps_start_record(wps_handle, show_log=True)
        logger.info("Capture started. Press '%s' to stop.", args.stop_key)
        try:
            request_console_focus()
        except Exception as exc:
            logger.warning("Unable to request console focus: %s", exc)
        wait_for_keypress(args.stop_key, logger)
        logger.info("Stop key pressed. Stopping capture.")
        wps_stop_record(wps_handle, show_log=True)
        wps_analyze_capture(wps_handle,show_log=True)
        wps_save_capture(wps_handle, capture_file,show_log=True)
        wps_close(wps_handle, recv_retry_attempts=10, recv_retry_sleep=4,show_log=True)
        logger.info("Capture saved successfully.")
    except Exception as exc:
        logger.exception("Capture failed: %s", exc)
        raise
    finally:
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
                logger.info("Automation server terminated gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning("Automation server did not terminate in time, killing it.")
                server_process.kill()
        
        # Also kill any lingering WPS GUI processes
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "fts.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            logger.info("WPS GUI process terminated.")
        except Exception as exc:
            logger.debug("Could not kill WPS GUI (may not be running): %s", exc)


if __name__ == "__main__":
    main()
