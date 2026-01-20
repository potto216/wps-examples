#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, Iterable, Optional, Tuple

module_path = os.path.abspath(os.path.join("./wpshelper"))
if module_path not in sys.path:
    sys.path.append(module_path)

from wpshelper import (
    wps_close,
    wps_export_pcapng,
    wps_find_installations,
    wps_open,
    wps_open_capture,
    wps_wireless_devices,
    wps_analyze_capture,
    wps_close_capture
)

TCP_IP = "127.0.0.1"
TCP_PORT = 22901
MAX_TO_READ = 1000
SLEEP_TIME = 2
MAX_WAIT_TIME = 60

DEFAULT_WPS_BASE_DIR = r"C:\Program Files (x86)\Teledyne LeCroy Wireless"
DEFAULT_WPS_PATH = r"C:\Program Files (x86)\Teledyne LeCroy Wireless\Wireless Protocol Suite 4.50"

LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}


def configure_logging(log_level: str, log_file: Optional[str]) -> logging.Logger:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("wps_cfax_to_pcapng_cli")


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


def validate_args(args: argparse.Namespace) -> None:
    if args.log_level not in LOG_LEVELS:
        raise ValueError(f"Invalid log level '{args.log_level}'. Expected one of {sorted(LOG_LEVELS)}.")
    if not args.input_path:
        raise ValueError("input_path is required.")
    if not os.path.exists(args.input_path):
        raise ValueError(f"Path does not exist: {args.input_path}")
    if not args.tcp_ip or not str(args.tcp_ip).strip():
        raise ValueError("--tcp-ip must be a non-empty string.")
    if not (1 <= args.tcp_port <= 65535):
        raise ValueError("--tcp-port must be in range 1..65535.")
    if args.sleep_time <= 0:
        raise ValueError("--sleep-time must be a positive number of seconds.")
    if args.max_wait_time <= 0:
        raise ValueError("--max-wait-time must be a positive number of seconds.")


def iter_cfax_files(path: str, recursive: bool) -> Iterable[str]:
    if os.path.isfile(path):
        if path.lower().endswith(".cfax"):
            yield path
        return
    if recursive:
        for root, _, files in os.walk(path):
            for name in files:
                if name.lower().endswith(".cfax"):
                    yield os.path.join(root, name)
        return
    for name in os.listdir(path):
        if name.lower().endswith(".cfax"):
            yield os.path.join(path, name)


def pcapng_path_for(cfax_path: str) -> str:
    base, _ = os.path.splitext(cfax_path)
    return f"{base}.pcapng"


def resolve_function_params(
    function: object,
    include_show_log: bool,
    technology_filter: Optional[str] = None,
) -> Tuple[Dict[str, object], Optional[str]]:
    import inspect

    signature = inspect.signature(function)
    params = signature.parameters
    kwargs: Dict[str, object] = {}

    # If the target accepts **kwargs, it is safe to pass optional flags
    # even if they are not explicitly declared.
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if include_show_log and ("show_log" in params or accepts_kwargs):
        kwargs["show_log"] = True

    if technology_filter:
        for name in ("technology_filter", "filter", "technology", "tech_filter"):
            if name in params:
                return kwargs, name
        if accepts_kwargs:
            return kwargs, "technology_filter"
        # Not supported by this wpshelper version; caller should ignore.
        return kwargs, None

    return kwargs, None


def export_capture(
    wps_handle: dict,
    cfax_path: str,
    verbose: bool,
    technology_filter: Optional[str],
    logger: logging.Logger,
) -> None:
    pcapng_path = pcapng_path_for(cfax_path)
    logger.info("Opening capture %s", cfax_path)
    #open_kwargs, _ = resolve_function_params(wps_open_capture, verbose)

    wps_open_capture(wps_handle, cfax_path, show_log=verbose)
    wps_wireless_devices(wps_handle, action="select", action_parameters={'type':'bluetooth','address':'all','select':'yes'}, show_log=verbose)
    wps_analyze_capture(wps_handle,show_log=verbose)
    logger.info("Exporting pcapng to %s", pcapng_path)
    time.sleep(5)
    # export_kwargs, filter_param = resolve_function_params(
    #     wps_export_pcapng,
    #     verbose,
    #     technology_filter=technology_filter,
    # )
    # if filter_param and technology_filter:
    #     export_kwargs[filter_param] = technology_filter
    # elif technology_filter:
    #     logger.debug(
    #         "wps_export_pcapng does not accept a technology filter parameter; continuing without --technology-filter=%s",
    #         technology_filter,
    #     )
    wps_export_pcapng(wps_handle, pcapng_path, tech=technology_filter, show_log=verbose)


def main() -> None:
    wps_installations = wps_find_installations(base_dir=DEFAULT_WPS_BASE_DIR, show_log=False)
    latest_install = (wps_installations or {}).get("latest") or {}
    default_wps_path = latest_install.get("path") or DEFAULT_WPS_PATH

    parser = argparse.ArgumentParser(
        description="Convert WPS .cfax captures to .pcapng using WPS automation."
    )
    parser.add_argument("input_path", help="Path to a .cfax file or directory containing .cfax files.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .cfax files.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",        
        help="Skip conversion if the .pcapng file already exists.",
    )
    parser.add_argument(
        "--technology-filter",
        default="LE",
        help="Technology filter string to pass to the pcapng export, if supported by WPS.",
    )
    parser.add_argument("--log-level", default="info", help="Log level (debug, info, warning, error, critical).")
    parser.add_argument("--log-file", help="Optional log file path.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging and WPS show_log output.")
    parser.add_argument(
        "--wps-path",
        default=default_wps_path,
        help="Base path for WPS install (defaults to latest detected installation when available).",
    )
    parser.add_argument("--auto-server-path", help="Path to FTSAutoServer.exe.")
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

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("WPS installations scan (base_dir=%s):\n%s", DEFAULT_WPS_BASE_DIR, json.dumps(wps_installations, indent=2, sort_keys=True))
    logger.info("Default WPS path selected: %s", default_wps_path)

    wps_executable_path = os.path.join(args.wps_path, "Executables", "Core")
    auto_server_path = args.auto_server_path or os.path.join(wps_executable_path, "FTSAutoServer.exe")

    cfax_files = sorted(iter_cfax_files(args.input_path, args.recursive))
    if not cfax_files:
        logger.warning("No .cfax files found under %s", args.input_path)
        return

    server_process = None
    wps_handle = None
    try:
        server_process = start_server(auto_server_path, logger)
        wps_handle = wps_open(
            tcp_ip=args.tcp_ip,
            tcp_port=args.tcp_port,
            max_to_read=MAX_TO_READ,
            wps_executable_path=wps_executable_path,
            personality_key="VIEW",
            sleep_time=args.sleep_time,
            max_wait_time=args.max_wait_time,
            recv_retry_attempts=10,
            recv_retry_sleep=5,
        )
        close_previous_capture=False
        for cfax_path in cfax_files:
            if close_previous_capture:
                # TODO: Resolve why getting this error:
                # 2026-01-18 23:50:53,237 INFO Opening capture x240_bredr_le_2m_20260114_064910.cfax
                # wps_open_capture: sending: b'Open Capture File;x240_bredr_le_2m_20260114_064910.cfax;notify=1'
                # _recv_and_parse: PCAPNG EXPORT;FAILED;Timestamp=1/18/2026 11:50:57 PM;Reason=Could not open capture  file             
                time.sleep(10)
                # Adding a delay because getting this unusual error
                logger.info("Slept now closing and  starting the next file")
                wps_close_capture(wps_handle,capture_absolute_filename=cfax_path, show_log=args.verbose)

            pcapng_path = pcapng_path_for(cfax_path)
            if args.skip_existing and os.path.exists(pcapng_path):
                logger.info("Skipping %s (pcapng already exists)", cfax_path)
                continue
            export_capture(wps_handle, cfax_path, args.verbose, args.technology_filter, logger)
            logger.info("Converted %s -> %s", cfax_path, pcapng_path)
            close_previous_capture=True
            
            # This is temorary to close the capture between files to avoid errors
            wps_close(wps_handle, show_log=args.verbose)
            server_process.terminate()
            # terminate the program
            sys.exit(0)
           

    except Exception as exc:
        logger.exception("Conversion failed: %s", exc)
        raise
    finally:
        if wps_handle:
            wps_close(wps_handle, show_log=args.verbose)
        if server_process:
            server_process.terminate()


if __name__ == "__main__":
    main()
