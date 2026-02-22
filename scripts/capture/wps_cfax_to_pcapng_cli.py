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
    wps_close_capture,
    wps_log_text,
)

TCP_IP = "127.0.0.1"
TCP_PORT = 22901
MAX_TO_READ = 1000
SLEEP_TIME = 2
MAX_WAIT_TIME = 60
DEFAULT_RECV_RETRY_ATTEMPTS = 10
DEFAULT_RECV_RETRY_SLEEP = 15.0

DEFAULT_WPS_BASE_DIR = r"C:\Program Files (x86)\Teledyne LeCroy Wireless"
DEFAULT_WPS_PATH = r"C:\Program Files (x86)\Teledyne LeCroy Wireless\Wireless Protocol Suite 4.50"

LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}


def _explain_log_file_error(log_file: str, exc: BaseException) -> str:
    log_file = os.path.abspath(str(log_file))
    log_dir = os.path.dirname(log_file)

    lines = [
        f"Failed to create/open log file: {log_file}",
        f"Reason: {type(exc).__name__}: {exc}",
        f"Current working directory: {os.getcwd()}",
    ]

    if log_dir:
        lines.append(f"Log directory: {log_dir}")
        lines.append(f"Log directory exists: {os.path.exists(log_dir)}")
        # do not create the directory if it does not exist, but report on the parent folder existence and type to help diagnose common issues like a missing parent folder or a file in place of the expected log directory.
        if os.path.exists(log_dir):
            lines.append(f"Log directory is a directory: {os.path.isdir(log_dir)}")
        else:
            lines.append("Log directory does not exist (missing parent folder).")

    drive, _ = os.path.splitdrive(log_file)
    if drive:
        root = drive + os.sep
        lines.append(f"Drive root exists ({root}): {os.path.exists(root)}")

    return "\n".join(lines)


def _file_handler_stream_for(log_file: str):
    """Return the stream for the configured FileHandler pointing at log_file, if present."""
    if not log_file:
        return None

    target = os.path.abspath(log_file)
    root = logging.getLogger()
    for h in getattr(root, "handlers", []) or []:
        if isinstance(h, logging.FileHandler):
            base = os.path.abspath(getattr(h, "baseFilename", "") or "")
            if base == target:
                return getattr(h, "stream", None)
    return None


def _append_wps_automation_log_if_enabled(
    *,
    log_file: Optional[str],
    wps_handle: Optional[dict],
    logger: logging.Logger,
) -> None:
    """If file logging is enabled, append the cumulative WPS automation log (handle['log']) once, at exit."""
    if not log_file or not wps_handle:
        return

    try:
        txt = wps_log_text(wps_handle)
    except Exception as e:
        logger.debug("Could not format WPS automation log via wps_log_text: %s", e, exc_info=True)
        return

    if not txt.strip():
        return

    header = "\n===== WPS automation log (cumulative) =====\n"
    footer = "===== End WPS automation log =====\n"

    # Prefer writing to the existing FileHandler stream (avoids Windows file locking issues).
    try:
        stream = _file_handler_stream_for(log_file)
        if stream is not None:
            stream.write(header)
            stream.write(txt if txt.endswith("\n") else (txt + "\n"))
            stream.write(footer)
            stream.flush()
            return
    except Exception as e:
        logger.debug("Could not append WPS automation log via FileHandler stream: %s", e, exc_info=True)

    # Fallback: open and append directly.
    try:
        log_file_abs = os.path.abspath(log_file)
        log_dir = os.path.dirname(log_file_abs)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_file_abs, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(txt if txt.endswith("\n") else (txt + "\n"))
            f.write(footer)
    except OSError as e:
        logger.error(_explain_log_file_error(log_file, e))


def configure_logging(log_level: str, log_file: Optional[str]) -> logging.Logger:
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file = os.path.abspath(log_file)
        log_dir = os.path.dirname(log_file)

        if log_dir:
            try:
                if os.path.exists(log_dir) and not os.path.isdir(log_dir):
                    raise NotADirectoryError(f"Log directory path exists but is not a directory: {log_dir}")
                elif not os.path.exists(log_dir):
                    raise FileNotFoundError(f"Log directory does not exist: {log_dir}")
            except OSError as e:
                raise RuntimeError(_explain_log_file_error(log_file, e)) from e

        # Create/open the log file (surface any issues with a useful explanation).
        try:
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        except OSError as e:
            raise RuntimeError(_explain_log_file_error(log_file, e)) from e

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
    logger.info("Opening capture: %s", cfax_path)

    wps_open_capture(wps_handle, cfax_path, show_log=verbose)
    wps_wireless_devices(
        wps_handle,
        action="select",
        action_parameters={"type": "bluetooth", "address": "all", "select": "yes"},
        show_log=verbose,
    )
    wps_analyze_capture(wps_handle, show_log=verbose)

    logger.info("Exporting pcapng (tech=%s) -> %s", technology_filter, pcapng_path)
    time.sleep(5)
    wps_export_pcapng(wps_handle, pcapng_path, tech=technology_filter, show_log=verbose)


def main() -> None:
    # Create a logger early so we can still log from finally even if configure_logging fails mid-way.
    logger = logging.getLogger("wps_cfax_to_pcapng_cli")

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
        help="Technology filter string to pass to the pcapng export, currently supported by WPS [Classic,LE,80211,WPAN]. Multiple names should be listed without spaces, e.g. 'LE,80211' not 'LE, 80211'.",
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
    parser.add_argument(
        "--recv-retry-attempts",
        type=int,
        default=DEFAULT_RECV_RETRY_ATTEMPTS,
        help=f"Number of receive retry attempts for WPS automation calls (default: {DEFAULT_RECV_RETRY_ATTEMPTS}).",
    )
    parser.add_argument(
        "--recv-retry-sleep",
        type=float,
        default=DEFAULT_RECV_RETRY_SLEEP,
        help=f"Seconds to sleep between receive retries (default: {DEFAULT_RECV_RETRY_SLEEP}).",
    )

    args = parser.parse_args()
    args.log_level = args.log_level.lower()

    validate_args(args)
    logger = configure_logging(args.log_level, args.log_file)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Log key settings (including recursive) and input
    logger.info(
        "Settings: input_path=%s recursive=%s skip_existing=%s technology_filter=%s verbose=%s log_file=%s log_level=%s",
        args.input_path,
        args.recursive,
        args.skip_existing,
        args.technology_filter,
        args.verbose,
        args.log_file,
        args.log_level,
    )
    logger.info(
        "WPS automation: tcp_ip=%s tcp_port=%s sleep_time=%s max_wait_time=%s recv_retry_attempts=%s recv_retry_sleep=%s wps_path=%s",
        args.tcp_ip,
        args.tcp_port,
        args.sleep_time,
        args.max_wait_time,
        args.recv_retry_attempts,
        args.recv_retry_sleep,
        args.wps_path,
    )

    logger.info(
        "WPS installations scan (base_dir=%s):\n%s",
        DEFAULT_WPS_BASE_DIR,
        json.dumps(wps_installations, indent=2, sort_keys=True),
    )
    logger.info("Default WPS path selected: %s", default_wps_path)

    wps_executable_path = os.path.join(args.wps_path, "Executables", "Core")
    auto_server_path = args.auto_server_path or os.path.join(wps_executable_path, "FTSAutoServer.exe")

    cfax_files = sorted(iter_cfax_files(args.input_path, args.recursive))
    logger.info("Discovered %d .cfax file(s).", len(cfax_files))
    if args.verbose and cfax_files:
        for i, p in enumerate(cfax_files, start=1):
            logger.debug("cfax[%d/%d]=%s", i, len(cfax_files), p)

    if not cfax_files:
        logger.warning("No .cfax files found under %s (recursive=%s)", args.input_path, args.recursive)
        return

    server_process = None
    wps_handle = None
    last_opened_cfax: Optional[str] = None

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
            recv_retry_attempts=args.recv_retry_attempts,
            recv_retry_sleep=args.recv_retry_sleep,
        )

        for idx, cfax_path in enumerate(cfax_files, start=1):
            logger.info("Processing file %d/%d: %s", idx, len(cfax_files), cfax_path)

            pcapng_path = pcapng_path_for(cfax_path)
            if args.skip_existing and os.path.exists(pcapng_path):
                logger.info("Skipping %s (pcapng already exists: %s)", cfax_path, pcapng_path)
                continue

            # Close the previous capture (if any) before opening the next one.
            if last_opened_cfax:
                try:
                    # Small delay to reduce WPS "Could not open capture file" flakiness between captures.
                    time.sleep(2)
                    logger.debug("Closing previous capture: %s", last_opened_cfax)
                    wps_close_capture(wps_handle, capture_absolute_filename=last_opened_cfax, show_log=args.verbose)
                except Exception:
                    logger.debug("Failed to close previous capture (continuing): %s", last_opened_cfax, exc_info=True)

            export_capture(wps_handle, cfax_path, args.verbose, args.technology_filter, logger)
            logger.info("Converted %s -> %s", cfax_path, pcapng_path)

            last_opened_cfax = cfax_path

    except Exception as exc:
        logger.exception("Conversion failed: %s", exc)
        raise
    finally:
        # Only at program exit: append the cumulative WPS automation log to the same log file (if enabled).
        _append_wps_automation_log_if_enabled(
            log_file=getattr(args, "log_file", None),
            wps_handle=wps_handle,
            logger=logger,
        )

        if wps_handle:
            try:
                wps_close(wps_handle, show_log=getattr(args, "verbose", False))
            except Exception:
                logger.debug("wps_close failed during cleanup", exc_info=True)

        if server_process:
            try:
                server_process.terminate()
            except Exception:
                logger.debug("Failed to terminate server process during cleanup", exc_info=True)

if __name__ == "__main__":
    main()
