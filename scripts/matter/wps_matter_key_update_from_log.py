import argparse
import re
import time
import datetime
import serial
import sys
import os
import json

module_path = os.path.abspath(os.path.join('./wpshelper'))
if module_path not in sys.path:
    sys.path.append(module_path)
from wpshelper import wps_open, wps_configure, wps_start_record, wps_stop_record, wps_analyze_capture, wps_save_capture, wps_update_matter_keys, wps_close

# --------------------- API CONFIGURATION (modify as needed) ---------------------
TCP_IP = '192.168.58.1'    # IP address of the automation server interface
TCP_PORT = 22901            # Default port
MAX_TO_READ = 1000

# Capture technology settings (for Bluetooth LE, etc.)
capture_technology = "capturetechnology=bredr-off|le-on|2m-on|spectrum-off|wifi-off|wpan-on"

# Only one personality_key should be active.
personality_key = "X500"
personality_key = personality_key.strip()

# Setup paths (modify these to suit your environment)
wps_path = r'C:\Program Files (x86)\Teledyne LeCroy Wireless\Wireless Protocol Suite 4.30 (BETA)'
data_path = r'C:\Users\Public\Documents\Teledyne LeCroy Wireless\My Capture Files'

# Create a unique capture name using the current date and time
current_datetime = datetime.datetime.now()
datetime_str = current_datetime.strftime("%Y_%m_%d_%H_%M")
capture_name = 'matter_capture_' + datetime_str
capture_absolute_filename = os.path.join(data_path, capture_name + ".cfax")
wps_executable_path = os.path.join(wps_path, 'Executables', 'Core')

# ------------------------------------------------------------------------------

# Regular expression to capture hex values (do not include the "0x")
# Example log entries:
# [00:00:28.851,867] <inf> chip: [EM]__CryptoMaterial__ AES_CCM_decrypt key = 0x48e8b550bfc8523e5906c28a8b53d50a
# [00:00:28.852,203] <inf> chip: [EM]__CryptoMaterial__ AES_CCM_decrypt nonce = 0x004d7b000c0000000000000000
pattern = re.compile(r"AES_CCM_(encrypt|decrypt) (key|nonce) = 0x([0-9a-fA-F]+)")

def process_line(line):
    """Extract session key or source node id (from nonce) from a log line."""
    match = pattern.search(line)
    if match:
        operation, type_, value = match.groups()
        if type_ == 'key':
            return {"key": value}
        elif type_ == 'nonce':
            # Extract the last 8 bytes (16 hex characters) representing the Source Node ID,
            # then flip the byte order (two characters at a time)
            value = value[-16:]
            value = ''.join(reversed([value[i:i+2] for i in range(0, len(value), 2)]))
            return {"source_node_id": value}
    return {}

def update_file(update_file_name, matter_mapping):
    """
    Append the entire stored source node id dictionary as a log line entry with a date time stamp 
    and the dictionary in JSON format.
    """
    try:
        with open(update_file_name, "a") as f:
            timestamp = datetime.datetime.now().isoformat()
            log_entry = f"{timestamp} {json.dumps(matter_mapping)}\n"
            f.write(log_entry)
    except Exception as e:
        print(f"Error: could not open or write to file {update_file_name}: {e}")

def main(args):
    # Dictionary to maintain mapping of source node id -> list of session keys (all with "0x" prefix)
    matter_mapping = {}
    current_source_node_id = None
    current_session_keys = []  # Temporary list to hold session keys for the upcoming node id

    # ----------------- Start WPS Capture using API functions -----------------
    wps_handle = wps_open(tcp_ip=TCP_IP, tcp_port=TCP_PORT, max_to_read=MAX_TO_READ,
                          wps_executable_path=wps_executable_path, personality_key=personality_key)
    wps_configure(wps_handle, personality_key, capture_technology, show_log=True)
    wps_start_record(wps_handle, show_log=True)
    # -------------------------------------------------------------------------

    last_update_time = time.time() - args.min_duration  # Force an immediate file update

    # Open input (either from a file or serial port)
    lines = []
    if args.input_file:
        try:
            with open(args.input_file, 'r') as file:
                lines = file.readlines()
        except Exception as e:
            print(f"Error: could not open input file {args.input_file}: {e}")
            return
    else:
        try:
            ser = serial.Serial(args.serial_port, args.baud_rate, timeout=None)
            lines = iter(lambda: ser.readline().decode('utf-8'), '')
        except Exception as e:
            print(f"Error: could not open serial port {args.serial_port}: {e}")
            return

    # Open output file to log incoming data
    try:
        output_file = open(args.output_file_name, "w")
    except Exception as e:
        print(f"Error: could not open output file {args.output_file_name}: {e}")
        return

    try:
        for line in lines:
            output_file.write(line)
            print(line.strip())
            result = process_line(line)
            if "key" in result:
                # Buffer the session key if not already buffered
                if result["key"] not in current_session_keys:
                    current_session_keys.append(result["key"])
            if "source_node_id" in result:
                new_node_id = result["source_node_id"]
                if current_source_node_id is not None:
                    formatted_node_id = "0x" + current_source_node_id
                    new_keys = ["0x" + key for key in current_session_keys]
                    # Check if this node id already exists in the mapping
                    if formatted_node_id in matter_mapping:
                        existing_keys = matter_mapping[formatted_node_id]
                        # Only add and update if at least one key is new
                        keys_to_add = [k for k in new_keys if k not in existing_keys]
                        if keys_to_add:
                            matter_mapping[formatted_node_id].extend(keys_to_add)
                            wps_update_matter_keys(wps_handle, formatted_node_id, matter_mapping[formatted_node_id], show_log=True)
                    else:
                        # New node id encountered so add the keys (if any)
                        matter_mapping[formatted_node_id] = new_keys
                        if new_keys:
                            wps_update_matter_keys(wps_handle, formatted_node_id, new_keys, show_log=True)
                    # Reset the temporary session keys list for the next node id
                    current_session_keys = []
                # Set the current source node id to the new one
                current_source_node_id = new_node_id

            # Periodically update the mapping log file
            current_time = time.time()
            if current_time - last_update_time >= args.min_duration:
                update_file(args.update_file_name, matter_mapping)
                last_update_time = current_time

    except KeyboardInterrupt:
        print("Interrupted by the user")
        # On interrupt, update the pending node (if any) before exiting
        if current_source_node_id is not None:
            formatted_node_id = "0x" + current_source_node_id
            new_keys = ["0x" + key for key in current_session_keys]
            if formatted_node_id in matter_mapping:
                existing_keys = matter_mapping[formatted_node_id]
                keys_to_add = [k for k in new_keys if k not in existing_keys]
                if keys_to_add:
                    matter_mapping[formatted_node_id].extend(keys_to_add)
                    wps_update_matter_keys(wps_handle, formatted_node_id, matter_mapping[formatted_node_id], show_log=True)
            else:
                matter_mapping[formatted_node_id] = new_keys
                if new_keys:
                    wps_update_matter_keys(wps_handle, formatted_node_id, new_keys, show_log=True)
        update_file(args.update_file_name, matter_mapping)

    # Final update for any pending keys after loop completion
    if current_source_node_id is not None:
        formatted_node_id = "0x" + current_source_node_id
        new_keys = ["0x" + key for key in current_session_keys]
        if formatted_node_id in matter_mapping:
            existing_keys = matter_mapping[formatted_node_id]
            keys_to_add = [k for k in new_keys if k not in existing_keys]
            if keys_to_add:
                matter_mapping[formatted_node_id].extend(keys_to_add)
                wps_update_matter_keys(wps_handle, formatted_node_id, matter_mapping[formatted_node_id], show_log=True)
        else:
            matter_mapping[formatted_node_id] = new_keys
            if new_keys:
                wps_update_matter_keys(wps_handle, formatted_node_id, new_keys, show_log=True)
    update_file(args.update_file_name, matter_mapping)

    output_file.close()

    # ----------------- Stop and Save WPS Capture -----------------
    wps_stop_record(wps_handle)
    wps_analyze_capture(wps_handle)
    wps_save_capture(wps_handle, capture_absolute_filename)
    wps_close(wps_handle)
    # --------------------------------------------------------------

if __name__ == "__main__":
    # Make sure C:\Program Files (x86)\Teledyne LeCroy Wireless\Wireless Protocol Suite 4.30 (BETA)\Executables\Core\FTSAutoServer.exe is running
    parser = argparse.ArgumentParser(
        description="Capture data, extract Matter session keys and source node IDs, and update WPS Matter keys."
    )
    parser.add_argument("--serial_port", default='COM1', help="Serial port to use (default: COM1).")
    parser.add_argument("--baud_rate", type=int, default=115200, help="Baud rate for serial communication (default: 115200).")
    parser.add_argument("--min_duration", type=int, default=10, help="Minimum duration (seconds) between file updates (default: 10).")
    parser.add_argument("--update_file_name", default="hex_values.txt", help="File name for logging Matter keys mapping (default: hex_values.txt).")
    parser.add_argument("--output_file_name", default="output.txt", help="File name for the serial data output (default: output.txt).")
    parser.add_argument("--input_file", help="Optional input file to read from instead of serial port.")
    
    args = parser.parse_args()
    main(args)