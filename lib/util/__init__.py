import os
import re
import socket
import subprocess
import time

from typing import Optional
# run the remote commands on a Linux machine
def run_command(command):
    if command is None:
        command = "ls" # default command 

    # Run the command and capture the output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    stdout, stderr = process.communicate()
    stdout_str = str(stdout)
    # Check for errors
    if process.returncode != 0:
        log_entry = f"run_command: An error occurred while executing the command {command}:\n{stderr}"
    else:
        log_entry = f"run_command: Command output:\n{stdout}"
    print(log_entry)
    return stdout_str

def start_server(wps_executable_path: str) -> Optional[subprocess.Popen]:
    server_program = 'FTSAutoServer.exe'
    auto_server_path = os.path.join(wps_executable_path, server_program)
    if not os.path.exists(auto_server_path):
        raise FileNotFoundError(
            f"Auto server executable not found at {auto_server_path}. "
            "Set --auto-server-path to the correct location."
        )
    process = subprocess.Popen([auto_server_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    return process