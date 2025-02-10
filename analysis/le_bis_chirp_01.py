import pandas as pd
import numpy as np
from scapy.all import rdpcap
from scapy.layers.bluetooth import BTLEAdvertisingHdr
import scipy.optimize as optimize
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import os

@dataclass

class BroadcastParameters:
    channels: List[int]
    iso_interval: int  # in microseconds
    subevents: int
    stream_spacing: int  # in microseconds
    bursts: int
    immediate_repeat_count: int

@dataclass
class EnvironmentData:
    ble_packets: pd.DataFrame
    wifi_packets: pd.DataFrame
    zigbee_packets: pd.DataFrame
    rssi_data: pd.DataFrame

def load_pcap_data(pcap_file: str) -> Dict[str, pd.DataFrame]:
    """
    Load and parse pcapng files containing BLE, WiFi, and Zigbee packets
    """
    packets = rdpcap(pcap_file)
    
    ble_data = []
    wifi_data = []
    zigbee_data = []
    
    for packet in packets:
        timestamp = packet.time
        if BTLEAdvertisingHdr in packet:
            ble_data.append({
                'timestamp': timestamp,
                'channel': packet[BTLEAdvertisingHdr].channel,
                'rssi': packet[BTLEAdvertisingHdr].rssi,
                'type': 'ble'
            })
        # Add similar processing for WiFi and Zigbee packets
        
    return {
        'ble': pd.DataFrame(ble_data),
        'wifi': pd.DataFrame(wifi_data),
        'zigbee': pd.DataFrame(zigbee_data)
    }

def load_rssi_data(csv_file: str) -> pd.DataFrame:
    """
    Load RSSI spectrogram data from CSV
    """
    return pd.read_csv(csv_file, parse_dates=['timestamp'])

class InterferenceAnalyzer:
    def __init__(self, env_data: EnvironmentData):
        self.env_data = env_data
    
    def calculate_channel_interference(self, channel: int, timestamp: float) -> float:
        """
        Calculate interference level for a specific channel at a given time
        """
        # Get nearby WiFi and Zigbee packets
        wifi_interference = self._calculate_wifi_interference(channel, timestamp)
        zigbee_interference = self._calculate_zigbee_interference(channel, timestamp)
        
        # Get RSSI level from spectrogram
        rssi_level = self._get_rssi_level(channel, timestamp)
        
        return max(wifi_interference, zigbee_interference, rssi_level)
    
    def _calculate_wifi_interference(self, channel: int, timestamp: float) -> float:
        # Implementation for WiFi interference calculation
        pass
    
    def _calculate_zigbee_interference(self, channel: int, timestamp: float) -> float:
        # Implementation for Zigbee interference calculation
        pass
    
    def _get_rssi_level(self, channel: int, timestamp: float) -> float:
        # Get RSSI level from spectrogram data
        pass

class BroadcastOptimizer:
    def __init__(self, env_data: EnvironmentData, target_probability: float):
        self.env_data = env_data
        self.interference_analyzer = InterferenceAnalyzer(env_data)
        self.target_probability = target_probability
    
    def optimize_parameters(self, initial_params: BroadcastParameters) -> Tuple[BroadcastParameters, int]:
        """
        Optimize broadcast parameters to maximize number of streams while meeting
        target reception probability
        """
        def objective_function(x):
            # Unpack parameters
            channels, iso_interval, subevents, spacing, bursts, repeats = x
            
            # Calculate expected packet reception probability
            prob = self._calculate_reception_probability(
                BroadcastParameters(
                    channels=channels,
                    iso_interval=iso_interval,
                    subevents=subevents,
                    stream_spacing=spacing,
                    bursts=bursts,
                    immediate_repeat_count=repeats
                )
            )
            
            # Calculate maximum number of streams possible
            max_streams = self._calculate_max_streams(
                iso_interval, subevents, spacing, bursts, repeats
            )
            
            # Penalty if probability is below target
            penalty = max(0, self.target_probability - prob) * 1000
            
            return -(max_streams - penalty)
        
        # Define bounds and constraints
        bounds = [
            (1, 37),  # channels
            (7500, 160000),  # iso_interval
            (1, 64),  # subevents
            (100, 10000),  # spacing
            (1, 10),  # bursts
            (1, 4)  # repeats
        ]
        
        result = optimize.minimize(
            objective_function,
            x0=[
                initial_params.channels[0],
                initial_params.iso_interval,
                initial_params.subevents,
                initial_params.stream_spacing,
                initial_params.bursts,
                initial_params.immediate_repeat_count
            ],
            bounds=bounds,
            method='SLSQP'
        )
        
        optimized_params = BroadcastParameters(
            channels=[int(result.x[0])],
            iso_interval=int(result.x[1]),
            subevents=int(result.x[2]),
            stream_spacing=int(result.x[3]),
            bursts=int(result.x[4]),
            immediate_repeat_count=int(result.x[5])
        )
        
        max_streams = self._calculate_max_streams(
            optimized_params.iso_interval,
            optimized_params.subevents,
            optimized_params.stream_spacing,
            optimized_params.bursts,
            optimized_params.immediate_repeat_count
        )
        
        return optimized_params, max_streams
    
    def _calculate_reception_probability(self, params: BroadcastParameters) -> float:
        """
        Calculate packet reception probability based on interference analysis
        """
        # Implement probability calculation based on interference levels
        # and broadcast parameters
        pass
    
    def _calculate_max_streams(self, iso_interval: int, subevents: int,
                             spacing: int, bursts: int, repeats: int) -> int:
        """
        Calculate maximum number of streams possible with given parameters
        """
        total_time_per_stream = bursts * repeats * spacing
        return iso_interval // total_time_per_stream

def main():
    
    # print the current working directory
    print(os.getcwd())
    
    # Example usage
    pcap_file = "./data/le_bis/le_bis_chirp_01_select_01.pcapng"
    rssi_file = "spectrogram.csv"
    
    # Load data
    packet_data = load_pcap_data(pcap_file)
    #rssi_data = load_rssi_data(rssi_file)
    
    env_data = EnvironmentData(
        ble_packets=packet_data['ble'],
        wifi_packets=packet_data['wifi'],
        zigbee_packets=packet_data['zigbee'],
        rssi_data=rssi_data
    )
    
    # Initial parameters
    initial_params = BroadcastParameters(
        channels=[37],
        iso_interval=10000,
        subevents=4,
        stream_spacing=1000,
        bursts=2,
        immediate_repeat_count=2
    )
    
    # Create optimizer
    optimizer = BroadcastOptimizer(env_data, target_probability=0.99)
    
    # Optimize parameters
    optimal_params, max_streams = optimizer.optimize_parameters(initial_params)
    
    # Print results
    print(f"Optimal Parameters:")
    print(f"Channels: {optimal_params.channels}")
    print(f"ISO Interval: {optimal_params.iso_interval} μs")
    print(f"Subevents: {optimal_params.subevents}")
    print(f"Stream Spacing: {optimal_params.stream_spacing} μs")
    print(f"Bursts: {optimal_params.bursts}")
    print(f"Immediate Repeat Count: {optimal_params.immediate_repeat_count}")
    print(f"\nMaximum Streams: {max_streams}")

if __name__ == "__main__":
    main()