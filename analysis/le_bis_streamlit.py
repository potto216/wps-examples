import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from io import BytesIO
import soundfile as sf  # For writing audio data to an in-memory buffer

# ------------------------------
# Utility / Helper functions
# ------------------------------

@st.cache_data
def load_csv(uploaded_file):
    """Load a CSV file into a pandas DataFrame."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

@st.cache_data
def load_audio_data(file_path):
    """
    Load a WAV file from a given path (or uploaded file), returning:
      - sample_rate
      - data array
      - an in-memory BytesIO buffer containing the exact WAV data
    so that we can play it via st.audio without re-loading from disk.
    """
    if file_path is None:
        return None, None, None

    # Read the WAV file (sample rate + data)
    sample_rate, data = wavfile.read(file_path)

    # Write the data into a BytesIO buffer in WAV format
    wav_buffer = BytesIO()
    # Use `soundfile` to write the data properly with the correct samplerate
    sf.write(wav_buffer, data, sample_rate, format='WAV')
    wav_buffer.seek(0)  # reset buffer to the beginning

    return sample_rate, data, wav_buffer


def compute_packet_statistics(df, start_time_relative_sec, end_time_relative_sec, time_col='Timestamp'):
    """
    Given a DataFrame with a time_col column, count total packets in
    the [start_time, end_time] window (inclusive).
    """
    print(f"v1 {df.columns}")
    if df is None or time_col not in df.columns:
        return {}

    # the start time is the first timestamp of the df  added to the start_time_relative_sec
    first_timestamp = df[time_col].min()  # Earliest timestamp in the DataFrame

    # Add seconds as a timedelta
    start_time = first_timestamp + pd.Timedelta(seconds=start_time_relative_sec)
    end_time = first_timestamp + pd.Timedelta(seconds=end_time_relative_sec)

    # Filter rows in the desired time range:
    window_df = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)]
    count = len(window_df)
    return {"packet_count": count}

# ------------------------------
# Streamlit Application
# ------------------------------

def main():
    st.title("Bluetooth LE Broadcast Audio Analysis")

    st.header("1. Upload Wireless Capture Data")
    st.markdown("""
        Upload your CSV files with wireless captures for each technology:
        - Wi-Fi (802.11)
        - IEEE 802.15.4
        - Bluetooth LE
        - Bluetooth BR/EDR
    """)

    # Example with local file paths for demonstration;
    # in a real Streamlit app, you might use file_uploader
    wifi_file = '../data/le_bis/wi-fi_802_11.csv'
    ieee154_file = '../data/le_bis/802_15_4.csv'
    btle_file = '../data/le_bis/bluetooth_le.csv'
    btclassic_file = '../data/le_bis/bluetooth_br_edr.csv'

    st.header("2. Upload Audio Files")
    st.markdown("""
        Upload two WAV files:
        - The **original broadcast** audio
        - The **captured broadcast** audio
    """)

    # Same approach here:
    original_audio_file = '../data/le_bis/le_bis_chirp_01_stream2.wav'
    captured_audio_file = '../data/le_bis/le_bis_chirp_01_stream2_mod_01.wav'

    # ------------------------------
    # Load all data into memory (cached by Streamlit)
    # ------------------------------
    wifi_df = load_csv(wifi_file)
    if wifi_df is not None:
        wifi_df['Timestamp'] = wifi_df['Timestamp'].str.replace('"', '', regex=False)
        wifi_df['Timestamp'] = wifi_df['Timestamp'].str.strip('=')
        wifi_df['Timestamp'] = pd.to_datetime(wifi_df['Timestamp'], format="%m/%d/%Y %I:%M:%S.%f %p")

    ieee154_df = load_csv(ieee154_file)
    if ieee154_df is not None:
        ieee154_df['Timestamp'] = ieee154_df['Timestamp'].str.replace('"', '', regex=False)
        ieee154_df['Timestamp'] = ieee154_df['Timestamp'].str.strip('=')
        ieee154_df['Timestamp'] = pd.to_datetime(ieee154_df['Timestamp'], format="%m/%d/%Y %I:%M:%S.%f %p")

    btle_df = load_csv(btle_file)
    if btle_df is not None:
        btle_df['Timestamp'] = btle_df['Timestamp'].str.replace('"', '', regex=False)
        btle_df['Timestamp'] = btle_df['Timestamp'].str.strip('=')
        btle_df['Timestamp'] = pd.to_datetime(btle_df['Timestamp'], format="%m/%d/%Y %I:%M:%S.%f %p")

    btclassic_df = load_csv(btclassic_file)
    if btclassic_df is not None:
        btclassic_df['Timestamp'] = btclassic_df['Timestamp'].str.replace('"', '', regex=False)
        btclassic_df['Timestamp'] = btclassic_df['Timestamp'].str.strip('=')
        btclassic_df['Timestamp'] = pd.to_datetime(btclassic_df['Timestamp'], format="%m/%d/%Y %I:%M:%S.%f %p")

    # ------------------------------
    # Load WAV files (cached)
    # ------------------------------
    sr_original, data_original, original_wav_buffer = load_audio_data(original_audio_file)
    sr_captured, data_captured, captured_wav_buffer = load_audio_data(captured_audio_file)

    # Determine earliest timestamp for alignment
    earliest_timestamp = None
    for df in [wifi_df, ieee154_df, btle_df, btclassic_df]:
        if df is not None and 'TimestampFake' in df.columns:
            df_min = df['TimestampFake'].min()
            if earliest_timestamp is None:
                earliest_timestamp = df_min
            else:
                earliest_timestamp = min(earliest_timestamp, df_min)

    # If earliest_timestamp is None, fallback to 0
    if earliest_timestamp is None:
        earliest_timestamp = 0.0

    # ------------------------------
    # Display the audio players if loaded
    # ------------------------------
    if sr_original is not None and data_original is not None:
        st.subheader("Original Audio")
        st.audio(original_wav_buffer, format='audio/wav')  # uses the BytesIO buffer

    if sr_captured is not None and data_captured is not None:
        st.subheader("Captured Broadcast Audio")
        st.audio(captured_wav_buffer, format='audio/wav')  # uses the BytesIO buffer

    # ------------------------------
    # Plot the waveforms
    # ------------------------------
    if (sr_original and data_original is not None) and (sr_captured and data_captured is not None):
        st.header("Waveform Comparison")

        # Convert the WAV data into time axes
        duration_original = len(data_original) / sr_original
        duration_captured = len(data_captured) / sr_captured
        
        time_original = np.linspace(
            earliest_timestamp,
            earliest_timestamp + duration_original,
            num=len(data_original)
        )
        time_captured = np.linspace(
            earliest_timestamp,
            earliest_timestamp + duration_captured,
            num=len(data_captured)
        )
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(time_original, data_original, label="Original Audio", color='blue')
        ax[0].set_title("Original Audio Waveform")
        ax[0].set_ylabel("Amplitude")

        ax[1].plot(time_captured, data_captured, label="Captured Audio", color='orange')
        ax[1].set_title("Captured Audio Waveform")
        ax[1].set_xlabel("Time (seconds)")
        ax[1].set_ylabel("Amplitude")

        st.pyplot(fig)

        # ------------------------------
        # Let the user specify region of interest
        # ------------------------------
        st.subheader("Select Time Range to Analyze")
        min_time = float(earliest_timestamp)
        max_time = float(max(time_original[-1], time_captured[-1]))

        # We'll allow the user to pick start/end times
        user_time_range = st.slider(
            "Time Range (seconds)",
            min_value=min_time,
            max_value=max_time,
            value=(min_time, min_time + 1.0),
            step=0.1
        )
        start_time, end_time = user_time_range

        st.write(f"Selected Time Range: {start_time:.3f} s to {end_time:.3f} s")

        # ------------------------------
        # Show packet stats in this time range
        # ------------------------------
        if st.button("Compute Wireless Statistics for Selected Time Range"):
            st.subheader("Wireless Activity Summary")

            wifi_stats = compute_packet_statistics(wifi_df, start_time, end_time)
            ieee154_stats = compute_packet_statistics(ieee154_df, start_time, end_time)
            btle_stats = compute_packet_statistics(btle_df, start_time, end_time)
            btclassic_stats = compute_packet_statistics(btclassic_df, start_time, end_time)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Wi-Fi (802.11):**")
                st.write(wifi_stats)

                st.markdown("**IEEE 802.15.4:**")
                st.write(ieee154_stats)

            with col2:
                st.markdown("**Bluetooth LE:**")
                st.write(btle_stats)

                st.markdown("**Bluetooth BR/EDR:**")
                st.write(btclassic_stats)

            st.markdown("""
            Based on these stats, you can determine if there is an unusually high
            packet count in a specific technology that might cause interference
            with your Bluetooth LE Broadcast Audio.
            """)

# streamlit run le_bis_streamlit_mapping.py --server.headless true --server.address 192.168.147.128 --server.port 8501
if __name__ == "__main__":
    print('Running Streamlit app...')
    main()
