# Optimizing Bluetooth Audio Broadcasts with Python

This repository contains a Python-based solution that leverages a human-in-the-loop approach to optimize Bluetooth audio broadcasts in public spaces. By combining interactive visualizations, audio playback, and wireless packet analysis, the application provides actionable insights for diagnosing and improving audio broadcast performance in environments with multiple overlapping wireless technologies.

The project shows how Python tools like Streamlit, pandas, NumPy, and matplotlib can transform imperfect wireless capture data into insights that support real-time diagnostics and optimization of Bluetooth audio broadcasts.


## Project Overview

Bluetooth Low Energy (LE) Broadcast Audio can be challenging in public settings such as airports, malls, gyms, and workspaces due to interference from Wi-Fi, IEEE 802.15.4 devices, Bluetooth Classic, and other RF sources. This project demonstrates how to:
- **Visualize and compare audio signals:** Compare the original broadcast audio with the captured audio using interactive waveform plots and audio players.
- **Analyze wireless captures:** Load CSV files containing wireless capture data for Wi-Fi (802.11), IEEE 802.15.4, Bluetooth LE, and Bluetooth BR/EDR.
- **Compute packet statistics:** Allow the user to select a time range on the audio waveform and compute packet counts for the different technologies within that interval.
- **Facilitate human-in-the-loop diagnostics:** Provide visual and auditory feedback to help diagnose interference issues and inform adjustments to broadcast parameters.

The idea behind this project is to blend automated analysis with human expertise—making it easier to work with messy, real-world data when machine learning or purely statistical models fall short.

## Features

- **Data Upload and Parsing:**  
  Loads CSV files for wireless captures and WAV files for both original and captured broadcast audio.

- **Interactive Audio Analysis:**  
  Displays side-by-side audio players and waveform comparisons to assess broadcast quality.

- **Time Range Selection:**  
  Users can select a specific time interval on the audio waveform to analyze wireless packet activity.

- **Wireless Packet Statistics:**  
  Computes and displays packet counts for Wi-Fi, IEEE 802.15.4, Bluetooth LE, and Bluetooth BR/EDR within the selected time range.

## Repository Structure

```
.
├── le_bis_streamlit.py      # Main Streamlit application for analysis
└── data/                    # Directory containing sample CSV and WAV files (see file paths in code)
    └── le_bis/
         ├── wi-fi_802_11.csv
         ├── 802_15_4.csv
         ├── bluetooth_le.csv
         ├── bluetooth_br_edr.csv
         ├── le_bis_chirp_01_stream2.wav
         └── le_bis_chirp_01_stream2_mod_01.wav
```

## Requirements

Make sure you have Python 3 installed along with the following packages:

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://scipy.org/)
- [SoundFile](https://pysoundfile.readthedocs.io/)

You can install these dependencies using pip. For example:

```bash
pip install streamlit pandas numpy matplotlib scipy soundfile
```

## How to Run the Application

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/potto216/wps-examples.git
   cd wps-examples
   ```

2. **Prepare the Data:**

   Ensure that your data files (CSV for wireless captures and WAV files for audio) are stored in the appropriate locations as referenced in the code (e.g., `data/le_bis/`).

3. **Run the Streamlit App:**

   Launch the application with the following command:

   ```bash
   streamlit run le_bis_streamlit.py
   ```

   Optionally, you can specify additional server options. For example:

   ```bash
   streamlit run le_bis_streamlit.py --server.headless true --server.address 192.168.147.128 --server.port 8501
   ```

4. **Interact with the App:**

   - **Upload Data:**  
     The app automatically loads the CSV and WAV files from the predefined paths. In a production setting, you may replace these with Streamlit file upload widgets.

   - **Audio Playback and Visualization:**  
     Listen to both the original and captured audio files. Examine the waveform plots to identify areas of interest.

   - **Select Time Range:**  
     Use the slider to select a specific time window on the audio waveform.

   - **Compute Packet Statistics:**  
     Click the button to compute and display wireless packet counts for the selected time range. These statistics can help identify potential interference issues affecting the audio broadcast.

