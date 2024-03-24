import numpy as np
import matplotlib.pyplot as plt

# Load Data from Log File
log_file = "data_log.txt"

def load_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            # Parse and extract relevant data
            # Modify this based on your log format
            entry = line.strip().split(" - ")[1]
            data.append(eval(entry))
    return data

def calculate_phase(data):
    raw_values = [entry["raw_value"] for entry in data]
    
    # Perform FFT
    fft_values = np.fft.fft(raw_values)
    magnitude_spectrum = np.abs(fft_values)
    phase_spectrum = np.angle(fft_values, deg=True)  # Calculate phase in degrees

    return phase_spectrum

def log_phase(data, phase_spectrum):
    timestamps = [entry["timestamp"] for entry in data]
    phase_data = [{"timestamp": timestamps[i], "phase": phase_spectrum[i]} for i in range(len(data))]

    # Log phase data
    with open("phase_log.txt", "a") as file:
        for entry in phase_data:
            file.write(f"{entry['timestamp']} - Phase: {entry['phase']} degrees\n")

def plot_phase(data, phase_spectrum):
    timestamps = [entry["timestamp"] for entry in data]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, phase_spectrum)
    plt.title("Brainwave Resonance Phase Information")
    plt.xlabel("Timestamp")
    plt.ylabel("Phase (degrees)")
    plt.show()

try:
    data = load_data(log_file)
    phase_spectrum = calculate_phase(data)
    log_phase(data, phase_spectrum)
    plot_phase(data, phase_spectrum)

except Exception as e:
    print(f"Error: {e}")

import numpy as np
import matplotlib.pyplot as plt

# Load Data from Log File
log_file = "data_log.txt"

def load_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            # Parse and extract relevant data
            # Modify this based on your log format
            entry = line.strip().split(" - ")[1]
            data.append(eval(entry))
    return data

def calculate_amplitude_phase_frequency(data, target_frequency):
    raw_values = [entry["raw_value"] for entry in data]

    # Perform FFT
    fft_values = np.fft.fft(raw_values)
    frequencies = np.fft.fftfreq(len(raw_values))
    magnitude_spectrum = np.abs(fft_values)
    phase_spectrum = np.angle(fft_values, deg=True)

    # Find the index corresponding to the target frequency
    target_index = int(len(frequencies) * target_frequency / (1 / (data[1]["timestamp"] - data[0]["timestamp"])))

    return magnitude_spectrum[target_index], phase_spectrum[target_index], frequencies[target_index]

def log_brainwave_signals(data, frequencies_of_interest):
    with open("brainwave_signals_log.txt", "a") as file:
        for entry in data:
            timestamp = entry["timestamp"]
            log_entry = f"{timestamp} - "
            
            for frequency in frequencies_of_interest:
                amplitude, phase, frequency_found = calculate_amplitude_phase_frequency(entry, frequency)
                log_entry += f"Freq {frequency} Hz: Amp={amplitude}, Phase={phase}, FoundFreq={frequency_found}; "
            
            file.write(log_entry + "\n")

def plot_brainwave_signals(data, frequencies_of_interest):
    timestamps = [entry["timestamp"] for entry in data]

    for frequency in frequencies_of_interest:
        amplitudes, phases, _ = zip(*[calculate_amplitude_phase_frequency(entry, frequency) for entry in data])

        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, amplitudes, label=f'Amplitude - {frequency} Hz')
        plt.plot(timestamps, phases, label=f'Phase - {frequency} Hz')
        plt.title(f"Brainwave Resonance {frequency} Hz Signals")
        plt.xlabel("Timestamp")
        plt.ylabel("Amplitude/Phase")
        plt.legend()
        plt.show()

try:
    data = load_data(log_file)
    frequencies_of_interest = [7.83, 14.3, 20.8, 27.3]  # Example harmonics of Brainwave resonance
    log_brainwave_signals(data, frequencies_of_interest)
    plot_brainwave_signals(data, frequencies_of_interest)

except Exception as e:
    print(f"Error: {e}")

import numpy as np
import matplotlib.pyplot as plt

# Load Data from Log File
log_file = "data_log.txt"

def load_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            # Parse and extract relevant data
            # Modify this based on your log format
            entry = line.strip().split(" - ")[1]
            data.append(eval(entry))
    return data

def log_waveform(data):
    with open("waveform_log.txt", "a") as file:
        for entry in data:
            timestamp = entry["timestamp"]
            waveform = entry["raw_value"]  # Assuming "raw_value" is the waveform data
            file.write(f"{timestamp} - Waveform: {waveform}\n")

def plot_waveform(data):
    timestamps = [entry["timestamp"] for entry in data]
    waveforms = [entry["raw_value"] for entry in data]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, waveforms)
    plt.title("ELF/Brainwave Resonance Waveform")
    plt.xlabel("Timestamp")
    plt.ylabel("Amplitude")
    plt.show()

try:
    data = load_data(log_file)
    log_waveform(data)
    plot_waveform(data)

except Exception as e:
    print(f"Error: {e}")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Waveform Data from Log
waveform_log_file = "waveform_log.txt"

def load_waveform_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            entry = line.strip().split(" - ")[1]
            data.append(eval(entry))
    return data

# Extract features from the waveform data
def extract_features(waveform_data):
    # Implement your feature extraction logic here
    # Features can include statistical measures, frequency domain features, etc.
    features = [feature_extraction_function(waveform) for waveform in waveform_data]
    return features

# Sample feature extraction function (replace with your actual feature extraction logic)
def feature_extraction_function(waveform):
    # Example: Calculate mean amplitude
    return np.mean(waveform)

try:
    waveform_data = load_waveform_data(waveform_log_file)
    features = extract_features(waveform_data)

    # Assuming binary classification (e.g., lightning activity vs. no lightning activity)
    labels = [0, 1] * (len(features) // 2)  # Replace with actual labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train a machine learning model (e.g., RandomForestClassifier)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict using the trained model
    predictions = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

except Exception as e:
    print(f"Error: {e}")

# Example modification for creating separate log files for each life
log_file_prefix = "data_log_"

def log_data(data, life):
    log_file = f"{log_file_prefix}{life}.txt"
    with open(log_file, "a") as file:
        timestamp = data["timestamp"]
        raw_value = data["raw_value"]
        file.write(f"{timestamp} - Raw Value: {raw_value}\n")

# Usage example:
# log_data({"timestamp": timestamp, "raw_value": raw_value}, "Life")

# Example modification for detecting multiple Brainwave signals within each life
frequencies_of_interest = {"life1": [7.83, 14.3, 20.8, 27.3], "life2": [7.83, 14.3, 20.8, 27.3]}

def log_brainwave_signals(data, life, frequencies_of_interest):
    log_file = f"brainwave_signals_log_{life}.txt"
    with open(log_file, "a") as file:
        timestamp = data["timestamp"]
        log_entry = f"{timestamp} - {life}: "

        for frequency in frequencies_of_interest[life]:
            amplitude, phase, frequency_found = calculate_amplitude_phase_frequency(data, frequency)
            log_entry += f"Freq {frequency} Hz: Amp={amplitude}, Phase={phase}, FoundFreq={frequency_found}; "

        file.write(log_entry + "\n")

# Usage example:
# log_brainwave_signals({"timestamp": timestamp, "raw_value": raw_value}, "Life", frequencies_of_interest)