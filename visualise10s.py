import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file for a single patient
file_path = '/Users/indreshmr/dev/ppg2bp-net/data/raw_api2/24/24.down50.npz'
data = np.load(file_path)

# Extract signals and sampling frequency
ppg = data['ppg_ds']
abp = data['abp_ds']
fs = int(data['fs'])  # Sampling rate (500 Hz)

# Calculate the index corresponding to 30 minutes
offset_seconds = 25 * 60  # 30 minutes in seconds
offset_samples = offset_seconds * fs  # Number of samples into the recording

# Ensure we have at least some data after 30 minutes
if offset_samples >= len(ppg):
    raise ValueError("Recording is shorter than 30 minutes, cannot extract post-30-min window.")

# Define a window of 10 seconds for visualization
window_duration_seconds = 10
window_samples = window_duration_seconds * fs

# Extract 10-second window starting at 30 minutes
start_idx = offset_samples
end_idx = offset_samples + window_samples

ppg_window = ppg[start_idx:end_idx]
abp_window = abp[start_idx:end_idx]

# Compute NaN counts within the 10-second window
nan_count_ppg = np.isnan(ppg_window).sum()
nan_count_abp = np.isnan(abp_window).sum()

# Time axis for plotting (in seconds)
time_axis = np.linspace(0, window_duration_seconds, window_samples, endpoint=False)

# Plot PPG and ABP for the 10-second window
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_axis, ppg_window)
plt.title("PPG Signal (10-second window starting at 30 min)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(time_axis, abp_window)
plt.title("ABP Signal (10-second window starting at 30 min)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# Print summary information
print(f"Sampling Frequency (fs): {fs} Hz")
print(f"Offset (samples) at 30 minutes: {offset_samples}")
print(f"Length of PPG signal: {len(ppg)} samples")
print(f"Length of ABP signal: {len(abp)} samples")
print(f"NaN count in PPG window: {nan_count_ppg} out of {window_samples} samples")
print(f"NaN count in ABP window: {nan_count_abp} out of {window_samples} samples")
