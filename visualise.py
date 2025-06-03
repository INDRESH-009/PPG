from tracemalloc import start
import numpy as np 
import matplotlib.pyplot as plt 

load_path = "/Users/indreshmr/dev/ppg2bp-net/data/raw_api2/10/signals.npz"
data = np.load(load_path)

ppg = data['ppg']
abp = data['abp']
fs = int(data['fs'])

start_idx = 30 * 60 * fs
end_idx = 60 * 60 * fs

ppg_segment = ppg[start_idx:end_idx]
abp_segment = abp[start_idx:end_idx]
t_minutes = np.arange(start_idx, end_idx) / fs / 60  # Fixed typo: np.arrange → np.arange
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t_minutes, ppg_segment, label='PPG', color='blue')
plt.title('PPG Signal Segment')
plt.xlabel('Time (minutes)')
plt.ylabel('Amplitude')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t_minutes, abp_segment, label='ABP', color='red')
plt.title('ABP Signal Segment')
plt.xlabel('Time (minutes)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()