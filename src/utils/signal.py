import numpy as np
import scipy.signal as sg

# ------------------------------------------------------------------ #
# 1. Butterworth band-pass filter exactly like the paper (0.5-10 Hz) #
# ------------------------------------------------------------------ #
def bandpass_ppg(x: np.ndarray,
                 fs: int = 500,
                 low: float = 0.5,
                 high: float = 10.0,
                 order: int = 2) -> np.ndarray:
    """Return band-pass-filtered copy of the PPG vector."""
    nyq = 0.5 * fs
    b, a = sg.butter(order, [low / nyq, high / nyq], btype="band")
    return sg.filtfilt(b, a, x, padlen=3 * (max(len(b), len(a)) - 1))

# --------------------------------------- #
# 2. FIR decimation (anti-alias built in) #
# --------------------------------------- #
def downsample(x: np.ndarray, factor: int = 10) -> np.ndarray:
    """FIR-based decimate (zero-phase) – factor 10 ⇒ 500 Hz → 50 Hz."""
    if factor == 1:
        return x
    return sg.decimate(x, factor, ftype="fir", zero_phase=True)
