from scipy.signal import welch, filtfilt, butter, lfilter
from numpy import array, mean, logical_and, trapz


def baseline_shift(signal_uv: list, t_start, t_end, sr=1000, ):
    # Time window
    # t_start: lower limit of time window (s)
    sample_start = int(t_start * sr)
    # t_end: upper limit of time window (s)
    sample_end = int(t_end * sr)

    # Cutoff frequencies: f1, f2
    # Baseline shift of window
    signal_shift_window = array(signal_uv[sample_start:sample_end]) - mean(
        array(signal_uv[sample_start:sample_end]))
    return signal_shift_window


def filtered(signal_uv: list, f1=3, f2=30, sr=1000):
    # Digital Bandpass filtering with cutoff frequencies of f1=3 and f2=30 Hz using bandpass
    filtered_signal = bandpass(signal_uv, f1, f2, order=2, fs=sr)

    return filtered_signal


def show_psd(signal_uv: list, welch_tw=4, sr=1000):
    # Time Windows for Welchs method
    win = welch_tw * sr  # welch_tw seconds time windows.

    # FFT with time windows using scipy.signal.welch
    freq_axis, power_spect = welch(signal_uv, sr, nperseg=win)
    return freq_axis, power_spect


def clc_power(freq_axis, power_spect, freq_low=8, freq_high=12):
    # Define Frequency Band limits: freq_low, freq_high
    # Find the intersection Values of the alpha band in the frequency vector [Eyes Closed]
    idx_alpha = logical_and(freq_axis >= freq_low, freq_axis <= freq_high)
    # Frequency Resolution
    try:
        freq_res = freq_axis[1] - freq_axis[0]
    except IndexError:
        freq_res = 1

    # Compute the Absolute Power with numpy.trapz:
    alpha_power = trapz(power_spect[idx_alpha], dx=freq_res)
    alpha_power = round(alpha_power, 2)
    return alpha_power


def bandpass(s, f1, f2, order=2, fs=1000.0, use_filtfilt=False):
    [b, a] = butter(Wn=[f1 * 2 / fs, f2 * 2 / fs], btype='bandpass', N=order, output='ba')

    if use_filtfilt:
        return filtfilt(b, a, s)

    return lfilter(b, a, s)
