import re
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


def convert_to_single_column(file_path):
    # 读取文件内容
    with open(file_path, "r") as file:
        content = file.read()

    # 从内容中提取所有数字
    numbers = re.findall(r'\d+', content)

    # 将数字从字符串格式转换为整数
    numbers = [int(num) for num in numbers]

    return numbers


def output_psd_txt(file_path):
    # 使用此函数
    # file_path = "state2.txt"
    values = convert_to_single_column(file_path)

    data_list = list(values)

    welch_tw = 0.8
    psd_sr = 1000
    freq_low_delta, freq_high_delta = 2, 4
    freq_low_theta, freq_high_theta = 4, 8
    freq_low_alpha, freq_high_alpha = 8, 14
    freq_low_beta, freq_high_beta = 14, 30
    freq_low_gamma, freq_high_gamma = 30, 100

    input_a, input_b, n = 8, 9, 0
    file_name = f"ttt_cvt.txt"
    if file_path == './data/state1.txt':
        file_name = f"./data/state1_cvt.txt"
    elif file_path == './data/state2.txt':
        file_name = f"./data/state2_cvt.txt"

    while n < 500:
        bs_data = baseline_shift(data_list, t_start=input_a, t_end=input_b)

        filter_delta = filtered(bs_data, f1=freq_low_delta, f2=freq_high_delta, sr=psd_sr)
        psd_delta = show_psd(filter_delta, welch_tw=welch_tw, sr=psd_sr)
        power_delta = clc_power(psd_delta[0], psd_delta[1], freq_low=freq_low_delta, freq_high=freq_high_delta)
        # Theta
        filter_theta = filtered(bs_data, f1=freq_low_theta, f2=freq_high_theta, sr=psd_sr)
        psd_theta = show_psd(filter_theta, welch_tw=welch_tw, sr=psd_sr)
        power_theta = clc_power(psd_theta[0], psd_theta[1], freq_low=freq_low_theta, freq_high=freq_high_theta)
        # Alpha
        filter_alpha = filtered(bs_data, f1=freq_low_alpha, f2=freq_high_alpha, sr=psd_sr)
        psd_alpha = show_psd(filter_alpha, welch_tw=welch_tw, sr=psd_sr)
        power_alpha = clc_power(psd_alpha[0], psd_alpha[1], freq_low=freq_low_alpha, freq_high=freq_high_alpha)
        # Beta
        filter_beta = filtered(bs_data, f1=freq_low_beta, f2=freq_high_beta, sr=psd_sr)
        psd_beta = show_psd(filter_beta, welch_tw=welch_tw, sr=psd_sr)
        power_beta = clc_power(psd_beta[0], psd_beta[1], freq_low=freq_low_beta, freq_high=freq_high_beta)
        # Gamma
        filter_gamma = filtered(bs_data, f1=freq_low_gamma, f2=freq_high_gamma, sr=psd_sr)
        psd_gamma = show_psd(filter_gamma, welch_tw=welch_tw, sr=psd_sr)
        power_gamma = clc_power(psd_gamma[0], psd_gamma[1], freq_low=freq_low_gamma, freq_high=freq_high_gamma)

        abs_power = [power_delta, power_theta, power_alpha, power_beta, power_gamma]
        power = f'{abs_power[0]},\t{abs_power[1]},\t{abs_power[2]},\t{abs_power[3]},\t{abs_power[4]}\n'

        # print(power)

        with open(file_name, "a") as file:
            # if not file_name == 'datas.txt':
            file.write(power)
        n = n + 1
        input_a = input_a + 0.05
        input_b = input_b + 0.05
