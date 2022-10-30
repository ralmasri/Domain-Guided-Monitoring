"""Remove or replace periodic events based on fourier analysis."""

import datetime
import util
import math
import numpy as np
import scipy.fftpack
import scipy.signal

round_int = lambda x: int(x + 0.5)

periodic_count = 5 # Required times of periodic event appearance to repeat
periodic_term = "1m" # Required term length of periodic event to repeat

# threshold for fourier analysis
threshold_spec = 0.4
threshold_eval = 0.1
threshold_restore = 0.5

def pretest(l_stat, binsize):
    """bool: Test that the data is enough long to judge as periodic data.
    """
    if sum(l_stat) == 0:
        return False
    return is_enough_long(l_stat, periodic_count, util.str2dur(periodic_term), binsize)


def remove(l_stat, binsize):
    data = np.array(l_stat)
    fdata = scipy.fftpack.fft(data)
    flag, interval = is_periodic(data, fdata, binsize, threshold_spec, threshold_eval)

    return flag, interval


def replace(l_stat, binsize):
    data = np.array(l_stat)
    fdata = scipy.fftpack.fft(data)
    flag, interval = is_periodic(data, fdata, binsize, threshold_spec, threshold_eval)
    if flag:
        data_filtered = part_filtered(fdata, threshold_spec)
        data_remain = restore_data(data, data_filtered, threshold_restore)
        return True, data_remain, interval
    else:
        return False, None, None


def is_periodic(data, fdata, binsize, th_spec, th_std):
    peak_order = 1
    peaks = 101

    dt = binsize.total_seconds()
    a_label = scipy.fftpack.fftfreq(len(data), d = dt)[1:int(0.5 * len(data))]
    a_spec = np.abs(fdata)[1:int(0.5 * len(data))]
    max_spec = max(a_spec)
    a_peak = scipy.signal.argrelmax(a_spec, order = peak_order)

    l_interval = []
    prev_freq = 0.0
    for freq, spec in (np.array([a_label, a_spec]).T)[a_peak]:
        if spec > th_spec * max_spec:
            interval = freq - prev_freq
            l_interval.append(interval)
            prev_freq = freq
        else:
            pass
    if len(l_interval) == 0:
        return False, None

    dist = np.array(l_interval[:(peaks - 1)])
    std = np.std(dist)
    mean = np.mean(dist)
    val = 1.0 * std / mean
    interval = round_int(1.0 / np.median(dist)) * datetime.timedelta(
            seconds = 1)
    return val < th_std, interval


def part_filtered(fdata, th_spec):
    a_spec = np.abs(fdata)
    max_spec = max(a_spec)
    
    fdata[a_spec <= th_spec * max_spec] = np.complex(0)
    data_filtered = np.real(scipy.fftpack.ifft(fdata))
    return data_filtered

def restore_data(data, data_filtered, th_restore):
    thval = th_restore * max(data_filtered)

    periodic_time = (data > 0) & (data_filtered >= thval) # bool
    periodic_cnt = np.median(data[periodic_time])
    data_periodic = np.zeros(len(data))
    data_periodic[periodic_time] = periodic_cnt
    data_remain = data - data_periodic

    return data_remain


def is_enough_long(l_stat, p_cnt, p_term, binsize):
    if sum(l_stat) < p_cnt:
        return False
    l_index = [ind for ind, val in enumerate(l_stat) if val > 0]
    length = (max(l_index) - min(l_index)) * binsize
    if length < p_term:
        return False
    else:
        return True


def power2(length):
    return 2 ** int(np.log2(length))

def power2ceil(length):
    return 2 ** math.ceil(np.log2(length))