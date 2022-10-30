import datetime
import math
from typing import List
import numpy as np


def remove_dist(l_dt: List[datetime.datetime], top_dt: datetime.datetime, end_dt: datetime.datetime, binsize: datetime.timedelta, threshold):
    assert isinstance(threshold, float)

    length = (end_dt - top_dt).total_seconds()
    bin_length = binsize.total_seconds()
    bins = math.ceil(1.0 * length / bin_length)
    a_stat = np.array([0] * int(bins))
    for dt in l_dt:
        cnt = int((dt - top_dt).total_seconds() / bin_length)
        assert cnt < len(a_stat)
        a_stat[cnt:] += 1

    a_linear = (np.array(range(int(bins))) + 1) * (1.0 * len(l_dt) / bins)
    val = sum((a_stat - a_linear) ** 2) / (bins * len(l_dt))
    return val < threshold
