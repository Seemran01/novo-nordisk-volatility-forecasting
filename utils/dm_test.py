import numpy as np
from scipy import stats

import numpy as np
from scipy import stats

def dm_test(actual, pred1, pred2, h=1, power=2):

    actual = np.array(actual)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)

    # Loss functions (MSE if power=2)
    loss1 = np.abs(actual - pred1) ** power
    loss2 = np.abs(actual - pred2) ** power

    d = loss1 - loss2
    mean_d = np.mean(d)

    # HAC-style variance adjustment (simplified for h=1 case)
    var_d = np.var(d, ddof=1)

    # DM statistic (h-adjusted)
    DM_stat = mean_d / np.sqrt((var_d / len(d)) * h)

    # p-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(abs(DM_stat)))

    return DM_stat, p_value