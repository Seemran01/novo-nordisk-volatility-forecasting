import numpy as np
from scipy import stats

def dm_test(actual, pred1, pred2, h=1, power=2):
    """
    Diebold-Mariano Test
    actual: true values
    pred1: predictions model 1
    pred2: predictions model 2
    h: forecast horizon (1 for your case)
    power: 2 for MSE loss
    """

    actual = np.array(actual)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)

    # Loss differential
    d = np.abs(actual - pred1)**power - np.abs(actual - pred2)**power

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    DM_stat = mean_d / np.sqrt(var_d / len(d))

    # p-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(abs(DM_stat)))

    return DM_stat, p_value