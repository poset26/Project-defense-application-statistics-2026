"""Chi-square and Student-t helpers used instead of SciPy."""

import numpy as np

from anomaly_guard.core.stat_lite import chisquare, student_t_ppf


def test_chisquare_known_statistic():
    stat, p_value = chisquare(f_obs=np.array([10.0, 20.0, 30.0]), f_exp=np.array([20.0, 20.0, 20.0]))
    assert stat == np.sum((np.array([10, 20, 30]) - 20) ** 2 / 20)
    assert 0.0 < p_value < 1.0


def test_student_t_two_sided_95_is_near_normal_for_large_df():
    crit = student_t_ppf(0.975, 500)
    assert 1.96 < crit < 1.97
