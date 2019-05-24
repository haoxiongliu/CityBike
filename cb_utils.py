"""
This module defines some utilities used in my implementation of City Bike System.
"""

import numpy as np
from station import Station
import pandas as pd
import datetime
from math import inf


def dist_bet(s1: Station, s2: Station):
    """This function calculates distances between two stations
    according to great-circle distance formula

    :param s1: station 1
    :param s2: station 2
    :return: distance between s1, s2
    """
    phi1, phi2, lambda1, lambda2 = np.deg2rad(s1.latitude), np.deg2rad(s2.latitude), \
                                   np.deg2rad(s1.longitude), np.deg2rad(s2.longitude)
    r = 6371  # mean radius of earth in kilometers
    delta_sigma = np.arccos(np.sin(phi1)*np.sin(phi2) +
                            np.cos(phi1)*np.cos(phi2)*np.cos(lambda1-lambda2) - 1e-10)
    distance = r*delta_sigma
    return distance


def log_normalize(x: np.ndarray):
    """used in color"""
    min_x, max_x = np.min(x), np.max(x)
    assert min_x < 0 < max_x
    k = min(abs(min_x), max_x)
    res = np.array([])
    for ele in x:
        if ele >= 0:
            res = np.append(res, max(0.5 + 0.5*np.log2(1 + ele/k), 1))
        else:
            res = np.append(res, min(0.5 - 0.5*np.log2(1 + ele/min_x), 0))
    return res


def select_first_week(df: pd.DataFrame):
    """Select the first seven days' data in the datafram"""
    res = []
    for ele in df.starttime:
        if datetime.datetime.strptime(ele, '%Y-%m-%d %H:%M:%S.%f').day <= 7:
            res.append(True)
        else:
            res.append(False)
    return res


def safety(k):
    """Measure of potential rewards. More details in Solutions.pdf"""
    if 0 <= k <= 5:
        res = 2*k
    elif 5 < k <= 15:
        res = k + 5
    elif 15 < k <= 20:
        res = 20
    elif 20 < k <= 25:
        res = 40 - k
    else:
        res = -inf
    return res


def supplyA(k):
    """Another measure of potential rewards. More details in Solutions.pdf"""
    return safety(k-10) - safety(k)
