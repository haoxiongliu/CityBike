"""
This file implements the scheme
"""

from city_bike import CityBike
from cb_utils import select_first_week
import pandas as pd
import sys


if __name__ == '__main__':

    # if len(sys.argv) != 2:
    #     raise ValueError("wrong argument number")
    # df = pd.read_csv(sys.argv[1])
    df = pd.read_csv("tripdata.csv")
    df = df.loc[:, ['tripduration', 'starttime', 'stoptime', 'start station id',
                    'start station latitude', 'start station longitude',
                    'end station id', 'end station latitude', 'end station longitude'
                    ]].copy()
    df = df.loc[select_first_week, :].copy()
    df = df.reset_index()
    system = CityBike(df)
    system.start_experiment()
