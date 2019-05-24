"""
Run this file to test the scheme.You will get the accumulated loss curve
and the total loss will be printed in the python console.
Usage: python run_system.py file_name_of_data_in_csv C T strategy_code
Example:
    python run_system.py tripdata.csv 0.1 10 1
    This will run the system with C=0.1, T=10, using Passive Strategy,
    with the data in file "tripdata.csv" under the same directory.
Notice: this program doesn't select the first week's data.
        Add df.select_first_week() if you want.
"""

import sys

import pandas as pd
from matplotlib import pyplot as plt

from city_bike import CityBike

if __name__ == '__main__':

    if len(sys.argv) != 5:
        raise ValueError("wrong argument number")
    df = pd.read_csv(sys.argv[1])
    df = df.loc[:, ['tripduration', 'starttime', 'stoptime', 'start station id',
                    'start station latitude', 'start station longitude',
                    'end station id', 'end station latitude', 'end station longitude'
                    ]].copy()
    df = df.reset_index()
    C, T, strategy = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    system = CityBike(df, T=T, C=C, strategy=strategy)
    system.start_experiment()
    system.plot_loss_wrt_time()
    plt.title('Total loss w.r.t. time, C = {}, strategy: {}'.format(C, strategy))
    plt.xlabel('time/s')
    plt.ylabel('total loss')
    plt.legend()
