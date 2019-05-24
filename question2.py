"""
We can use
"""

from city_bike import CityBike
from cb_utils import select_first_week
import pandas as pd
import sys
from matplotlib import pyplot as plt
import numpy as np


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
    C, strategy = 0.1, 2
    loss_list = []
    for T in [10, 100, 1000]:
        system = CityBike(df, T=T, C=C, strategy=strategy)
        system.start_experiment()
        system.plot_loss_wrt_time()
        loss_list.append(system.neg_re + system.truck.total_fee)
    plt.title('Total loss w.r.t. time, C = {}, strategy: {}'.format(C, strategy))
    plt.xlabel('time/s')
    plt.ylabel('total loss')
    plt.legend()
    plt.savefig("graphs/Total_loss_C_{}__strategy_{}.png".format(C, strategy))
    print(loss_list)
