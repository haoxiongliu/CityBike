from collections import deque
from queue import PriorityQueue
import heapq
import numpy as np
from matplotlib import pyplot as plt
from city_bike import CityBike
import datetime

if __name__ == '__main__':
    system = None # type: CityBike
    t0 = system.station_popularity_history[0][0]
    xx = [(x[0] - t0).total_seconds() for x in system.station_popularity_history]
    yy = [x[1] for x in system.loss_history]
    plt.plot(xx, yy)
