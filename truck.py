"""
This module defines class truck
"""

from cb_utils import dist_bet, Station
import datetime


class Truck:

    def __init__(self, start_station: Station, T, C):
        self.capacity = 10
        self.carry = 0
        self.loc = start_station
        self.T = T  # T given in the problem
        self.C = C  # C given in the problem
        self.total_fee = 0  # C*Dist()
        self.arrival_time = None
        self.destination = None
        self.on_the_way = False

    def head_to(self, destination_station: Station, time: datetime.datetime):
        self.destination = destination_station
        dist = dist_bet(self.loc, self.destination)
        self.total_fee += self.C * dist
        self.arrival_time = time + datetime.timedelta(seconds=self.T * dist)
        self.on_the_way = True

    def take_or_drop(self, take):
        self.loc.num_of_bike -= take
        self.carry += take
