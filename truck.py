"""
This module defines object truck.
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
        self.destination = None     # type: Station
        self.on_the_way = False

    def head_to(self, destination_station: Station, time: datetime.datetime):
        self.destination = destination_station
        dist = dist_bet(self.loc, self.destination)
        self.total_fee += self.C * dist
        self.arrival_time = time + datetime.timedelta(seconds=self.T * dist)
        self.on_the_way = True

    def load(self, take):
        true_take = min(take, self.capacity - self.carry)
        self.loc.K -= true_take
        self.carry += true_take

    def unload(self):
        true_drop = min(self.carry, self.loc.capacity - self.loc.K)
        self.loc.K += true_drop
        self.carry -= true_drop

    def arrive(self):
        t = self.arrival_time
        self.loc = self.destination
        self.on_the_way = False
        self.destination = None
        self.arrival_time = None
        self.unload()
        return t
