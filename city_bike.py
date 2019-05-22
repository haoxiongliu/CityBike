"""
This module defines the CityBike class, containing all information and operations
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import cb_utils
from truck import Truck
import datetime
from collections import deque
import heapq
import math


class CityBike:

    def __init__(self, df: pd.DataFrame, T=1, C=1):

        # statistical information
        self.df = df
        self.id2station = dict()    # id to station instance
        self.station_set = set()    # set of id
        self.df_pop = None  # type: pd.DataFrame

        # finish id2station and station_set
        for i in range(len(self.df)):
            s = self.df["start station id"][i]
            if s not in self.station_set:
                self.station_set.add(s)
                self.id2station[s] = cb_utils.Station(s, self.df["start station latitude"][i],
                                                      self.df["start station longitude"][i])
            s = self.df["end station id"][i]
            if s not in self.station_set:
                self.station_set.add(s)
                self.id2station[s] = cb_utils.Station(s, self.df["end station latitude"][i],
                                                      self.df["end station longitude"][i])

        # members for running the scheme
        self.t = None   # type: datetime.datetime
        self.T, self.C = T, C
        self.truck = Truck(self.df["start station id"][0], T=T, C=C)
        self.dist = dict()  # s_id1, s_id2 distance
        for s1 in self.station_set:
            for s2 in self.station_set:
                self.dist[(s1, s2)] = \
                    cb_utils.dist_bet(self.id2station[s1], self.id2station[s2])
        self.loss = 0   # loss till t, 2*miss + 1*redirect + C*\sum d_{ij}
        temp_df = df.sort_values(by='starttime')[['stoptime', 'starttime',
                                                  'start station id', 'end station id']]
        self.start_sequence = deque(tuple(x) for x in temp_df.values)  # type: deque
        self.end_sequence = []

    def start_experiment(self):
        """Start the system

        :return:
        """
        while self.next_event():
            print("loss at time {}: {}".format(self.t, self.loss))
        print("total loss: {}".format(self.loss + self.truck.total_fee))

    def next_event(self):
        """Check the next nearest event
        It's the core part of this problem.
        We assume that the truck will not change its decision on its way.
        :return:    1 for continue, 0 for no more event
        """
        if not self.start_sequence:
            return 0    # if all customers start this week finished, we finish
        turn = 0    # 0 for start, 1 for end, 2 for truck
        min_t = self.start_sequence[0][1]
        if self.end_sequence and self.end_sequence[0][0] < min_t:
            turn, min_t = 1, self.end_sequence[0][0]
        if self.truck.arrival_time and self.truck.arrival_time < min_t:
            turn, min_t = 2, self.truck.arrival_time

        if turn == 0:
            trip = self.start_sequence.popleft()    # (stoptime, starttime, start id, end id)
            start_s = self.id2station[trip[2]]
            if start_s.num_of_bike == 0:
                self.loss += 2
                return 1
            heapq.heappush(self.end_sequence, trip)
            start_s.num_of_bike -= 1
            self.t = datetime.datetime.strptime(trip[1], '%Y-%m-%d %H:%M:%S.%f')

        elif turn == 1:
            trip = heapq.heappop(self.end_sequence)
            end_s = self.id2station[trip[3]]
            self.t = datetime.datetime.strptime(trip[0], '%Y-%m-%d %H:%M:%S.%f')
            # redirect
            if end_s.num_of_bike == end_s.capacity:
                # redirect the person to the nearest station with capacity
                dist_min = math.inf
                redirect_s = None  # type: cb_utils.Station
                for new_s_id in self.station_set - {trip[3]}:
                    new_s = self.id2station[new_s_id]
                    dist = cb_utils.dist_bet(new_s, end_s)
                    if dist < dist_min and new_s.num_of_bike < new_s.capacity:
                        redirect_s, dist_min = new_s, dist
                redirect_s.num_of_bike += 1
                self.loss += 1
            else:
                end_s.num_of_bike += 1
        else:
            self.t = self.truck.arrival_time
            self.truck.loc = self.truck.destination
            self.truck.on_the_way = False
            self.truck.destination = None

        # check if we need to assign task to the truck
        self.truck_strategy(0)
        return 1

    def truck_strategy(self, strategy_code):
        """Check the state of system and decides the next action
        It's the core part of this problem.
        We use strategy_code 0, 1, 2 to represent the strategy we design
        :return:
        """
        if strategy_code == 0:  # test strategy, pass
            pass

    """
    Following methods for question1
    """
    def draw_triplength_distribution(self):
        """Get trip length distribution"""
        fig, ax = plt.subplots()
        max_len = max(self.df["tripduration"])
        self.df["tripduration"].hist(bins=np.logspace(np.log10(60), np.log10(max_len), 100), ax=ax)
        ax.set_xscale('log')
        plt.title("Trip Length Distribution")
        plt.xlabel("Trip length")
        plt.ylabel("Number of trips")

    def get_popularity_dataframe(self):
        """
        Get station popularity
        popularity defined as bike out - bike in (net bike out)
        """

        popularity, bike_out, bike_in = dict(), dict(), dict()

        for s in self.station_set:
            bike_out[s] = bike_in[s] = popularity[s] = 0

        for i in range(len(self.df)):
            s = self.df["start station id"][i]
            bike_out[s] += 1
            popularity[s] += 1
            s = self.df["end station id"][i]
            bike_in[s] += 1
            popularity[s] -= 1

        station_list = list(self.station_set)
        station_list.sort()
        popularity_list = np.array([popularity[s] for s in station_list])
        bike_out_list = np.array([bike_out[s] for s in station_list])
        bike_in_list = np.array([bike_in[s] for s in station_list])

        self.df_pop = pd.DataFrame(np.transpose([popularity_list, bike_out_list, bike_in_list]),
                                   index=station_list,
                                   columns=["popularity", "bike out", "bike_in"])

    def draw_popularity_distribution(self):
        """draw populairty distribution over different stations"""
        if self.df_pop is None:
            raise ValueError("Should first get df_pop")
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        self.df_pop["popularity"].hist(bins=200, ax=ax)
        plt.title("Station Popularity Distribution")
        plt.xlabel("Popularity")
        plt.ylabel("Number of stations")

    def draw_popularity_map(self):
        """Draw Popularity Map"""
        if self.df_pop is None:
            raise ValueError("Should first get df_pop")
        fig, ax = plt.subplots()
        popularity = np.array(self.df_pop["popularity"])
        area = (np.log2(np.abs(popularity) + 1))**2
        color = 2*np.pi*cb_utils.log_normalize(popularity)
        index = self.df_pop.index.values
        stations = [self.id2station[i] for i in index]
        x = [s.longitude for s in stations]
        y = [s.latitude for s in stations]
        plt.scatter(x, y, s=area, c=color)
