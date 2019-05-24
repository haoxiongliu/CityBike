"""
This module defines the CityBike class, containing all information and operations
"""

import datetime
from collections import deque
import heapq
import math

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import cb_utils
from cb_utils import safety, supplyA
from truck import Truck


class CityBike:

    def __init__(self, df: pd.DataFrame, T=math.inf, C=math.inf, strategy=0):

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
        self.T, self.C, self.strategy = T, C, strategy
        self.truck = Truck(self.id2station[self.df["start station id"][0]], T=T, C=C)
        self.dist = dict()  # s_id1, s_id2 distance
        for s1 in self.station_set:
            for s2 in self.station_set:
                self.dist[(s1, s2)] = \
                    cb_utils.dist_bet(self.id2station[s1], self.id2station[s2])
        self.neg_re = 0   # negtive reward till t
        temp_df = df.sort_values(by='starttime')[['stoptime', 'starttime',
                                                  'start station id', 'end station id']]
        # use these two sequences to detect events
        self.start_sequence = deque(tuple(x) for x in temp_df.values)  # type: deque
        self.end_sequence = []

        # members for observation
        self.loss_history = []  # (time, loss)
        self.station_popularity_history = []
        self.observe_s = None

    def start_experiment(self):
        """Start Experiment
        Every time we capture an event, handle it.
        Print accumulated loss every 1000 events.
        """
        i = 0
        # s_id = self.df['start station id'][0]   # observe a particular station
        # self.observe_s = s_id = 324      # the minimum popularity s_id
        # self.observe_s = s_id = 3164     # the maximum popularity s_id
        while self.next_event():
            i += 1
            if i % 1000 == 0:
                print("total loss at time {}: {}".
                      format(self.t, self.neg_re + self.truck.total_fee))
            self.loss_history.append((self.t, self.neg_re + self.truck.total_fee))
            # self.station_popularity_history.append((self.t, self.id2station[s_id].popularity))
        print("total loss: {}".format(self.neg_re + self.truck.total_fee))

    def next_event(self):
        """Check the next nearest event
        It's the core part of this problem.
        We assume that the truck will not change its decision on its way.
        :return:    1 for continue, 0 for no more event
        """
        if not self.start_sequence:
            return 0    # if all customers start this week finished, we finish
        turn = 0    # 0 for start, 1 for end, 2 for truck
        min_t = self.start_sequence[0][1]   # type: str
        if self.end_sequence and self.end_sequence[0][0] < min_t:
            turn, min_t = 1, self.end_sequence[0][0]
        if self.truck.arrival_time:
            truck_t = datetime.datetime.strftime(self.truck.arrival_time,
                                                 "%Y-%m-%d %H:%M:%S.%f")
            if truck_t < min_t:
                turn = 2

        # a trip starts
        if turn == 0:
            trip = self.start_sequence.popleft()   # (stoptime, starttime, start id, end id)
            start_s = self.id2station[trip[2]]
            if start_s.offer_bike() == 0:   # miss
                self.neg_re += 2
                return 1    # no need to check strategy
            else:
                heapq.heappush(self.end_sequence, trip)
                self.t = datetime.datetime.strptime(trip[1], '%Y-%m-%d %H:%M:%S.%f')

        # a trip ends
        elif turn == 1:
            trip = heapq.heappop(self.end_sequence)
            end_s = self.id2station[trip[3]]
            self.t = datetime.datetime.strptime(trip[0], '%Y-%m-%d %H:%M:%S.%f')
            # redirect
            if end_s.receive_bike() == 0:
                # redirect the person to the nearest station with capacity
                dest_list = []
                for s_id in self.station_set - {end_s.id}:
                    s = self.id2station[s_id]
                    if s.K < s.capacity:
                        dest_list.append((self.dist[(end_s.id, s.id)], s))
                dest = min(dest_list, key=lambda x: x[0])
                dest[1].K += 1
                self.neg_re += 1

                # it seems that the original redirecting algorithm becomes a bottleneck...
                # dist_min = math.inf
                # redirect_s = None  # type: cb_utils.Station
                # for new_s_id in self.station_set - {trip[3]}:
                #     new_s = self.id2station[new_s_id]
                #     dist = cb_utils.dist_bet(new_s, end_s)
                #     if dist < dist_min and new_s.K < new_s.capacity:
                #         redirect_s, dist_min = new_s, dist
                # redirect_s.K += 1
                # self.loss += 1

        # the truck arrives its destination
        else:
            self.t = self.truck.arrive()

        # check if we need to assign task to the truck
        self.truck_strategy(self.strategy)  # provide strategy code
        return 1

    def truck_strategy(self, strategy_code):
        """Check the state of system and decides the next action
        It's the core part of this problem.
        We use strategy_code 0, 1, 2 and so on to represent the strategy we design
        :return:
        """
        if strategy_code == 0:  # test strategy, pass
            pass
        if strategy_code == 1:  # passive strategy
            self.passive_strategy()
        if strategy_code == 2:  # active strategy
            self.active_strategy()

    def ER1(self, s1, s2):
        return 20 - 2*self.C*self.dist[(s1, s2)]

    def ER2(self, s1: cb_utils.Station, s2: cb_utils.Station):
        return 5 - 2*self.C*self.dist[(s1, s2)]

    def passive_strategy(self):
        """Refer to Solutions.pdf for details"""
        if self.truck.on_the_way:
            return

        # 1st level alert
        loc = self.truck.loc
        if loc.K <= 20:
            dest_list = []
            for s_id in self.station_set:
                s = self.id2station[s_id]
                if s.K > 20 and self.ER1(loc.id, s.id) > 0:
                    dest_list.append((self.dist[(loc.id, s.id)], s))
            if dest_list:
                dest = min(dest_list, key=lambda x: x[0])
                self.truck.head_to(dest[1], self.t)
                return
        else:
            alert_1st = []
            for s_id in self.station_set:
                s = self.id2station[s_id]
                if s.K < 5:
                    alert_1st.append((self.ER1(loc.id, s.id), s))
            if alert_1st:
                dest = max(alert_1st, key=lambda x: x[0])
                if dest[0] > 0:
                    self.truck.load(10)
                    self.truck.head_to(dest[1], self.t)
                    return

        # 2nd level alert
        alert_2nd = []
        for s_id in self.station_set:
            s = self.id2station[s_id]
            if s.K == s.capacity:
                alert_2nd.append((self.ER2(loc.id, s.id), s))
        if alert_2nd:
            dest = max(alert_2nd, key=lambda x: x[0])
            if dest[1] is not loc and dest[0] > 0:
                self.truck.head_to(dest[1], self.t)
            elif dest[1] is loc:
                dest_list = []
                for s_id in self.station_set:
                    s = self.id2station[s_id]
                    if s.K < 20 and self.ER2(loc.id, s.id) > 0:
                        dest_list.append((self.dist[(loc.id, s.id)], s))
                if dest_list:
                    dest = min(dest_list, key=lambda x: x[0])
                    self.truck.load(5)
                    self.truck.head_to(dest[1], self.t)
                    return

        return

    def ERT1(self, loc: cb_utils.Station, s: cb_utils.Station):
        carry_list = []
        dist = self.dist[(s.id, loc.id)]
        base = 2 * self.C + (safety(loc.K) + safety(s.K)) / dist
        for m in range(0, 11):
            carry_list.append((((safety(loc.K - m) + safety(s.K + m))/dist - base)/self.T, m))
        pair = max(carry_list)
        return pair

    def ERT2(self, loc: cb_utils.Station, s: cb_utils.Station):
        dist = self.dist[(loc.id, s.id)]
        return ((supplyA(s.K) - supplyA(loc.K))/dist - 2*self.C)/self.T

    def active_strategy(self):
        """Refer to Solutions.pdf for details"""
        if self.truck.on_the_way:
            return

        # ERT1 selection
        loc = self.truck.loc
        dest_list, dest_1 = [], None
        for s_id in self.station_set - {loc.id}:
            s = self.id2station[s_id]
            pair = self.ERT1(loc, s)
            if pair[0] > 0:
                dest_list.append((pair[0], pair[1], s))
        if dest_list:
            dest_1 = max(dest_list, key=lambda x: x[0])

        # ERT2 selection
        dest_list, dest_2 = [], None
        for s_id in self.station_set - {loc.id}:
            s = self.id2station[s_id]
            if self.ERT2(loc, s) > 0:
                dest_list.append((self.ERT2(loc, s), s))
        if dest_list:
            dest_2 = max(dest_list, key=lambda x: x[0])

        # decide 1 or 2
        if dest_1 and dest_2:
            if dest_1[0] > dest_2[0]:
                self.truck.load(dest_1[1])
                self.truck.head_to(dest_1[2], self.t)
            else:
                self.truck.head_to(dest_2[1], self.t)
        elif dest_1:
            self.truck.load(dest_1[1])
            self.truck.head_to(dest_1[2], self.t)
        elif dest_2:
            self.truck.head_to(dest_2[1], self.t)
        else:
            pass

    """
    Following methods are for plots.
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

        bike_out, bike_in = dict(), dict()

        for s in self.station_set:
            bike_out[s] = bike_in[s] = 0

        for i in range(len(self.df)):
            s = self.df["start station id"][i]
            bike_out[s] += 1
            s = self.df["end station id"][i]
            bike_in[s] += 1

        station_list = list(self.station_set)
        station_list.sort()
        bike_out_list = np.array([bike_out[s] for s in station_list])
        bike_in_list = np.array([bike_in[s] for s in station_list])

        self.df_pop = pd.DataFrame(np.transpose([bike_out_list, bike_in_list]),
                                   index=station_list,
                                   columns=["bike_out", "bike_in"])
        self.df_pop['popularity'] = self.df_pop['bike_out'] - self.df_pop['bike_in']
        self.df_pop['abs_pop'] = self.df_pop['bike_out'] + self.df_pop['bike_in']

    def draw_popularity_distribution(self):
        """draw populairty distribution over different stations"""
        if self.df_pop is None:
            raise ValueError("Should first get df_pop")

        fig, axes = plt.subplots(2, 2)

        axes[0, 0].set_yscale('log')
        self.df_pop["popularity"].hist(bins=200, ax=axes[0, 0])
        axes[0, 0].set_title("Station Popularity Distribution")
        axes[0, 0].set_xlabel("Popularity")
        axes[0, 0].set_ylabel("Number of stations")

        axes[0, 1].set_yscale('log')
        self.df_pop["bike_out"].hist(bins=200, ax=axes[0, 1])
        axes[0, 1].set_title("Bike Out Distribution")
        axes[0, 1].set_xlabel("Number of bikes out")
        axes[0, 1].set_ylabel("Number of stations")

        axes[1, 0].set_yscale('log')
        self.df_pop["bike_in"].hist(bins=200, ax=axes[1, 0])
        axes[1, 0].set_title("Bike In Distribution")
        axes[1, 0].set_xlabel("Number of bikes in")
        axes[1, 0].set_ylabel("Number of stations")

        axes[1, 1].set_yscale('log')
        self.df_pop["abs_pop"].hist(bins=200, ax=axes[1, 1])
        axes[1, 1].set_title("Absolute Popularity Distribution")
        axes[1, 1].set_xlabel("Number of bikes in and out")
        axes[1, 1].set_ylabel("Number of stations")

    def draw_popularity_map(self):
        """Draw Popularity Map"""
        if self.df_pop is None:
            raise ValueError("Should first get df_pop")
        fig, ax = plt.subplots()
        popularity = np.array(self.df_pop["popularity"])
        area = (np.log2(np.abs(popularity) + 1))**2
        color = np.pi*cb_utils.log_normalize(popularity)
        index = self.df_pop.index.values
        stations = [self.id2station[i] for i in index]
        x = [s.longitude for s in stations]
        y = [s.latitude for s in stations]
        plt.scatter(x, y, s=area, c=color, alpha=0.7)
        plt.title('Popularity Distribution Map')
        plt.xlabel('longitude/degree')
        plt.ylabel('latitude/degree')

    def plot_station_wrt_time(self):
        assert self.observe_s is not None
        t0 = self.station_popularity_history[0][0]
        xx = [(x[0] - t0).total_seconds() for x in self.station_popularity_history]
        yy = [x[1] for x in self.station_popularity_history]
        plt.plot(xx, yy)
        plt.title('popularity of station id {}'.format(self.observe_s))
        plt.xlabel('time/s')
        plt.ylabel('popularity')

    def plot_loss_wrt_time(self):
        t0 = self.loss_history[0][0]
        xx = [(x[0] - t0).total_seconds() for x in self.loss_history]
        yy = [x[1] for x in self.loss_history]
        plt.plot(xx, yy, label='T = {}'.format(self.T))
        # plt.title('Total loss w.r.t. time, C = {}, T = {}, strategy: {}'
        #           .format(self.C, self.T, self.strategy))
        # plt.xlabel('time/s')
        # plt.ylabel('total loss')
        # plt.savefig("graphs/Total_loss_C_{}_T_{}_strategy_{}.png".
        #             format(self.C, self.T, self.strategy))
