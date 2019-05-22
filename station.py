"""
This module defines class station and its methods.
"""


class Station:
    """
    Information of a station consists of id, latitude and longitude,
    capacity, number of bikes at the instant.
    """

    def __init__(self, s_id, latitude, longitude):
        self.id = s_id
        self.latitude = latitude
        self.longitude = longitude
        self.capacity = 25  # given capacity of station 25
        self.num_of_bike = 20  # given initial number of bikes

    def offer_bike(self):
        """
        when a customer starts at this station,
        return 0 for miss and 1 for success
        """
        if self.num_of_bike:
            self.num_of_bike -= 1
            ret = 1
        else:
            ret = 0
        return ret

    def receive_bike(self):
        """
        when a customer stops at this station,
        return 0 for redirect and 1 for success
        """
        if self.num_of_bike == self.capacity:
            ret = 0
        else:
            self.num_of_bike += 1
            ret = 1
        return ret

