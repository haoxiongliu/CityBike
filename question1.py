"""
This file answers question 1.
Example Usage:
    quetion1.py tripdata.csv
"""

from city_bike import CityBike
import pandas as pd
import sys

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError("wrong argument number")
    df = pd.read_csv(sys.argv[1])

    system = CityBike(df)
    system.draw_triplength_distribution()
    system.get_popularity_dataframe()
    system.draw_popularity_distribution()
    system.draw_popularity_map()
