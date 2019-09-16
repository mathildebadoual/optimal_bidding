import os
import pandas as pd
import csv
from os import listdir
from os.path import isfile, join
import warnings

from datetime import date, timedelta, datetime


class TransitionMap():
    def __init__(self):
        self._transition_maps = self._download_transition_maps()
        self._energy_type = _energy_type

    def get_transition_map_hour(self, hour):
        """Get the transition map for a specific hour

        Args:
          hour: timestamp

        Return:
          transition_map_hour: pandas.DataFrame
        """

        return self._transition_maps[hour]

    def _download_transition_maps(self):
        """Load the CSV transition maps

        JOE -- your input should match this  

        Return:
          transition_maps: list of pandas.DataFrames indexed by hour
        """

        working_directory = os.getcwd()

        transition_maps = {}

        for hour in range(0,23):
            filepath = (working_directory + str(self._energy_type)/
                + "_hour_" + str(hour))
            transition_maps[hour] = pd.read_csv(filepath)

        return transition_maps

def consolidate_csvs(folder_path, csv_path):
    """
    This will take a folder with MMS csvs and create a new csv with just the demand and energy data.
    """
#     with open(csv_path) as output:
#         writer = csv.writer(output)
#         # write header of output file
#         writer.writerow(["Timestamp", "RRP", "Total_Demand"])
    
    df = pd.DataFrame(columns=["Timestamp", "Region", "Price", "Demand"])
        
   # grab csvs from the specified folder
    onlycsvs = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.lower().endswith(".csv")]
    for csv_name in onlycsvs:
        with open(csv_name) as csvfile:
            reader = csv.reader(csvfile)
            
            demand_index = None
            price_index = None
            timestamp_index = None
            region_index = None
            for row in reader:
                if row[0] == "C":
                    # logging rows are useless
                    pass
                elif row[0] == "I":
                    # header row (sometimes the format of the csv changes in the middle so there can be multiple header rows)
                    demand_index = row.index("TOTALDEMAND")
                    price_index = row.index("RRP")
                    timestamp_index = row.index("SETTLEMENTDATE")
                    region_index = row.index("REGIONID")
                elif row[0] == "D":
                    # data row
                    data = {}
                    data["Timestamp"] = pd.to_datetime(row[timestamp_index])
                    data["Region"] = row[region_index]
                    data["Price"] = row[price_index]
                    data["Demand"] = row[demand_index]
                    df = df.append(data, ignore_index=True)
                else:
                    warnings.warn("Unrecognized row type in {}. Ignoring.".format(csv_name), UserWarning)

    df = df.set_index("Timestamp")
    # sort by date
    df = df.sort_index()
    # write to specified output
    df.to_csv(csv_path)   

def sample_day_solar_generation(month):
    """
    """

    generation_data = pd.read_csv((str(os.getcwd())+
            "/New_TMY3_Real_Years/solar_generation_australia.csv")).rename(columns={
        "Hours since 00:00 Jan 1":"Hrs",
        "Hourly Data: Electricity load (year 1) (kW)":"kW"
    })

    start = datetime(2018,1,1, 0,0,0)     
    delta = [timedelta(hours = hr) for hr in generation_data["Hrs"]]     # Create a time delta object from the number of days
    generation_data["time"]=[start+d for d in delta]

    month_data = generation_data.loc[[time.month == 2 for time in generation_data["time"]]]
    random_day = np.random.choice(30,1)

    day_data = month_data.loc[[time.day == random_day[0] for time in month_data["time"]]]

    return day_data








    