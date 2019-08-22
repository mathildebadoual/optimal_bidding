import os
import pandas as pd
import csv
from os import listdir
from os.path import isfile, join
import warnings

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
    