import os
import pandas as pd
import numpy as np
import csv
from os.path import isfile, join, abspath, dirname
import warnings

from datetime import timedelta, datetime


def consolidate_csvs(data_path, output_folder, output_prefix):
    """
    Takes a folder with MMS csvs and create a new csv
    with just the demand and energy data.
    Assumes that all csvs in the directory are
    to be used and follow the MMS format.
    Warns the user if there are missing datetimes.

    Parameters
    ----------
    data_path: string
        Absolute path to directory containing csvs with MMS data.

    output_folder: string
        Absolute path to directory where outputted csvs will be created.

    output_prefix: string
        Prefix for the filename of the outputted csvs.

    Returns
    -------
    None

    """

    five_min_df = pd.DataFrame(
        columns=["Timestamp", "Region", "Price", "Demand"])
    thirty_min_df = pd.DataFrame(
        columns=["Timestamp", "Region", "Price", "Demand"])

    # grab csvs from the specified folder
    onlycsvs = [
        join(data_path, f) for f in os.listdir(data_path)
        if isfile(join(data_path, f)) and f.lower().endswith(".csv")
    ]
    for csv_name in onlycsvs:
        print("Reading {}".format(csv_name.split("/")[-1]))
        with open(csv_name) as csvfile:
            reader = csv.reader(csvfile)

            demand_index = None
            price_index = None
            timestamp_index = None
            region_index = None

            freq = None

            for row in reader:
                if row[0] == "C":
                    # logging rows are useless
                    pass
                elif row[0] == "I":
                    # header row (sometimes the format of the csv changes
                    # in the middle so there can be multiple header rows)
                    demand_index = row.index("TOTALDEMAND")
                    price_index = row.index("RRP")
                    timestamp_index = row.index("SETTLEMENTDATE")
                    region_index = row.index("REGIONID")
                    if row[1] == "DREGION":
                        freq = 5
                    elif row[1] == "TREGION":
                        freq = 30
                    else:
                        freq = None
                elif row[0] == "D":
                    # data row
                    data = {}
                    data["Timestamp"] = pd.to_datetime(row[timestamp_index])
                    data["Region"] = row[region_index]
                    data["Price"] = row[price_index]
                    data["Demand"] = row[demand_index]
                    if freq == 5:
                        five_min_df = five_min_df.append(data,
                                                         ignore_index=True)
                    elif freq == 30:
                        thirty_min_df = thirty_min_df.append(data,
                                                             ignore_index=True)
                    else:
                        warnings.warn(
                            "Unrecognized frequency in {}. Ignoring row.".
                            format(csv_name), UserWarning)

                else:
                    warnings.warn(
                        "Unrecognized row type in {}. Ignoring.".format(
                            csv_name), UserWarning)

    five_min_df = five_min_df.set_index("Timestamp")
    thirty_min_df = thirty_min_df.set_index("Timestamp")
    # sort by date
    five_min_df = five_min_df.sort_index()
    thirty_min_df = thirty_min_df.sort_index()

    # drop duplicates
    five_min_df = five_min_df.drop_duplicates()
    thirty_min_df = thirty_min_df.drop_duplicates()

    # write to specified output
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    five_min_df.to_csv(join(output_folder, output_prefix + "_5min.csv"))
    thirty_min_df.to_csv(join(output_folder, output_prefix + "_30min.csv"))


def get_regional_data_from_csv(csv_path, region_ID):
    df = pd.read_csv(csv_path, index_col=["Timestamp"], parse_dates=True)
    # filter for region
    df = df[df["Region"] == region_ID]
    return df


def round_to_nearest(x, base):
    # helper function to bin values
    return base * np.round(x / base)


def get_transition_probabilities(df, column="Price", bin_size=10, timestep=30):
    """
    First discretizes the data by rounding the values to the nearest bin_size.
    Then, calculates transition probabilites from consecutive rounded values
    in data. Calculates separate transition probabilities for each timestep.

    Parameters
    ----------
    df: dataframe
        Dataframe holding the data to be analyzed
    column: size
        The name of the column in df that holds the data.
    bin_size: int
        Values in in data will be rounded to the nearest bin_size.
    timestep: int
        The number of minutes in each timestep.

    Returns
    -------
    all_probs: list
        A list of nested dictionaries.
        The index in this list corresponds to the time of day.
        An element with index n contains the transition probabilities
        for the timestep that starts at n*timestep minutes after 0:00.
        Each element is structured as:
            The keys of the outer dictionary are the "before" states.
            The values of the outer dictionary are also dictionaries
                  (inner dictionaries).
            The keys of the inner dictionaries are "after" states.
            The values of the inner dictionaries are transition probabilities.
    """
    data = df[column].values
    times = df.index
    all_counts = [dict() for x in range(int(24*60/timestep))]
    for i in range(len(data) - 1):
        matrix_index = int(times[i].hour + times[i].minute / timestep)
        counts = all_counts[matrix_index]
        current_state = data[i]
        next_state = data[i+1]
        if current_state in counts:
            temp = counts[current_state]
            if next_state in temp:
                temp[next_state] += 1
            else:
                temp[next_state] = 1
            temp["total"] += 1
        else:
            counts[current_state] = {next_state: 1, "total": 1}

    all_probs = [dict() for x in range(int(24*60/timestep))]
    for i in range(len(all_counts)):
        probs = all_probs[i]
        counts = all_counts[i]
        for state in counts:
            temp = counts[state]
            total_count = temp["total"]
            transitions = {}
            for next_state in temp:
                if next_state != "total":
                    transitions[next_state] = temp[next_state] / total_count
            probs[state] = transitions
    return all_probs


def sample_day_solar_generation(month):
    """
    """
    generation_data = pd.read_csv(
        (str(os.getcwd()) +
         "/New_TMY3_Real_Years/solar_generation_australia.csv")).rename(
             columns={
                 "Hours since 00:00 Jan 1": "Hrs",
                 "Hourly Data: Electricity load (year 1) (kW)": "kW"
             })

    start = datetime(2018, 1, 1, 0, 0, 0)
    delta = [timedelta(hours=hr) for hr in generation_data["Hrs"]
             ]  # Create a time delta object from the number of days
    generation_data["time"] = [start + d for d in delta]

    month_data = generation_data.loc[[
        time.month == 2 for time in generation_data["time"]
    ]]
    random_day = np.random.choice(30, 1)

    day_data = month_data.loc[[
        time.day == random_day[0] for time in month_data["time"]
    ]]

    return day_data


if __name__ == '__main__':
    dir_path = dirname(dirname(abspath(__file__)))
    static_path = join(dir_path, 'static')
    output_folder = join(static_path, 'consolidated_data')
    data_path = join(static_path, 'PUBLIC_PRICES_20180601')
    output_prefix = 'June2018'
    consolidate_csvs(data_path, output_folder, output_prefix)
