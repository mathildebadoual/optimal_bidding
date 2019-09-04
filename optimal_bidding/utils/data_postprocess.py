import os
import pandas as pd
import numpy as np

from datetime import timedelta, datetime


class TransitionMap():
    def __init__(self, energy_type):
        """
        Args:
          energy_type: string
        """
        self._energy_type = energy_type
        self._transition_maps = self._download_transition_maps()

    def get_transition_map_hour(self, hour):
        """Get the transition map for a specific hour

        Args:
          hour: timestamp

        Return:
          transition_map_hour: pandas.DataFrame
        """

        return self._transition_maps[hour]

    def get_next_state(self, current_state, hour):
        """From the transition map, generates the next state using the probabilities.

        Args:
          current_state: int?
          hour: timestamp

        Return:
          state: ?
        """
        print(self.get_transition_map_hour(hour))

    def _download_transition_maps(self):
        """Load the CSV transition maps

        JOE -- your input should match this

        Return:
          transition_maps: list of pandas.DataFrames indexed by hour
        """

        working_directory = os.getcwd()
        transition_maps = {}
        for hour in range(0, 23):
            filepath = (working_directory +
                        self._energy_type  + "/_hour_" + str(hour))
            transition_maps[hour] = pd.read_csv(filepath)

        return transition_maps


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
