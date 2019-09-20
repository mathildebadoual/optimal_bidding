import os
import pandas as pd
import numpy as np

from datetime import timedelta, datetime
from collections import OrderedDict

directory_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CSV_PATH = os.path.join(directory_path, 'static/consolidated_data/')


def round_to_nearest(x, base):
    # helper function to bin values
    return base * np.round(x / base)


def get_energy_price(timestamp):
    """Return the price from the AEMO data

    Args:
      timestamp: Timestamp

    Return:
      price: float
    """
    df_path = CSV_PATH + 'June2018_30min.csv'
    df_30min = pd.read_csv(df_path)
    df_30min['Timestamp'] = df_30min['Timestamp'].apply(
        lambda x: pd.to_datetime(
            x, format='%Y-%m-%d %H:%M:%S', errors='ignore'))
    return df_30min[df_30min['Timestamp'] ==
                    timestamp]['Energy_Price'].values[0]


def get_energy_demand(timestamp):
    """Return the demand from the AEMO data

    Args:
      timestamp: Timestamp

    Return:
      demand: float
    """
    df_path = CSV_PATH + 'June2018_30min.csv'
    df_30min = pd.read_csv(df_path)
    df_30min['Timestamp'] = df_30min['Timestamp'].apply(
        lambda x: pd.to_datetime(
            x, format='%Y-%m-%d %H:%M:%S', errors='ignore'))
    return df_30min[df_30min['Timestamp'] ==
                    timestamp]['Energy_Demand'].values[0]


def get_raise_demand(timestamp):
    """Return the raise demand from the AEMO data

    Args:
      timestamp: Timestamp

    Return:
      demand: float
    """
    df_path = CSV_PATH + 'June2018_30min.csv'
    df_30min = pd.read_csv(df_path)
    df_30min['Timestamp'] = df_30min['Timestamp'].apply(
        lambda x: pd.to_datetime(
            x, format='%Y-%m-%d %H:%M:%S', errors='ignore'))
    return df_30min[df_30min['Timestamp'] ==
                    timestamp]['5min_Raise_Demand'].values[0]


def get_raise_price(timestamp):
    """Return the raise prices from the AEMO data

    Args:
      timestamp: Timestamp

    Return:
      demand: float
    """
    df_path = CSV_PATH + 'June2018_30min.csv'
    df_30min = pd.read_csv(df_path)
    df_30min['Timestamp'] = df_30min['Timestamp'].apply(
        lambda x: pd.to_datetime(
            x, format='%Y-%m-%d %H:%M:%S', errors='ignore'))
    return df_30min[df_30min['Timestamp'] ==
                    timestamp]['5min_Raise_Price'].values[0]


def get_low_demand(timestamp):
    """Return the low demand from the AEMO data

    Args:
      timestamp: Timestamp

    Return:
      demand: float
    """
    df_path = CSV_PATH + 'June2018_30min.csv'
    df_30min = pd.read_csv(df_path)
    df_30min['Timestamp'] = df_30min['Timestamp'].apply(
        lambda x: pd.to_datetime(
            x, format='%Y-%m-%d %H:%M:%S', errors='ignore'))
    return df_30min[df_30min['Timestamp'] ==
                    timestamp]['5min_Lower_Demand'].values[0]


def get_low_price(timestamp):
    """Return the low prices from the AEMO data

    Args:
      timestamp: Timestamp

    Return:
      demand: float
    """
    df_path = CSV_PATH + 'June2018_30min.csv'
    df_30min = pd.read_csv(df_path)
    df_30min['Timestamp'] = df_30min['Timestamp'].apply(
        lambda x: pd.to_datetime(
            x, format='%Y-%m-%d %H:%M:%S', errors='ignore'))
    return df_30min[df_30min['Timestamp'] ==
                    timestamp]['5min_Lower_Price'].values[0]


def get_energy_price_day_ahead(start_timestamp, horizon=48):
    """
    Args:
      timestamp: Timestamp of the start date
      horizon: non neg integer of number of time step

    Return:
      energy_price_values: numpy.Array of size horizon
    """
    last_day_of_June = pd.Timestamp(
        year=2018,
        month=6,
        day=30,
        hour=0,
        minute=30,
    )
    minutes = horizon * 30
    time_48_steps = pd.Timedelta(str(minutes) + ' min')
    end_timestamp = start_timestamp + time_48_steps

    if end_timestamp > last_day_of_June:
        end_timestamp = last_day_of_June

    df_path = CSV_PATH + 'June2018_30min.csv'
    df_30min = pd.read_csv(df_path)
    df_30min['Timestamp'] = df_30min['Timestamp'].apply(
        lambda x: pd.to_datetime(
            x, format='%Y-%m-%d %H:%M:%S', errors='ignore'))
    return df_30min[
        (df_30min['Timestamp'] >= start_timestamp)
        & (df_30min['Timestamp'] < end_timestamp)]['Energy_Price'].values


def get_energy_demand_day_ahead(start_timestamp, horizon=48):
    """
    Args:
      timestamp: Timestamp of the start date
      horizon: non neg integer of number of time step

    Return:
      energy_price_values: numpy.Array of size horizon
    """
    last_day_of_June = pd.Timestamp(
        year=2018,
        month=6,
        day=30,
        hour=0,
        minute=30,
    )
    minutes = horizon * 30
    time_48_steps = pd.Timedelta(str(minutes) + ' min')
    end_timestamp = start_timestamp + time_48_steps

    if end_timestamp > last_day_of_June:
        end_timestamp = last_day_of_June

    df_path = CSV_PATH + 'June2018_30min.csv'
    df_30min = pd.read_csv(df_path)
    df_30min['Timestamp'] = df_30min['Timestamp'].apply(
        lambda x: pd.to_datetime(
            x, format='%Y-%m-%d %H:%M:%S', errors='ignore'))
    return df_30min[
        (df_30min['Timestamp'] >= start_timestamp)
        & (df_30min['Timestamp'] < end_timestamp)]['Energy_Demand'].values


def get_raise_demand_day_ahead(start_timestamp, horizon=48):
    """
    Args:
      timestamp: Timestamp of the start date
      horizon: non neg integer of number of time step

    Return:
      energy_price_values: numpy.Array of size horizon
    """
    last_day_of_June = pd.Timestamp(
        year=2018,
        month=6,
        day=30,
        hour=0,
        minute=30,
    )
    minutes = horizon * 30
    time_48_steps = pd.Timedelta(str(minutes) + ' min')
    end_timestamp = start_timestamp + time_48_steps

    if end_timestamp > last_day_of_June:
        end_timestamp = last_day_of_June

    df_path = CSV_PATH + 'June2018_30min.csv'
    df_30min = pd.read_csv(df_path)
    df_30min['Timestamp'] = df_30min['Timestamp'].apply(
        lambda x: pd.to_datetime(
            x, format='%Y-%m-%d %H:%M:%S', errors='ignore'))
    return df_30min[
        (df_30min['Timestamp'] >= start_timestamp)
        & (df_30min['Timestamp'] < end_timestamp)]['5min_Raise_Demand'].values


def get_raise_price_day_ahead(start_timestamp, horizon=48):
    """
    Args:
      timestamp: Timestamp of the start date
      horizon: non neg integer of number of time step

    Return:
      energy_price_values: numpy.Array of size horizon
    """
    last_day_of_June = pd.Timestamp(
        year=2018,
        month=6,
        day=30,
        hour=0,
        minute=30,
    )
    minutes = horizon * 30
    time_48_steps = pd.Timedelta(str(minutes) + ' min')
    end_timestamp = start_timestamp + time_48_steps

    if end_timestamp > last_day_of_June:
        end_timestamp = last_day_of_June

    df_path = CSV_PATH + 'June2018_30min.csv'
    df_30min = pd.read_csv(df_path)
    df_30min['Timestamp'] = df_30min['Timestamp'].apply(
        lambda x: pd.to_datetime(
            x, format='%Y-%m-%d %H:%M:%S', errors='ignore'))
    return df_30min[
        (df_30min['Timestamp'] >= start_timestamp)
        & (df_30min['Timestamp'] < end_timestamp)]['5min_Raise_Price'].values


def get_low_price_day_ahead(start_timestamp, horizon=48):
    """
    Args:
      timestamp: Timestamp of the start date
      horizon: non neg integer of number of time step

    Return:
      energy_price_values: numpy.Array of size horizon
    """
    last_day_of_June = pd.Timestamp(
        year=2018,
        month=6,
        day=30,
        hour=0,
        minute=30,
    )
    minutes = horizon * 30
    time_48_steps = pd.Timedelta(str(minutes) + ' min')
    end_timestamp = start_timestamp + time_48_steps

    if end_timestamp > last_day_of_June:
        end_timestamp = last_day_of_June

    df_path = CSV_PATH + 'June2018_30min.csv'
    df_30min = pd.read_csv(df_path)
    df_30min['Timestamp'] = df_30min['Timestamp'].apply(
        lambda x: pd.to_datetime(
            x, format='%Y-%m-%d %H:%M:%S', errors='ignore'))
    return df_30min[
        (df_30min['Timestamp'] >= start_timestamp)
        & (df_30min['Timestamp'] < end_timestamp)]['5min_Lower_Price'].values


def get_low_demand_day_ahead(start_timestamp, horizon=48):
    """
    Args:
      timestamp: Timestamp of the start date
      horizon: non neg integer of number of time step

    Return:
      energy_price_values: numpy.Array of size horizon
    """
    last_day_of_June = pd.Timestamp(
        year=2018,
        month=6,
        day=30,
        hour=0,
        minute=30,
    )
    minutes = horizon * 30
    time_48_steps = pd.Timedelta(str(minutes) + ' min')
    end_timestamp = start_timestamp + time_48_steps

    if end_timestamp > last_day_of_June:
        end_timestamp = last_day_of_June

    df_path = CSV_PATH + 'June2018_30min.csv'
    df_30min = pd.read_csv(df_path)
    df_30min['Timestamp'] = df_30min['Timestamp'].apply(
        lambda x: pd.to_datetime(
            x, format='%Y-%m-%d %H:%M:%S', errors='ignore'))
    return df_30min[
        (df_30min['Timestamp'] >= start_timestamp)
        & (df_30min['Timestamp'] < end_timestamp)]['5min_Lower_Demand'].values


def get_transition_probabilities(df, column="Price", bin_size=10, timestep=30):
    """
    First discretizes the data by rounding the values to the nearest bin_size.
    Then, calculates transition probabilites from consecutive rounded values
    in data. Calculates separate transition probabilities for each timestep.

    Args:
      df: (pandas.Dataframe)
          Dataframe holding the data to be analyzed
      column: (string)
          The name of the column in df that holds the data.
      bin_size: (int)
          Values in in data will be rounded to the nearest bin_size.
      timestep: (int)
          The number of minutes in each timestep.

    Returns:
      all_probs: (list)
          A list of nested dictionaries.
          The index in this list corresponds to the time of day.
          An element with index n contains the transition probabilities
          for the timestep that starts at n*timestep minutes after 0:00.
          Each element is structured as:
              The keys of the outer dictionary are the "before" states.
              The values of the outer dictionary are also dictionaries
                    (inner dictionaries).
              The keys of the inner dictionaries are "after" states.
              The values of the inner dictionaries are transition
              probabilities.
    """
    data = df[column].values
    times = df.index
    all_counts = [dict() for x in range(int(24*60/timestep))]
    for i in range(len(data) - 1):
        matrix_index = int(times[i].hour + times[i].minute / timestep)
        counts = all_counts[matrix_index]
        current_state = round_to_nearest(data[i], bin_size)
        next_state = round_to_nearest(data[i+1], bin_size)
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


class TransitionMap():
    def __init__(self, datatype="Price", bin_size=10, timestep=30):
        self._transition_maps = self._load_transition_map_from_csv(
            datatype, bin_size, timestep)

    def get_transition_map_hour(self, hour):
        """Get the transition map for a specific hour

        Args:
          hour: timestamp

        Return:
          transition_map_hour: nested dictionary which is structured as:
                The keys of the outer dictionary are the "before" states.
                The values of the outer dictionary are also dictionaries
                      (inner dictionaries).
                The keys of the inner dictionaries are "after" states.
                The values of the inner dictionaries are transition
                probabilities.
        """
        return self._transition_maps[hour]

    def get_next_state(self, current_state, hour):
        """
        From the transition map, generates the next
        state using the probabilities.
        The current state is returned if it is not in
        the transition matrix for the specified hour.

        Args:
          current_state: int
                current state rounded to the nearest self.bin_size.
          hour: int

        Return:
            state: int
                Randomly sampled next state from the transition matrix.
        """

        transition_map_hour = self.get_transition_map_hour(hour)
        if current_state in transition_map_hour:
            probs = transition_map_hour[current_state]
        else:
            raise Warning(
                UserWarning,
                "Current state isn't in the transition map. \
                        Returning current state as next state."
            )
            return current_state

        cumulative_probs = OrderedDict()
        running_sum = 0
        for state, prob in probs.items():
            running_sum += prob
            cumulative_probs[state] = running_sum

        # randomly sample
        r = np.random.random()
        for state, cumulative in cumulative_probs.items():
            if r <= cumulative:
                return state

    def _load_transition_map_from_csv(self, datatype, bin_size,
                                      timestep):
        """Load the CSV transition maps

        Args:
          datatype: string
              The type of data that the transition map is supposed to reflect.
              Corresponds to a column name in the csv.
          bin_size: int
              Data values will be rounded to the nearest bin_size.
          timestep: int
              The number of minutes in each timestep.

        Return:
          transition_maps: list
                A list of nested dictionaries.
                The index in this list corresponds to the time of day.
                An element with index n contains the transition probabilities
                for the timestep that starts at n*timestep minutes after 0:00.
                Each element is structured as:
                    The keys of the outer dictionary are the "before" states.
                    The values of the outer dictionary are also dictionaries
                          (inner dictionaries).
                    The keys of the inner dictionaries are "after" states.
                    The values of the inner dictionaries are transition
                    probabilities.
        """

        df = pd.read_csv(CSV_PATH, index_col=["Timestamp"], parse_dates=True)
        transition_maps = get_transition_probabilities(df, datatype, bin_size,
                                                       timestep)
        return transition_maps


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
