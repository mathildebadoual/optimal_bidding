import os

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
