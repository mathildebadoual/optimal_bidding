

class TransitionMap():
    def __init__(self):
        self._transition_map = self._download_transition_map()

    def get_transition_map_hour(self, hour):
        """Get the transition map for a specific hour

        Args:
          hour: timestamp

        Return:
          transition_map_hour: pandas.DataFrame
        """
        return self._transition_map

    def _download_transition_map(self):
        """Load the CSV transition map

        Return:
          transition_map: pandas.DataFrame
        """
        pass
