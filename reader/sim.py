import pandas as pd
import numpy as np


class SimDataset:
    """
    Class loading and handling the simulation dataset
    """

    def __init__(
        self,
        path,
        train_flights,
        test_flights,
        to_normalize=None,
        initialize_train_test=True,
        add_flight_phase=False,
        filter_train_phases=False,
    ):
        self.path = path
        self.train_flights = train_flights
        self.test_flights = test_flights
        self.to_normalize = to_normalize
        self.abs_max = None
        self._train = None
        self._test = None
        if initialize_train_test:
            print("Loading train and test flights...")
            self.train
            print("Train flights loaded")
            self.test
            print("Test flights loaded")
        if add_flight_phase or filter_train_phases:
            self._train["Flight phase"] = get_flight_phase(self.train)
            self._test["Flight phase"] = get_flight_phase(self.test)
        if filter_train_phases:
            self.filter_train_phases()

    @property
    def train(self):
        if self._train is None:
            df = self._load(self.train_flights)
            if self.to_normalize is not None:
                normalization_suffix = "_normalized"
                new_column_names = [
                    col + normalization_suffix for col in self.to_normalize
                ]
                self.abs_max = df[self.to_normalize].abs().max()
                # lorsque le max est 0, on remplace par 1 pour éviter la division par 0
                self.abs_max[self.abs_max == 0] = 1
                df[new_column_names] = df[self.to_normalize] / self.abs_max
            df.index.names = ["Flight name", "Time, s"]
            self._train = df
        return self._train

    @train.setter
    def train(self, value):
        self._train = value

    @property
    def test(self):
        if self._test is None:
            df = self._load(self.test_flights)
            if self.to_normalize is not None:
                if self.abs_max is None:
                    self.train
                normalization_suffix = "_normalized"
                new_column_names = [
                    col + normalization_suffix for col in self.to_normalize
                ]
                df[new_column_names] = df[self.to_normalize] / self.abs_max
            df.index.names = ["Flight name", "Time, s"]
            self._test = df
        return self._test

    @test.setter
    def test(self, value):
        self._test = value

    @property
    def train_unnormalized(self):
        return self._load(self.train_flights)

    @property
    def test_unnormalized(self):
        return self._load(self.test_flights)

    def get_flight(self, flight_name):
        if flight_name in self.train_flights:
            return self.train[
                self.train.index.get_level_values("Flight name") == flight_name
            ]
        elif flight_name in self.test_flights:
            return self.test[
                self.test.index.get_level_values("Flight name") == flight_name
            ]
        else:
            raise ValueError(
                f"Flight {flight_name} not found in train or test flights."
            )

    def filter_train_phases(self, phases=None):
        """
        Filter the train dataset to keep only the flights in the specified phases
        Arguments:
        - phases : list of int, the phases to keep. See get_flight_phase for the phase values
        if None, keep all the flight phases but remove the "cruise" phase
        """
        if phases is None:
            phases = [0, 1, 3, 4, 5]
        self.train = self.train[self.train["Flight phase"].isin(phases)]

    def _load(self, flights):
        df = pd.concat([self._load_flight(flight) for flight in flights])
        df = df.set_index(["Flight_name", "Time, s"])
        return df

    def _load_flight(self, flight):
        df = pd.read_csv(f"{self.path}/{flight}.csv")
        df["Flight_name"] = flight
        return df.iloc[:-1]


def get_flight_phase(flight_df):
    """
    Get the flight phase from a flight dataframe
    The different phases are:
    - 0 : Departure : the plane climbs with a vertical speed greater than 3m/s and its flaps are extended
    - 1 : Climb : the plane climbs with a vertical speed greater than 3m/s and its flaps are retracted
    - 2 : Cruise : the plane flies horizontally with a vertical speed less than 3m/s and its roll angle is less than 10°
    - 3 : Turn : the plane flies with a vertical speed less than 3m/s and its roll angle is greater than 10°
    - 4 : Descent : the plane descends with a vertical speed less than 3m/s and its flaps are retracted
    - 5 : Approach : the plane descends with a vertical speed less than 3m/s and its flaps are extended
    Function arguments:
    - flight_df : a dataframe with the columns ...
    Returns:
    - a series with the flight phase for each row of the input dataframe
    """
    # Creation of the vertical speed column
    time_name = "Time, s"
    neg_alt_name = "Negative of c.m. altitude WRT Earth, ze = -h, m"
    flaps_name = "Position of the flaps, fl, rad"
    roll_name = "Roll angle of body WRT Earth, phir, rad"
    df_copy = flight_df.copy()
    df_copy["Time, s"] = df_copy.index.get_level_values("Time, s")
    df_copy["vertical speed"] = (
        -df_copy[neg_alt_name].diff() / df_copy[time_name].diff()
    )

    flap_lim = 0.5 * 38 * np.pi / 180  # 0.5 * 38° in rad
    roll_lim = 10 * np.pi / 180  # 10° in rad
    vertical_speed_lim = 3  # m/s

    def get_flight_phase_single_row(row):
        high_vertical_speed = abs(row["vertical speed"]) > vertical_speed_lim
        positive_vertical_speed = row["vertical speed"] > 0
        flaps_extended = row[flaps_name] > flap_lim
        high_roll = abs(row[roll_name]) > roll_lim
        if high_vertical_speed and positive_vertical_speed and flaps_extended:
            return 0
        elif high_vertical_speed and positive_vertical_speed and not flaps_extended:
            return 1
        elif not high_vertical_speed and not high_roll:
            return 2
        elif not high_vertical_speed and high_roll:
            return 3
        elif high_vertical_speed and not positive_vertical_speed and not flaps_extended:
            return 4
        elif high_vertical_speed and not positive_vertical_speed and flaps_extended:
            return 5
        else:
            raise ValueError("Flight phase not found")

    return df_copy.apply(get_flight_phase_single_row, axis=1)
