import logging
import random as rd

import pandas as pd
import numpy as np

from .preprocessing import preprocess_train_test_df


class FromPandasDataset:
    """
    Class loading and preprocessing the flights dataset
    """

    def __init__(
        self,
        dataframe,
        train_flight_names=None,
        data_reduction=None,
        to_normalize=None,
        add_flight_phase=False,
        filter_train_phases=False,
        flight_phase_parameters=dict(),
    ):
        """
        Arguments :
        - self
        - dataframe : pd.DataFrame of the flights. The index of the DataFrame has to be a multi-index
        with the flight name and the time
        - resampling_interval, optional : interval in indexes at which the data will be resampled
        - to_normalize, optional : list of the columns needing a normalization. The normalized columns
        are added to the train and test dataframes with the suffix "_normalized".
        - add_flight_phase : if True, adds a "Flight phase" column with the current phase of the plane.
        The details of the flight phases can be found in "get_flight_phase"
        - filter_train_phases : if True, the cruise phase is removed from the train dataframe
        """
        self.logger = logging.getLogger("FromPandasDataset")
        flight_names = list(dataframe.index.levels[0])
        flight_names.sort()
        if train_flight_names is None:
            rd.Random(1).shuffle(
                flight_names
            )  # Utilisation d'une seed pour la reproductibilité
            self.train_flight_names = flight_names[: int(0.7 * len(flight_names))]
            self.test_flight_names = flight_names[int(0.7 * len(flight_names)) :]
        else:
            self.train_flight_names = train_flight_names
            self.test_flight_names = list(set(flight_names) - set(train_flight_names))
        self.logger.info("Loading train and test flights...")
        self._train, self.train_abs_max = preprocess_train_test_df(
            dataframe.loc[self.train_flight_names],
            to_normalize=to_normalize,
            data_reduction=data_reduction,
        )
        self.logger.info("Train flights loaded")
        self._test, self.test_abs_max = preprocess_train_test_df(
            dataframe.loc[self.test_flight_names],
            to_normalize=to_normalize,
            data_reduction=data_reduction,
        )
        self.logger.info("Test flights loaded")
        self.data_reduction = data_reduction
        self.to_normalize = to_normalize
        if add_flight_phase or filter_train_phases:
            self.logger.info("Adding flight phase")
            self._train["Flight phase"] = get_flight_phase(
                self.train, **flight_phase_parameters
            )
            self._test["Flight phase"] = get_flight_phase(
                self.test, **flight_phase_parameters
            )
        if filter_train_phases:
            self.logger.info("Filtering train phases")
            self.filter_train_phases()

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, value):
        self._test = value

    def get_flight(self, flight_name):
        if flight_name in self.train_flight_names:
            return self.train.loc[flight_name]
        elif flight_name in self.test_flight_names:
            return self.test.loc[flight_name]
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


def get_flight_phase(
    flight_df,
    vertical_speed_lim=3,
    roll_lim=10 * np.pi / 180,
    flap_lim=0.5,
    alt_name="ZBPIL",
    flaps_name="DVOLIG",
    roll_angle_name="PHI",
):
    """
    Get the flight phase from a flight dataframe
    The different phases are:
    - 0 : Departure : the plane climbs with a vertical speed greater than vertical_speed_lim and its flaps are extended
    - 1 : Climb : the plane climbs with a vertical speed greater than vertical_speed_lim and its flaps are retracted
    - 2 : Cruise : the plane flies horizontally with a vertical speed less than vertical_speed_lim and its roll angle is less than roll_lim
    - 3 : Turn : the plane flies with a vertical speed less than vertical_speed_lim and its roll angle is greater than roll_lim
    - 4 : Descent : the plane descends with a vertical speed less than vertical_speed_lim and its flaps are retracted
    - 5 : Approach : the plane descends with a vertical speed less than vertical_speed_lim and its flaps are extended
    Function arguments:
    - flight_df : the flight dataframe
    - vertical_speed_lim, optional : the vertical speed in m/s limit to consider the vertical speed as high
    - roll_lim, optional : the roll angle limit in deg to consider the roll angle as high
    - flap_lim, optional : the value above which the flaps are no longer considered as retracted. Since
    the column is a Boolean, the limit value is 0.5, to be changed if the flaps are given as an angle.
    - alt_name, flaps_name, roll_angle name, optionnals : the names of the columns of altitude, flaps and roll angle in the dataframe
    Returns:
    - a series with the flight phase for each row of the input dataframe
    """
    # Creation of the vertical speed column
    temp_df = pd.DataFrame()
    temp_df["timestamp"] = flight_df.index.get_level_values(1)
    temp_df.index = flight_df.index
    # timestamp = flight_df.index.get_level_values(1)
    temp_df["vertical speed"] = (
        flight_df[alt_name].diff() / temp_df["timestamp"].diff().dt.total_seconds()
    )
    temp_df["flaps"] = flight_df[flaps_name]
    temp_df["roll angle"] = flight_df[roll_angle_name]
    temp_df.index = flight_df.index

    def get_flight_phase_single_row(row):
        high_vertical_speed = abs(row["vertical speed"]) > vertical_speed_lim
        positive_vertical_speed = row["vertical speed"] > 0
        flaps_extended = row["flaps"] > flap_lim
        high_roll = abs(row["roll angle"]) > roll_lim
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

    return temp_df.apply(get_flight_phase_single_row, axis=1)
