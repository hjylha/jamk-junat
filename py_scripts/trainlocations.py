
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .api_fcns import request_data
from .db_fcns import save_df_to_db, get_df_from_db, EXTRA_DB_PATH
from .data_fcns import check_station_stops, get_raw_train_location_data, select_data_for_train
from .data_fcns import get_acceleration, from_speed_to_distance, get_stops, scale_distance


def train_dict(date, train_num, train_type, train_category):
    return {
        "departureDate": date,
        "trainNumber": train_num,
        "trainType": train_type,
        "trainCategory": train_category
    }


def get_stop_times(timetable, extra_time=60):
    tt = timetable[timetable["trainStopping"]]

    station_stop_times = []
    stations = tt["stationShortCode"].unique()
    # num_of_stations = len(stations)
    # for i, station in enumerate(stations):
    for station in stations:
        tt_s = tt[tt["stationShortCode"] == station]
        time_arrival = tt_s["actualTime"].min() - pd.Timedelta(seconds=extra_time)
        time_departure = tt_s["actualTime"].max() + pd.Timedelta(seconds=extra_time)
        station_stop_times.append({
            "stationShortCode": station,
            "arrival_time": time_arrival,
            "departure_time": time_departure
        })
    return pd.DataFrame(station_stop_times)


# sama nimi kuin data_fcns-funktiolla
def get_station(row, station_stop_times):
    if row["speed"] > 0:
        return
    sst = station_stop_times[(station_stop_times["arrival_time"] <= row["timestamp"]) & (station_stop_times["departure_time"] >= row["timestamp"])]
    if sst.empty:
        return
    return sst["stationShortCode"].unique()[0]


@dataclass
class StationsAndDates:
    start_station: str
    end_station: str
    start_date: str
    end_date: str


@dataclass
class IntervalDfs:
    start_station: str
    end_station: str
    distance: int
    trains: pd.DataFrame
    location_df: pd.DataFrame


class TrainLocations:
    
    def __init__(self, stations_and_dates, **kwargs):
        self.start_station = stations_and_dates.start_station
        self.end_station = stations_and_dates.end_station
        if self.start_station is None or self.end_station is None:
            raise Exception("No start or end station defined.")
        self.dates = pd.date_range(stations_and_dates.start_date, stations_and_dates.end_date, freq="1D")

        # self.db_path = EXTRA_DB_PATH.parent / f"{self.start_station}-{self.end_station}_{stations_and_dates.start_date}_{stations_and_dates.end_date}.db"
        self.db_path = EXTRA_DB_PATH.parent / f"{self.start_station}-{self.end_station}.db"
        
        self.timetables = kwargs.get("timetables")
        self.location_df_raw = kwargs.get("location_df_raw")
        # self.location_dfs = None
        self.train_df = kwargs.get("train_df")
        self.route = kwargs.get("route")
        self.interval_dfs = kwargs.get("interval_dfs")
    
    # jos haluaa kuluttaa aikaa...
    def find_trains_and_timetables(self, wait_time=None):
        SELECTED_KEYS = ["departureDate", "trainNumber", "operatorShortCode", "trainType", "trainCategory", "cancelled", "version", "timetableType"]
        self.timetables = pd.DataFrame()
        self.train_df = pd.DataFrame()
        for date in self.dates:
            timetable_data = request_data("trains", date=str(date.date()))
            if timetable_data is None:
                continue
            for train in timetable_data:
                stations = check_station_stops(train, self.start_station, self.end_station)
                if stations is None:
                    continue
                train_data = {key: item for key, item in train.items() if key in SELECTED_KEYS}
                train_data["stations"] = stations
                timetable = pd.DataFrame(train["timeTableRows"])
                train_data["actualTime_exists"] = True if "actualTime" in timetable.columns else False
                self.train_df = pd.concat([self.train_df, pd.DataFrame([train_data])])
                if not train_data["actualTime_exists"]:
                    # train_data["actualTime_exists"] = False
                    # self.train_df = pd.concat([self.train_df, pd.DataFrame([train_data])])
                    continue
                # train_data["actualTime_exists"] = True
                # self.train_df = pd.concat([self.train_df, pd.DataFrame([train_data])])
                timetable = timetable.loc[:, ["stationShortCode", "type", "trainStopping", "cancelled", "scheduledTime", "actualTime", "trainNumber"]]
                # timetable.drop(["countryCode", "differenceInMinutes", "causes", "trainReady"], axis=1, inplace=True)
                timetable["scheduledTime"] = pd.to_datetime(timetable["scheduledTime"])
                timetable["actualTime"] = pd.to_datetime(timetable["actualTime"])
                timetable["departureDate"] = train["departureDate"]
                self.timetables = pd.concat([self.timetables, timetable])

            if wait_time:
                time.sleep(wait_time)
        self.train_df = self.train_df.set_index(["departureDate", "trainNumber"])

    def find_trains(self, wait_time=0.5):
        SELECTED_KEYS = ["departureDate", "trainNumber", "operatorShortCode", "trainType", "trainCategory", "cancelled", "version", "timetableType"]
        self.train_df = pd.DataFrame()
        for date in self.dates:
            timetable_data = request_data("live-trains", date=str(date.date()), start_station=self.start_station, end_station=self.end_station)
            if timetable_data is None:
                continue
            more_trains = []
            for train in timetable_data:
                stations = check_station_stops(train, self.start_station, self.end_station)
                if stations is None:
                    continue
                train_data = {key: item for key, item in train.items() if key in SELECTED_KEYS}
                train_data["stations"] = stations
                # ei tietoa, onko toteutunutta aikataulua saatavilla
                train_data["actualTime_exists"] = None
                train_data["locations_exist"] = None
                more_trains.append(train_data)
            self.train_df = pd.concat([self.train_df, pd.DataFrame(more_trains)])

            if wait_time:
                time.sleep(wait_time)
        self.train_df = self.train_df.set_index(["departureDate", "trainNumber"])
        return self.train_df

    # reitit lienee hyvä saada helposti
    def get_routes(self):
        return self.train_df["stations"].value_counts()
    

    def find_timetables(self, wait_time=0.5):
        self.timetables = pd.DataFrame()
        for date, train_num in self.train_df.index:
            if int(self.train_df.loc[(date, train_num), "cancelled"]):
                continue
            timetable_data = request_data("trains", date=date, train_num=train_num)
            if timetable_data is None:
                continue

            tt = pd.DataFrame(timetable_data[0]["timeTableRows"])
            self.train_df.loc[(date, train_num), "actualTime_exists"] = True if "actualTime" in tt.columns else False
            if not "actualTime" in tt.columns:
                continue

            tt = tt.loc[:, ["stationShortCode", "type", "trainStopping", "actualTime", "trainNumber"]]
            # tt["scheduledTime"] = pd.to_datetime(tt["scheduledTime"])
            tt["actualTime"] = pd.to_datetime(tt["actualTime"])
            tt["departureDate"] = date
            self.timetables = pd.concat([self.timetables, tt])

            if wait_time:
                time.sleep(wait_time)
        
        self.timetables.reset_index(drop=True, inplace=True)
        return self.timetables

    def find_train_locations(self, do_filtering=True, wait_time=0.1):
        self.location_df_raw = pd.DataFrame()
        for date, train_num in self.train_df.index:
            loc_data = get_raw_train_location_data(train_num, date)
            if loc_data is None:
                continue
            if not do_filtering:
                self.location_df_raw = pd.concat([self.location_df_raw, loc_data.sort_values("timestamp")])
                if wait_time:
                    time.sleep(wait_time)
                continue
            # if loc_data is not None:
            if not self.train_df.loc[(date, train_num), "actualTime_exists"]:
                continue
            timetable = select_data_for_train(date, train_num, self.timetables)
            start_time = timetable[timetable["stationShortCode"] == self.start_station]["actualTime"].min() - pd.Timedelta(minutes=1)
            end_time = timetable[timetable["stationShortCode"] == self.end_station]["actualTime"].max() + pd.Timedelta(minutes=1)
            loc_data = loc_data[(start_time < loc_data["timestamp"]) & (loc_data["timestamp"] < end_time)]

            self.location_df_raw = pd.concat([self.location_df_raw, loc_data.sort_values("timestamp")])
            self.train_df.loc[(date, train_num), "locations_exist"] = True

            if wait_time:
                time.sleep(wait_time)
        
        self.train_df["locations_exist"].fillna(False, inplace=True)
        self.location_df_raw.reset_index(drop=True, inplace=True)
        return self.location_df_raw


    def limit_timetables(self):
        new_timetables = pd.DataFrame()
        for date, train_num in self.train_df.index:
            tt = select_data_for_train(date, train_num, self.timetables)
            start_index = tt[tt["stationShortCode"] == self.start_station].index.min()
            end_index = tt[tt["stationShortCode"] == self.end_station].index.max()
            new_timetables = pd.concat([new_timetables, tt.loc[start_index:end_index, :].copy()])
        new_timetables.reset_index(drop=True, inplace=True)
        self.timetables = new_timetables
        return new_timetables

    def limit_train_locations(self):
        new_locations = pd.DataFrame()
        for date, train_num in self.train_df[self.train_df["actualTime_exists"]].index:
            # if not self.train_df.loc[(date, train_num), "actualTime_exists"]:
            #     continue
            loc_df = select_data_for_train(date, train_num, self.location_df_raw)
            tt = select_data_for_train(date, train_num, self.timetables)
            start_time = tt[tt["stationShortCode"] == self.start_station]["actualTime"].min() - pd.Timedelta(minutes=1)
            end_time = tt[tt["stationShortCode"] == self.end_station]["actualTime"].max() + pd.Timedelta(minutes=1)

            start_index = loc_df[(loc_df["speed"] == 0) & (loc_df["timestamp"] > start_time)].index.min()
            end_index = loc_df[(loc_df["speed"] == 0) & (loc_df["timestamp"] < end_time)].index.max()

            new_locations = pd.concat([new_locations, loc_df.loc[start_index:end_index, :].copy()])
        
        new_locations.reset_index(drop=True, inplace=True)
        self.location_df_raw = new_locations
        return new_locations

    # vanha versio, ehkä päivitys luvassa...
    def process_train_locations(self, route):
        trains = self.train_df[(self.train_df["locations_exist"]) & (self.train_df["stations"] == route)]
        # self.location_dfs = [pd.DataFrame() for _ in range(len(route) - 1)]

        trains_df_for_intervals = trains.drop(["cancelled", "stations", "actualTime_exists", "locations_exist"], axis=1)
        trains_df_for_intervals["dist_from_speed"] = None
        trains_df_for_intervals["duration"] = None
        self.interval_dfs = []
        for i, station in enumerate(route[:-1]):
            station2 = route[i + 1]
            self.interval_dfs.append(IntervalDfs(station, station2, None, trains_df_for_intervals, pd.DataFrame()))

        for date, train_num in trains.index:
            loc_df = select_data_for_train(date, train_num, self.location_df_raw).copy().reset_index(drop=True)

            loc_df["duration"] = (loc_df["timestamp"] - loc_df["timestamp"].min()).apply(lambda t: t.total_seconds())
            loc_df["acceleration"] = get_acceleration(loc_df["speed"], loc_df["duration"])
            loc_df["dist_from_speed"] = from_speed_to_distance(loc_df["speed"], loc_df["duration"]).cumsum()

            loc_df["stops"] = loc_df.apply(lambda r: get_stops(r, id(loc_df["speed"]), "speed"), axis=1)
            station_stop_times = get_stop_times(select_data_for_train(date, train_num, self.timetables))
            loc_df["station"] = loc_df.apply(lambda r: get_station(r, station_stop_times), axis=1)

            for i, station in enumerate(route[:-1]):
                # station1 = route[i]
                station2 = route[i + 1]
                index1 = loc_df[loc_df["station"] == station].index.max()
                index2 = loc_df[loc_df["station"] == station2].index.min()
                df_to_add = loc_df.loc[index1:index2, :].copy()

                # kenties on parempi olla jotain asemien välillä
                df_to_add["station"] = df_to_add["station"].fillna(f"{station1}-{station2}")

                # tarviiko aloittaa nollasta?
                df_to_add["dist_from_speed"] = df_to_add["dist_from_speed"] - df_to_add["dist_from_speed"].min()
                df_to_add["duration"] = df_to_add["duration"] - df_to_add["duration"].min()

                self.interval_dfs[i].location_df = pd.concat([self.interval_dfs[i].location_df, df_to_add])
                # self.location_dfs[i] = pd.concat([self.location_dfs[i], df_to_add])

                self.interval_dfs[i].trains.loc[(date, train_num), "dist_from_speed"] = df_to_add["dist_from_speed"].max()
                self.interval_dfs[i].trains.loc[(date, train_num), "duration"] = df_to_add["duration"].max()

                # self.train_dfs[i].loc[(date, train_num), "dist_from_speed"] = df_to_add["dist_from_speed"].max()
                # self.train_dfs[i].loc[(date, train_num), "duration"] = df_to_add["duration"].max()

            # self.location_df = pd.concat([self.location_df, loc_df])

        # self.location_df.reset_index(drop=True, inplace=True)
        # self.location_dfs = [loc_df.reset_index(drop=True) for loc_df in self.location_dfs]
        # self.train_dfs = [get_distances_from_df(loc_df) for loc_df in self.location_dfs]
        return self.interval_dfs


    def calculate_best_distance_estimate(self, method="median"):
        if method == "median":
            for data in self.interval_dfs:
                data.distance = round(data.trains["dist_from_speed"].median(), -2)
            return
        if method == "mean":
            for data in self.interval_dfs:
                data.distance = round(data.trains["dist_from_speed"].mean(), -2)


    def filter_data_based_on_distance(self, percentage=1):
        if percentage < 0 or percentage > 100:
            raise Exception(f"nonsensical percentage: {percentage}")
        upper_multiplier = 1 + percentage / 100
        lower_multiplier = 1 - percentage / 100
        for data in self.interval_dfs:
            lower_bound = lower_multiplier * data.distance
            upper_bound = upper_multiplier * data.distance
            data.trains["in_analysis"] = data.trains["dist_from_speed"].apply(lambda d: d >= lower_bound and d <= upper_bound)


    def scale_distances(self):
        for data in self.interval_dfs:
            dist_reference = data.trains["dist_from_speed"]
            data.location_df["dist_from_speed"] = data.location_df.apply(lambda r: scale_distance(r, dist_reference, data.distance), axis=1)


    def insert_checkpoints(self, checkpoint_interval=100):
        pass

    def interpolate_checkpoints(self):
        pass
