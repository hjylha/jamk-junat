
import time
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .api_fcns import request_data
from .db_fcns import save_df_to_db, get_df_from_db, EXTRA_DB_PATH
from .data_fcns import check_station_stops, get_raw_train_location_data, select_data_for_train
from .data_fcns import get_acceleration, from_speed_to_distance, get_stops, interpolate_time, get_cluster_df
from .plot_fcns import draw_kmeans_centroids


def train_dict(date, train_num, train_type, train_category):
    return {
        "departureDate": date,
        "trainNumber": train_num,
        "trainType": train_type,
        "trainCategory": train_category
    }

def addition_dict(date, train_num, distance, duration, speed):
    return {
        "departureDate": date,
        "trainNumber": train_num,
        "dist_from_speed": distance,
        "duration": duration,
        "speed": speed
    }

def get_acceleration_w_3_points(speeds, durations):
    diff_quotient = (speeds.diff(2) / durations.diff(2)).shift(-1).fillna(0)
    return diff_quotient * (speeds != 0).astype(int)


def get_stop_times(timetable, extra_time=60):
    tt = timetable[timetable["trainStopping"] == True]

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

# sama nimi kuin data_fcns-funktiolla
def scale_distance(row, ref_df, best_dist_estimate, col_name="dist_from_speed"):
    # max_dist = df[(df["trainNumber"] == row["trainNumber"]) & (df["departureDate"] == row["departureDate"])]["dist_from_speed"].max()
    # max_dist = ref_df.loc[(row["trainNumber"], row["departureDate"]), "dist_from_speed"]
    max_dist = ref_df.loc[(row["departureDate"], row["trainNumber"])]
    return row[col_name] / max_dist * best_dist_estimate 


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
    checkpoints: np.ndarray
    cluster_df: pd.DataFrame
    kmeans: KMeans


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
        self.clustering_data = kwargs.get("clustering_data")


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
        if not self.train_df.empty:
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
        for date, train_num in self.train_df[self.train_df["actualTime_exists"] == True].index:
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


    def find_data(self, wait_times=None, do_filtering=True, do_limiting=True, force_reset=False):
        if wait_times is None:
            wait_times = {
                "trains": 0.5,
                "timetables": 0.5,
                "locations": 0.1
            }
        if isinstance(wait_times, (float, int)):
            wait_times = {"trains": wait_times, "timetables": wait_times, "locations": wait_times}
        if force_reset or self.train_df is None:
            self.find_trains(wait_times["trains"])
            print(f"Trains found: {len(self.train_df)}")
        if force_reset or self.timetables is None:
            self.find_timetables(wait_times["timetables"])
            print(f'Trains with timetables: {len(self.timetables.apply(lambda r: (r["departureDate"], r["trainNumber"]), axis=1).unique())}')
        if force_reset or self.location_df_raw is None:
            self.find_train_locations(do_filtering, wait_times["locations"])
            print(f'Trains with locations: {len(self.location_df_raw.apply(lambda r: (r["departureDate"], r["trainNumber"]), axis=1).unique())}')

        if do_limiting:
            self.limit_timetables()
            if not do_filtering:
                self.limit_train_locations()


    def save_raw_data_to_db(self, db_path=None, if_exists_action="fail"):
        if db_path is not None:
            self.db_path = db_path
        
        # helpot tallennukset
        save_df_to_db(self.timetables, "timetables", db_path=self.db_path, if_exists_action=if_exists_action)
        save_df_to_db(self.location_df_raw, "locations_raw", db_path=self.db_path, if_exists_action=if_exists_action)

        # index ja stations pitää ottaa huomioon
        trains_to_db = self.train_df.reset_index().copy()
        trains_to_db["stations"] = trains_to_db["stations"].apply(lambda t: ",".join(t))
        save_df_to_db(trains_to_db, "trains", db_path=self.db_path, if_exists_action=if_exists_action)


    def load_raw_data_from_db(self, db_path=None):
        if db_path is not None:
            self.db_path = db_path
        self.timetables = get_df_from_db("timetables", db_path=self.db_path)
        self.location_df_raw = get_df_from_db("locations_raw", db_path=self.db_path)
        # junat vaativat lisähuomiota
        trains_loading = get_df_from_db("trains", db_path=self.db_path)
        if trains_loading is not None:
            trains_loading["stations"] = trains_loading["stations"].apply(lambda s: tuple(s.split(",")))
            self.train_df = trains_loading.set_index(["departureDate", "trainNumber"])


    # jaetaan matka (ja data) osiin reitin perusteella
    def process_train_locations(self, route):
        trains = self.train_df[(self.train_df["locations_exist"] == True) & (self.train_df["stations"] == route)]
        # self.location_dfs = [pd.DataFrame() for _ in range(len(route) - 1)]

        trains_df_for_intervals = trains.drop(["cancelled", "stations", "actualTime_exists", "locations_exist"], axis=1)
        trains_df_for_intervals["dist_from_speed"] = None
        trains_df_for_intervals["duration"] = None
        trains_df_for_intervals["num_of_stops"] = None
        trains_df_for_intervals["in_analysis"] = True
        self.interval_dfs = []
        for i, station in enumerate(route[:-1]):
            station2 = route[i + 1]
            self.interval_dfs.append(IntervalDfs(station, station2, None, trains_df_for_intervals.copy(), pd.DataFrame(), None, None, None))

        for date, train_num in trains.index:
            loc_df = select_data_for_train(date, train_num, self.location_df_raw).copy().reset_index(drop=True)
            if loc_df.empty:
                print(f"emptyness: {date=}, {train_num=}")
                continue

            loc_df["duration"] = (loc_df["timestamp"] - loc_df["timestamp"].min()).apply(lambda t: t.total_seconds())
            # tarvitaanko kiihtyvyyyttä vielä?
            # loc_df["acceleration"] = get_acceleration(loc_df["speed"], loc_df["duration"])
            loc_df["dist_from_speed"] = from_speed_to_distance(loc_df["speed"], loc_df["duration"]).cumsum()

            loc_df["stops"] = loc_df.apply(lambda r: get_stops(r, id(loc_df["speed"]), "speed"), axis=1)
            station_stop_times = get_stop_times(select_data_for_train(date, train_num, self.timetables))
            loc_df["station"] = loc_df.apply(lambda r: get_station(r, station_stop_times), axis=1)

            # poistetaanko?
            loc_df.drop(["timestamp", "latitude", "longitude"], axis=1, inplace=True)

            for i, station in enumerate(route[:-1]):
                # station1 = route[i]
                station2 = route[i + 1]
                index1 = loc_df[loc_df["station"] == station].index.max()
                index2 = loc_df[loc_df["station"] == station2].index.min()
                df_to_add = loc_df.loc[index1:index2, :].copy()

                if df_to_add.empty:
                    # raise Exception(f"emptyness: {date=}, {train_num=}, stations={loc_df['station'].dropna().unique()}")
                    self.interval_dfs[i].trains.loc[(date, train_num), "in_analysis"] = False
                    continue

                # kenties on parempi olla jotain asemien välillä
                df_to_add["station"] = df_to_add["station"].fillna(f"{station}-{station2}")

                # tarviiko aloittaa nollasta?
                df_to_add["dist_from_speed"] = df_to_add["dist_from_speed"] - df_to_add["dist_from_speed"].min()
                df_to_add["duration"] = df_to_add["duration"] - df_to_add["duration"].min()

                self.interval_dfs[i].trains.loc[(date, train_num), "dist_from_speed"] = df_to_add["dist_from_speed"].max()
                self.interval_dfs[i].trains.loc[(date, train_num), "duration"] = df_to_add["duration"].max()
                self.interval_dfs[i].trains.loc[(date, train_num), "num_of_stops"] = df_to_add["stops"].iloc[-1] - df_to_add["stops"].iloc[0]

                df_to_add.drop(["stops", "station"], axis=1, inplace=True)
                self.interval_dfs[i].location_df = pd.concat([self.interval_dfs[i].location_df, df_to_add])
                # self.location_dfs[i] = pd.concat([self.location_dfs[i], df_to_add])

                # self.train_dfs[i].loc[(date, train_num), "dist_from_speed"] = df_to_add["dist_from_speed"].max()
                # self.train_dfs[i].loc[(date, train_num), "duration"] = df_to_add["duration"].max()

            # self.location_df = pd.concat([self.location_df, loc_df])

        # self.location_df.reset_index(drop=True, inplace=True)
        # self.location_dfs = [loc_df.reset_index(drop=True) for loc_df in self.location_dfs]
        # self.train_dfs = [get_distances_from_df(loc_df) for loc_df in self.location_dfs]
        return self.interval_dfs


    def calculate_best_distance_estimate(self, method="median", num_of_decimals=-2):
        if method == "median":
            for data in self.interval_dfs:
                data.distance = round(data.trains["dist_from_speed"].median(), num_of_decimals)
            return
        if method == "mean":
            for data in self.interval_dfs:
                data.distance = round(data.trains["dist_from_speed"].mean(), num_of_decimals)
        raise Exception(f"Method not supported: {method}")


    def filter_data_based_on_distance(self, percentage=1, min_error=500):
        if percentage < 0 or percentage > 100:
            raise Exception(f"nonsensical percentage: {percentage}")
        upper_multiplier = 1 + percentage / 100
        lower_multiplier = 1 - percentage / 100

        def filtering_fcn(row, lower_bound, upper_bound):
            if not row["in_analysis"] or not row["dist_from_speed"]:
                return False
            return row["dist_from_speed"] >= lower_bound and row["dist_from_speed"] <= upper_bound

        for data in self.interval_dfs:
            median_dist = data.trains["dist_from_speed"].median()
            lower_bound = min(lower_multiplier * median_dist, median_dist - min_error)
            upper_bound = max(upper_multiplier * median_dist, median_dist + min_error)
            # data.trains["in_analysis"] = data.trains["dist_from_speed"].apply(lambda d: d >= lower_bound and d <= upper_bound)
            data.trains["in_analysis"] = data.trains.apply(lambda r: filtering_fcn(r, lower_bound, upper_bound), axis=1)
            data.location_df["in_analysis"] = data.location_df.apply(lambda r: data.trains.loc[(r["departureDate"], r["trainNumber"]), "in_analysis"], axis=1)
            data.location_df = data.location_df[data.location_df["in_analysis"]].drop("in_analysis", axis=1).reset_index(drop=True)
        return self.interval_dfs


    def scale_distances(self):
        for data in self.interval_dfs:
            dist_reference = data.trains["dist_from_speed"]
            data.location_df["dist_from_speed"] = data.location_df.apply(lambda r: scale_distance(r, dist_reference, data.distance), axis=1)
            # entä nopeus? km/h -> m/s
            data.location_df["speed"] = data.location_df.apply(lambda r: scale_distance(r, dist_reference, data.distance, "speed"), axis=1) / 3.6


    def insert_checkpoints(self, checkpoint_interval=100):
        if checkpoint_interval <= 0:
            raise Exception(f"Checkpoint interval nonpositive: {checkpoint_interval}")
        # tämä on huono tarkastus, mutta ehkä se ei haittaa
        if checkpoint_interval > min(*[d.distance for d in self.interval_dfs]):
            raise Exception(f"Checkpoint interval too large: {checkpoint_interval}")
        for data in self.interval_dfs:
            data.checkpoints = np.arange(0, data.distance + checkpoint_interval, checkpoint_interval)
            additions_template = pd.DataFrame([addition_dict(None, None, checkpoint, None, None) for checkpoint in data.checkpoints[1:-1]])
            for date, train_num in data.trains[data.trains["in_analysis"] == True].index:
                additions = additions_template.copy()
                additions["departureDate"] = date
                additions["trainNumber"] = train_num
                data.location_df = pd.concat([data.location_df, additions])
            data.location_df.reset_index(drop=True, inplace=True)
            data.location_df.drop_duplicates(["departureDate", "trainNumber", "dist_from_speed"], inplace=True)
            data.location_df = data.location_df.sort_values(["departureDate", "trainNumber", "dist_from_speed"])


    # interpoloinnit yksi kerrallaan testausta varten
    def interpolate_duration_in_checkpoints(self):
        for data in self.interval_dfs:
            all_new_locations = pd.DataFrame()
            for date, train_num in data.trains[data.trains["in_analysis"] == True].index:
                new_locations = select_data_for_train(date, train_num, data.location_df).sort_index().reset_index(drop=True).sort_values("dist_from_speed")
                new_locations["duration"] = new_locations.apply(lambda r: interpolate_time(r, new_locations, "m/s"), axis=1)
                new_locations["dist_from_speed"] = new_locations["dist_from_speed"].round()
                # new_locations.drop_duplicates("dist_from_speed", inplace=True)
                # new_locations.drop_duplicates("duration", inplace=True)

                all_new_locations = pd.concat([all_new_locations, new_locations])
            data.location_df = all_new_locations

    def interpolate_speed_in_checkpoints(self):
        for data in self.interval_dfs:
            all_new_locations = pd.DataFrame()
            for date, train_num in data.trains[data.trains["in_analysis"] == True].index:
                new_locations = select_data_for_train(date, train_num, data.location_df).sort_values("dist_from_speed")
                # interpoloidaan ajan suhteen
                new_locations = new_locations.set_index("duration")
                # nopeus oikeassa muodossa
                new_locations["speed"] = new_locations["speed"].astype(float)
                new_locations = new_locations.interpolate(method="index")
                # palautetaan oletusindeksi
                new_locations.drop_duplicates("dist_from_speed", inplace=True)
                new_locations.reset_index(inplace=True)

                all_new_locations = pd.concat([all_new_locations, new_locations])
            data.location_df = all_new_locations


    # interpoloidaan molemmat yhtä aikaa
    def interpolate_values_in_checkpoints(self):
        for data in self.interval_dfs:
            all_new_locations = pd.DataFrame()
            for date, train_num in data.trains[data.trains["in_analysis"] == True].index:
                new_locations = select_data_for_train(date, train_num, data.location_df).sort_index().reset_index(drop=True).sort_values("dist_from_speed")
                new_locations["duration"] = new_locations.apply(lambda r: interpolate_time(r, new_locations, "m/s"), axis=1)
                new_locations["dist_from_speed"] = new_locations["dist_from_speed"].round()
                # new_locations.drop_duplicates("dist_from_speed", inplace=True)
                # new_locations.drop_duplicates("duration", inplace=True)

                # new_locations = new_locations.sort_values("duration")
                new_locations = new_locations.set_index("duration")
                new_locations["speed"] = new_locations["speed"].astype(float)
                new_locations = new_locations.interpolate(method="index")

                new_locations.drop_duplicates("dist_from_speed", inplace=True)
                new_locations.reset_index(inplace=True)
                
                all_new_locations = pd.concat([all_new_locations, new_locations])
            data.location_df = all_new_locations

    def restrict_to_checkpoints(self):
        for data in self.interval_dfs:
            data.location_df = data.location_df[data.location_df["dist_from_speed"].isin(data.checkpoints)]
            # data.location_df.drop_duplicates("dist_from_speed", inplace=True)
            data.location_df.reset_index(drop=True, inplace=True)


    # kaikki yhteen
    def focus_on_checkpoints(self, checkpoint_interval=100):
        self.insert_checkpoints(checkpoint_interval)
        self.interpolate_values_in_checkpoints()
        self.restrict_to_checkpoints()


    def save_checkpoint_data_to_db(self, db_path=None, if_exists_action="fail"):
        if db_path is not None:
            self.db_path = db_path
        for i, data in enumerate(self.interval_dfs):
            save_df_to_db(data.trains.reset_index(), f"trains_{i}",  db_path=self.db_path, if_exists_action=if_exists_action)
            save_df_to_db(data.location_df.reset_index(), f"locations_{i}",  db_path=self.db_path, if_exists_action=if_exists_action)
    
    def load_checkpoint_data_from_db(self, db_path=None):
        if db_path is not None:
            self.db_path = db_path
        trains_0 = get_df_from_db("trains_0", db_path=self.db_path).set_index(["departureDate", "trainNumber"])
        date0, train_num0 = trains_0.index[0]
        route = self.train_df.loc[(date0, train_num0), "stations"]

        # locations_0 = get_df_from_db("locations_0", db_path=self.db_path)
        # self.interval_dfs = [IntervalDfs(route[0], route[1], None, trains_0, locations_0, locations_0["dist_from_speed"].unique(), None, None)]
        self.interval_dfs = []
        for i, station in enumerate(route[:-1]):
            trains = get_df_from_db(f"trains_{i}", db_path=self.db_path).set_index(["departureDate", "trainNumber"])
            locations = get_df_from_db(f"locations_{i}", db_path=self.db_path)
            self.interval_dfs.append(IntervalDfs(station, route[i+1], locations["dist_from_speed"].max(), trains, locations, locations["dist_from_speed"].unique(), None, None))


    # lasketaan kiihtyvyydet
    def calculate_accelerations(self, method="constant_accel"):
        for data in self.interval_dfs:
            if method == "constant_accel":
                data.location_df["acceleration"] = get_acceleration(data.location_df["speed"], data.location_df["duration"], change_unit=False)
            elif method == "3_points":
                data.location_df["acceleration"] = get_acceleration_w_3_points(data.location_df["speed"], data.location_df["duration"])
                data.location_df["speed_derivative"] = data.location_df.apply(lambda r: r["acceleration"] / r["speed"] if r["speed"] != 0 else 0, axis=1)
            else:
                raise Exception(f"unknown method: {method}")
            data.location_df["acceleration+"] = data.location_df["acceleration"].apply(lambda n: max(n, 0))
            data.location_df["acceleration_abs"] = np.abs(data.location_df["acceleration"])

    # col_name voi olla myös lista tms
    def setup_for_clustering(self, col_name="acceleration"):
        for data in self.interval_dfs:
            data.cluster_df = get_cluster_df(data.location_df, col_name)

    def run_kmeans_clustering(self, nums_of_clusters, rng=None):
        if isinstance(nums_of_clusters, int):
            nums_of_clusters = [nums_of_clusters for _ in self.interval_dfs]
        for k, data in zip(nums_of_clusters, self.interval_dfs):
            data.kmeans = KMeans(n_clusters=k, n_init="auto", random_state=rng)
            data.kmeans.fit(data.cluster_df)

    def draw_cluster_centroids(self, clustered_based_on="acceleration"):
        for data in self.interval_dfs:
            clusters = pd.Series(data.kmeans.labels_, index=data.cluster_df.index, name="cluster_id")
            if clustered_based_on == "acceleration":
                title_text = f"Acceleration cluster centroids ({data.start_station}-{data.end_station})"
                draw_kmeans_centroids(data.kmeans, data.checkpoints, clusters, title_text=title_text)
            elif clustered_based_on == "speed":
                title_text_speed = f"Speed cluster centroids ({data.start_station}-{data.end_station})" 
                ylabel_text_speed = "speed ($km/h$)"
                speed_indices = list(range(len(data.checkpoints)))
                max_speed = np.round(3.6 * data.kmeans.cluster_centers_.max()) + 1
                draw_kmeans_centroids(data.kmeans, data.checkpoints, clusters, limits=(0, max_speed), title_text=title_text_speed, ylabel_text=ylabel_text_speed, unit_multiplier=3.6, checkpoint_indices=speed_indices)
            elif clustered_based_on == "speed_and_acceleration":
                # nopeus
                title_text_speed = f"Speed cluster centroids ({data.start_station}-{data.end_station})" 
                ylabel_text_speed = "speed ($km/h$)"
                speed_indices = list(range(len(data.checkpoints)))
                max_speed = np.round(3.6 * data.kmeans.cluster_centers_.max()) + 1
                draw_kmeans_centroids(data.kmeans, data.checkpoints, clusters, limits=(0, max_speed), title_text=title_text_speed, ylabel_text=ylabel_text_speed, unit_multiplier=3.6, checkpoint_indices=speed_indices)
                # kiihtyvyys
                # TODO: EI OLE OIKEIN!
                title_text_acceleration = f"Acceleration cluster centroids ({data.start_station}-{data.end_station})"
                accel_indices = list(range(len(data.checkpoints), 2 * len(data.checkpoints)))
                draw_kmeans_centroids(data.kmeans, data.checkpoints, clusters, title_text=title_text_acceleration, unit_multiplier=1, checkpoint_indices=accel_indices)
            else:
                raise Exception(f"Not valid basis for clustering: {clustered_based_on}")


    def do_clustering(self, nums_of_clusters, rng=None, clustered_based_on="acceleration", draw_graphs=True):
        if clustered_based_on == "acceleration":
            self.calculate_accelerations()
            self.setup_for_clustering()
        elif clustered_based_on == "speed_and_acceleration":
            self.calculate_accelerations(method="3_points")
            self.setup_for_clustering(col_name=["speed", "speed_derivative"])
        elif clustered_based_on == "speed":
            self.setup_for_clustering(col_name="speed")
        self.run_kmeans_clustering(nums_of_clusters, rng)
        if draw_graphs:
            self.draw_cluster_centroids(clustered_based_on)


    def compare_clusters(self):
        clusters = pd.DataFrame()
        for data in self.interval_dfs:
            clusters[f"{data.start_station}-{data.end_station}"] = pd.Series(data.kmeans.labels_, index=data.cluster_df.index)
        clusters.dropna(inplace=True)

        for comb in combinations(clusters.columns, 2):
            x = np.array(clusters[comb[0]]).reshape(-1, 1)
            y = clusters[comb[1]]
            rfc = RandomForestClassifier()
            rfc.fit(x, y)
            y_pred = rfc.predict(x)
            print(f"{comb[0]} vs. {comb[1]}")
            print(f"Accuracy: {rfc.score(x, y)}\n")
            print(classification_report(y, y_pred))
            # print(classification_report(clusters[comb[0]], clusters[comb[1]]))
            print()

