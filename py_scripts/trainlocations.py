
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .api_fcns import request_data
from .db_fcns import save_df_to_db, get_df_from_db, EXTRA_DB_PATH
from .data_fcns import check_station_stops, get_raw_train_location_data, select_data_for_train


def train_dict(date, train_num, train_type, train_category):
    return {
        "departureDate": date,
        "trainNumber": train_num,
        "trainType": train_type,
        "trainCategory": train_category
    }



@dataclass
class StationsAndDates:
    start_station: str
    end_station: str
    start_date: str
    end_date: str


class TrainLocations:
    
    def __init__(self, stations_and_dates, **kwargs):
        self.start_station = stations_and_dates.start_station
        self.end_station = stations_and_dates.end_station
        self.dates = pd.date_range(stations_and_dates.start_date, stations_and_dates.end_date, freq="1D")

        self.db_path = EXTRA_DB_PATH.parent / f"{self.start_station}-{self.end_station}_{stations_and_dates.start_date}_{stations_and_dates.end_date}.db"
        
        self.timetables = None
        self.location_df_raw = None
        self.location_df = None
        self.train_df = None
        if "timetables" in kwargs:
            self.timetables = kwargs["timetables"]
        if "location_df_raw" in kwargs:
            self.location_df_raw = kwargs["location_df_raw"]
        if "location_df" in kwargs:
            self.location_df = kwargs["location_df"]
        if "train_df" in kwargs:
            self.train_df = kwargs["train_df"]
    
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
                more_trains.append(train_data)
            self.train_df = pd.concat([self.train_df, pd.DataFrame(more_trains)])

            if wait_time:
                time.sleep(wait_time)
        self.train_df = self.train_df.set_index(["departureDate", "trainNumber"])
        return self.train_df

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

            if wait_time:
                time.sleep(wait_time)
        self.location_df_raw.reset_index(drop=True, inplace=True)
        return self.location_df_raw

    def process_train_locations(self):
        self.location_df = pd.DataFrame()
        # processing...
        return self.location_df
