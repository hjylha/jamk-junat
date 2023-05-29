import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# koordinaatit asteina, vastaus metreinä
def coords_to_distance_w_pyttis(latitude1, longitude1, latitude2, longitude2):
    R = 6_371_000
    
    lat_mean_r = (latitude1 + latitude2) / 2 * np.pi / 180
    lat_diff_r = (latitude2 - latitude1) * np.pi / 180
    lon_diff_r = (longitude2 - longitude1) * np.pi / 180
    
    x = lon_diff_r * np.cos(lat_mean_r)
    y = lat_diff_r
    
    return R * np.sqrt(x*x + y*y)


# nopeus km/h, duration s
def from_speed_to_distance(speeds, durations):
    distances = np.zeros(len(speeds))
    time_diff = durations[1:].to_numpy() - durations[:-1].to_numpy()
    distances[1:] = speeds[:-1] * time_diff / 3.6
    return distances


# arvioidaan nopeutta koordinaattien muutosten perusteella
def approximate_speed(location_changes, durations):
    speed = np.zeros(len(location_changes))
    time_diff = durations[1:].to_numpy() - durations[:-1].to_numpy()
    speed[1:] = location_changes[1:] / time_diff * 3.6
    return speed


def get_acceleration(speeds, durations):
    accel = np.zeros(len(speeds))
    speed_diff = speeds[1:].to_numpy() - speeds[:-1].to_numpy()
    time_diff = durations[1:].to_numpy() - durations[:-1].to_numpy()
    accel[1:] = speed_diff / time_diff / 3.6
    return accel


# numeroidaan pysähdykset, oletusarvojen ovelaa käyttöä toivottavasti...
def get_stops(row, identifier=None, col_name="speed", previous_stops={"previous": 0, "stop_count": 0, "identifier": None}):
    if identifier is not None and identifier != previous_stops["identifier"]:
        previous_stops["previous"] = 0
        previous_stops["stop_count"] = 0
        previous_stops["identifier"] = identifier
    if row[col_name] == 0 and previous_stops["previous"] == 0:
        previous_stops["stop_count"] += 1
        previous_stops["previous"] = previous_stops["stop_count"]
        return previous_stops["stop_count"]
    if row[col_name] == 0:
        return previous_stops["stop_count"]
    if row[col_name] > 0 and previous_stops["previous"] > 0:
        previous_stops["previous"] = 0
    return 0


def get_station(stop_num, list_of_stations):
    if stop_num > 0:
        return list_of_stations[stop_num - 1]
    return None


# get list of stations from location data (stops) and timetable
def get_list_of_stations(loc_df, timetable_df):
    stations = []
    num_of_stops = loc_df["stops_from_speed"].max()
    for stop_num in range(1, num_of_stops + 1):
        stop_df = loc_df[loc_df["stops_from_speed"] == stop_num]
        # ts_min = stop_df["timestamp"].min() - pd.Timedelta(1, "min")
        # ts_max = stop_df["timestamp"].max() + pd.Timedelta(1, "min")
        ts_min = stop_df["timestamp"].min()
        ts_max = stop_df["timestamp"].max()
        
        try:
            depart = timetable_df[timetable_df["actualTime"] >= ts_min].iloc[0, 0]
        except IndexError:
            depart = timetable_df.iloc[-1, 0]
        try:
            arrive = timetable_df[timetable_df["actualTime"] <= ts_max].iloc[-1, 0]
        except IndexError:
            arrive = timetable_df.iloc[0, 0]
        
        if depart == arrive:
            stations.append(depart)
        else:
            # print(f"{stop_num=} no match: {depart=}, {arrive=}")
            stations.append(None)
    return stations


# jos actualTIme puuttuu, korvataan se scheduledTimella
def fill_times(row):
    if row.isna()["actualTime"]:
        return row["scheduledTime"]
    return pd.to_datetime(row["actualTime"])


# asemat stationShortCodena, date muodossa "yyyy-mm-dd"
def get_train_nums(start_station, end_station, date):
    url_start = "https://rata.digitraffic.fi/api/v1/live-trains/"
    url = f"{url_start}station/{start_station}/{end_station}?departure_date={date}"
    req = requests.get(url)
    if req.status_code == 200:
        return [train["trainNumber"] for train in req.json()]
    raise Exception(f"Error: status code {req.status_code}")


# date muodossa "yyyy-mm-dd"
def get_train_timetable(train_num, date):
    url_start = "https://rata.digitraffic.fi/api/v1/trains/"
    url = f"{url_start}{date}/{train_num}"    
    req = requests.get(url)
    if req.status_code == 200:
        return pd.DataFrame(req.json()[0]["timeTableRows"])
    raise Exception(f"Error: status code {req.status_code}")


# date muodossa "yyyy-mm-dd"
def get_train_location_data_from_api(train_num, date):
    url_start = "https://rata.digitraffic.fi/api/v1/train-locations/"
    url = f"{url_start}{date}/{train_num}"
    req = requests.get(url)
    if req.status_code == 200:
        return pd.DataFrame(req.json())
    raise Exception(f"Error: status code {req.status_code}")


# get EVERYTHING
def get_train_location_data(train_num, date, with_graphs=True, with_all_graphs=False):
    df = get_train_location_data_from_api(train_num, date)
    # aikamuodot kuntoon
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["duration"] = (df["timestamp"] - df["timestamp"].min()).apply(lambda t: t.total_seconds())

    # koordinaatit talteen (ja vanhat pois)
    df["latitude"] = df["location"].apply(lambda d: d["coordinates"][1])
    df["longitude"] = df["location"].apply(lambda d: d["coordinates"][0])
    df.drop("location", inplace=True, axis=1)

    # toivottavasti näitäkään ei tarvita
    # df.drop(["trainNumber", "departureDate"], inplace=True, axis=1)

    # onko sorttaus paikallaan?
    df = df.sort_values("duration").reset_index(drop=True)

    # vanhat koordinaatit
    prev_lat = [df.loc[0, "latitude"]]
    prev_lat = prev_lat + list(df.loc[:len(df.index)-2, "latitude"])
    df["previous_latitude"] = prev_lat

    prev_lon = [df.loc[0, "longitude"]]
    prev_lon = prev_lon + list(df.loc[:len(df.index)-2, "longitude"])
    df["previous_longitude"] = prev_lon

    # kiihtyvyys kehiin
    df["acceleration"] = get_acceleration(df["speed"], df["duration"])
    # df["acceleration+"] = df["acceleration"].apply(lambda a: max(a, 0))

    # Approksimoidaan koordinaateista etäisyydet Pythagoraan lauseen avulla
    df["change_of_location"] = df.apply(lambda r: coords_to_distance_w_pyttis(r["latitude"], r["longitude"], r["previous_latitude"], r["previous_longitude"]), axis=1)

    # tarvitaanko enää vanhoja koordinaatteja?
    df.drop(["previous_latitude", "previous_longitude"], axis=1, inplace=True)

    # kuljettu matka koordinaattien muutosten summana
    df["dist_from_coords"] = df["change_of_location"].cumsum()
    # sama nopeuden kautta
    df["dist_from_speed"] = from_speed_to_distance(df["speed"], df["duration"]).cumsum()

    # nopeuksiakin voi arvioida koordinaattien avulla
    df["speed_from_coords"] = approximate_speed(df["change_of_location"], df["duration"])

    # pysähdykset
    df["stops_from_speed"] = df.apply(lambda r: get_stops(r, id(df["speed"]), "speed"), axis=1)
    df["stops_from_coords"] = df.apply(lambda r: get_stops(r, id(df["speed_from_coords"]), "speed_from_coords"), axis=1)

    # pysähdysasemat
    timetable = get_train_timetable(train_num, date)
    timetable = timetable[timetable["trainStopping"]]
    timetable["scheduledTime"] = pd.to_datetime(timetable["scheduledTime"])
    timetable["actualTime"] = timetable.apply(fill_times, axis=1)
    timetable = timetable.loc[:, ["stationShortCode", "actualTime"]]

    timetable = timetable.sort_values("actualTime")

    stations = get_list_of_stations(df, timetable)

    df["station"] = df.apply(lambda r: get_station(r["stops_from_speed"], stations), axis=1)

    # pari kuvaajaa
    if with_all_graphs:
        fig1 = plt.figure(figsize=(14, 6))
        ax1 = fig1.add_subplot(121)
        df.plot("duration", "dist_from_coords", ax=ax1)
        df.plot("duration", "dist_from_speed", ax=ax1)
        ax1.grid()
        ax1.set_title(f"Distance travelled by train {train_num} on {date}")
        ax1.set_ylabel("distance ($m$)")
        ax1.set_xlabel("duration ($s$)")
        # plt.show()

        ax2 = fig1.add_subplot(122)
        ax2.plot(df["duration"], df["dist_from_speed"] - df["dist_from_coords"])
        ax2.set_title(f"Difference of distances based on speed and coordinates (train {train_num}, date {date})")
        # ax2.set_ylabel("difference ($m$)")
        ax2.set_xlabel("duration ($s$)")
        ax2.grid()
        plt.show()

    # kiihtyvyyskuvaajia
    if with_graphs:
        print(f"Distance travelled by train {train_num} on {date}")
        print(f"Total distance travelled (based on speed): \t\t{round(df['dist_from_speed'].max() / 1000, 3)} km")
        print(f"Total distance travelled (based on coordinates): \t{round(df['dist_from_coords'].max() / 1000, 3)} km")
        print()
        print(f"Number of stops (based on speed): \t\t{df['stops_from_speed'].max()}")
        print(f"Number of stops (based on coordinates): \t{df['stops_from_coords'].max()} (probably nonsense)")
        stop_stations = df['station'].dropna().unique()
        print(f"Stops ({len(stop_stations)}): {stop_stations}")
        print()

        fig = plt.figure(figsize=(14, 6))
        fig.suptitle(f"Acceleration of train {train_num} on {date}")
        ax1 = fig.add_subplot(121)

        df.plot("duration", "acceleration", ax=ax1)
        ax1.set_xlabel("duration ($s$)")
        ax1.set_ylabel("acceleration ($m/s^2$)")
        ax1.grid()
        # plt.show()

        ax2 = fig.add_subplot(122)
        df.plot("dist_from_speed", "acceleration", ax=ax2)
        # df.plot("dist_from_coords", "acceleration", ax=ax)
        # plt.title(f"Acceleration of train {train_num} on {date}")
        ax2.set_xlabel("distance travelled ($m$)")
        # ax2.set_ylabel("acceleration ($m/s^2$)")
        ax2.grid()
        plt.show()

        
        print()

    return df
