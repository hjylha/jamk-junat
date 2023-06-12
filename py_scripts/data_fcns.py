import time

# import requests
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


# koordinaatit asteina, palautusarvo m
def coords_to_distance_w_pyttis(latitude1, longitude1, latitude2, longitude2):
    R = 6_371_000
    
    lat_mean_r = (latitude1 + latitude2) / 2 * np.pi / 180
    lat_diff_r = (latitude2 - latitude1) * np.pi / 180
    lon_diff_r = (longitude2 - longitude1) * np.pi / 180
    
    x = lon_diff_r * np.cos(lat_mean_r)
    y = lat_diff_r
    
    return R * np.sqrt(x*x + y*y)


# nopeus km/h, duration s, palautusarvo m
def from_speed_to_distance(speeds, durations):
    distances = np.zeros(len(speeds))
    # time_diff = durations[1:].to_numpy() - durations[:-1].to_numpy()
    time_diff = durations.diff(1).to_numpy()[1:]
    # distances[1:] = speeds[:-1] * time_diff / 3.6
    # otetaan nopeuden muutos huomioon
    distances[1:] = (speeds[:-1] * time_diff + 0.5 * speeds.diff(1).to_numpy()[1:] * time_diff) / 3.6
    return distances


# aiemmin otettiin jo kiihtyvyys tavallaan mukaan
# def from_speed_and_accel_to_distance(speeds, accels, durations):
#     distances = np.zeros(len(speeds))
#     time_diff = durations.diff(1).to_numpy()[1:]
#     distances[1:] = (speeds[:-1] * time_diff + 0.5 * speeds.diff(1).to_numpy()[1:] * time_diff) / 3.6
#     # distances[1:] = speeds[:-1] * time_diff / 3.6 + 0.5 * accels.to_numpy()[1:] * time_diff * time_diff
#     return distances


# arvioidaan nopeutta koordinaattien muutosten perusteella
# muutokset m, duration s, palautusarvo km/h
def approximate_speed(location_changes, durations):
    speed = np.zeros(len(location_changes))
    # time_diff = durations[1:].to_numpy() - durations[:-1].to_numpy()
    time_diff = durations.diff(1).to_numpy()[1:]
    speed[1:] = location_changes[1:] / time_diff * 3.6
    return speed


# nopeus km/h, duration s, palautusarvo m/s/s
def get_acceleration(speeds, durations):
    accel = np.zeros(len(speeds))
    # speed_diff = speeds[1:].to_numpy() - speeds[:-1].to_numpy()
    speed_diff = speeds.diff(1)[1:]
    # time_diff = durations[1:].to_numpy() - durations[:-1].to_numpy()
    time_diff = durations.diff(1).to_numpy()[1:]
    accel[1:] = speed_diff / time_diff / 3.6
    return accel


# piirretään kuvaajia
def draw_distance_graphs(df):
    train_num = df['trainNumber'].unique()[0]
    date = df["departureDate"].unique()[0]
    fig1 = plt.figure(figsize=(14, 6))
    ax1 = fig1.add_subplot(121)
    ax1.plot(df["duration"] / 3600, df["dist_from_coords"] / 1000)
    ax1.plot(df["duration"] / 3600, df["dist_from_speed"] / 1000)
    # df.plot("duration", "dist_from_coords", ax=ax1)
    # df.plot("duration", "dist_from_speed", ax=ax1)
    ax1.legend(["based on coords", "based on speed"])
    ax1.grid()
    ax1.set_title(f"Distance travelled by train {train_num} on {date}")
    ax1.set_ylabel("distance ($km$)")
    ax1.set_xlabel("duration ($h$)")
    # plt.show()

    ax2 = fig1.add_subplot(122)
    ax2.plot(df["duration"] / 3600, (df["dist_from_speed"] - df["dist_from_coords"]) / 1000)
    ax2.set_title(f"Difference of distances based on speed and coordinates (train {train_num}, date {date})")
    # ax2.set_ylabel("difference ($m$)")
    ax2.set_xlabel("duration ($h$)")
    ax2.grid()
    plt.show()


def draw_acceleration_graphs(df, graph_type="scatter"):
    train_num = df['trainNumber'].unique()[0]
    date = df["departureDate"].unique()[0]
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"Acceleration of train {train_num} on {date}")
    ax1 = fig.add_subplot(121)
    if graph_type == "scatter":
        ax1.scatter(df["duration"] / 3500, df["acceleration"], s=2)
    elif graph_type == "plot":
        ax1.plot(df["duration"] / 3500, df["acceleration"])
    ax1.set_xlabel("duration ($h$)")
    ax1.set_ylabel("acceleration ($m/s^2$)")
    ax1.set_ylim(-2, 2)
    ax1.grid()
    # plt.show()

    ax2 = fig.add_subplot(122)
    if graph_type == "scatter":
        ax2.scatter(df["dist_from_speed"] / 1000, df["acceleration"], s=2)
    elif graph_type == "plot":
        ax2.plot(df["dist_from_speed"] / 1000, df["acceleration"])
    # plt.title(f"Acceleration of train {train_num} on {date}")
    ax2.set_xlabel("distance travelled ($km$)")
    # ax2.set_ylabel("acceleration ($m/s^2$)")
    ax2.set_ylim(-2, 2)
    ax2.grid()
    plt.show()


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
    return


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
    train_list = request_data("live-trains", date=date, start_station=start_station, end_station=end_station)
    if train_list is not None:
        timetable = pd.DataFrame()
        for train in train_list:
            new_timetable = pd.DataFrame(train["timeTableRows"])
            new_timetable = new_timetable[new_timetable["cancelled"] != True]
            if not new_timetable.empty:
                timetable = pd.concat([timetable, new_timetable])
        return timetable


# date muodossa "yyyy-mm-dd"
def get_train_timetables(date):
    train_list = request_data("trains", date=date)
    if train_list is not None:
        timetable = pd.DataFrame()
        for train in train_list:
            timetable = pd.concat([timetable, pd.DataFrame(train["timeTableRows"])])
        return timetable


# date muodossa "yyyy-mm-dd"
def get_train_timetable(train_num, date):
    train_list = request_data("trains", train_num=train_num, date=date)
    if train_list is not None:
        return pd.DataFrame(train_list[0]["timeTableRows"])


# date muodossa "yyyy-mm-dd"
def get_train_location_data_from_api(train_num, date):
    locations = request_data("train-locations", train_num=train_num, date=date)
    if locations is not None:
        return pd.DataFrame(locations)


# get EVERYTHING
def get_train_location_data(train_num, date, with_graphs=True, with_all_graphs=False):
    df = get_train_location_data_from_api(train_num, date)
    if df is None:
        print(f"Data not found: {train_num=}, {date=}")
        return
    # aikamuodot kuntoon
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["duration"] = (df["timestamp"] - df["timestamp"].min()).apply(lambda t: t.total_seconds())

    # koordinaatit talteen (ja vanhat pois)
    df["latitude"] = df["location"].apply(lambda d: d["coordinates"][1])
    df["longitude"] = df["location"].apply(lambda d: d["coordinates"][0])
    df.drop("location", inplace=True, axis=1)

    # ei toisteta dataa
    # df.drop_duplicates(inplace=True)
    df.drop_duplicates("timestamp", inplace=True)

    # toivottavasti näitäkään ei tarvita
    if "accuracy" in df.columns:
        df.drop("accuracy", inplace=True, axis=1)
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
    # kokeilu
    # df["dist_from_coords"] = from_speed_and_accel_to_distance(df["speed"], df["acceleration"], df["duration"]).cumsum()
    # sama nopeuden kautta
    df["dist_from_speed"] = from_speed_to_distance(df["speed"], df["duration"]).cumsum()

    # nopeuksiakin voi arvioida koordinaattien avulla
    # df["speed_from_coords"] = approximate_speed(df["change_of_location"], df["duration"])

    # pysähdykset
    df["stops_from_speed"] = df.apply(lambda r: get_stops(r, id(df["speed"]), "speed"), axis=1)
    # df["stops_from_coords"] = df.apply(lambda r: get_stops(r, id(df["speed_from_coords"]), "speed_from_coords"), axis=1)

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
        draw_distance_graphs(df)

    # kiihtyvyyskuvaajia
    if with_graphs:
        print(f"Distance travelled by train {train_num} on {date}")
        print(f"Total distance travelled (based on speed): \t\t{round(df['dist_from_speed'].max() / 1000, 3)} km")
        print(f"Total distance travelled (based on coordinates): \t{round(df['dist_from_coords'].max() / 1000, 3)} km")
        print()
        print(f"Number of stops (based on speed): \t\t{df['stops_from_speed'].max()}")
        # print(f"Number of stops (based on coordinates): \t{df['stops_from_coords'].max()} (probably nonsense)")
        stop_stations = df['station'].dropna().unique()
        print(f"Stops at stations ({len(stop_stations)}): {stop_stations}")
        print()

        draw_acceleration_graphs(df)  
        print()

    return df


# datan valinta ehkä helpommin
def get_locations_for_train(train_num, date, big_df):
    return big_df[(big_df["trainNumber"] == train_num) & (big_df["departureDate"] == date)]


def get_location_data_for_trains(trains_and_dates, start_station=None, end_station=None, sleeptime=None):
    df = pd.DataFrame()
    for num, date in trains_and_dates:
        try:
            new_df = get_train_location_data(num, date, False)
        except KeyError as e:
            print(f"{type(e).__name__}: {e} [{date=}, {num=}]")
            continue
        if new_df is None:
            continue

        if start_station is not None and end_station is not None:    
            alku = new_df[new_df["station"] == start_station]
            if len(alku) > 0:
                index1 = alku.index.max()
            else:
                continue
            loppu = new_df[new_df["station"] == end_station]
            if len(loppu) > 0:
                index2 = loppu.index.min()
            else:
                continue

        new_df = new_df.loc[index1:index2, :]
        new_df["dist_from_speed"] = new_df["dist_from_speed"] - new_df["dist_from_speed"].min()
        new_df["dist_from_coords"] = new_df["dist_from_coords"] - new_df["dist_from_coords"].min()
        df = pd.concat([df, new_df])

        if sleeptime is not None:
            time.sleep(sleeptime)

    return df


def get_distances_from_df(df):
    groups = df.groupby(["trainNumber", "departureDate"])
    dist_from_coords = groups["dist_from_coords"].max() - groups["dist_from_coords"].min()
    dist_from_speed = groups["dist_from_speed"].max() - groups["dist_from_speed"].min()
    result = pd.concat([dist_from_speed, dist_from_coords], axis=1)
    return result.reset_index()


def scale_distance(row, ref_df, best_dist_estimate):
    # max_dist = df[(df["trainNumber"] == row["trainNumber"]) & (df["departureDate"] == row["departureDate"])]["dist_from_speed"].max()
    # max_dist = ref_df.loc[(row["trainNumber"], row["departureDate"]), "dist_from_speed"]
    max_dist = ref_df.loc[(row["trainNumber"], row["departureDate"])]
    return row["dist_from_speed"] / max_dist * best_dist_estimate 


# varmistetaan, että yllä max_dist palauttaa best_dist_estimaten
def peculiar_rounding(num, reference_estimate):
    res = round(num)
    if res == reference_estimate:
        return reference_estimate
    return num


def accel_dict(num, date, dist, accel=None):
    return {"trainNumber": num,
            "departureDate": date,
            "dist_from_speed": dist,
            "acceleration": accel
           }


# keskitytään olennaiseen ja lisätään checkpointit
def get_essential_df(big_df, best_dist_estimate, trains, checkpoint_interval=100):
    df = big_df.loc[:, ["trainNumber", "departureDate", "dist_from_speed", "acceleration"]].copy()
    df.reset_index(drop=True, inplace=True)

    # skaalataan kuljettu matka (ja pyöristetään, jos tarvetta)
    max_d = df.groupby(["trainNumber", "departureDate"])["dist_from_speed"].max()
    dist = df.apply(lambda r: scale_distance(r, max_d, best_dist_estimate), axis=1)
    df["dist_from_speed"] = dist.apply(lambda n: peculiar_rounding(n, best_dist_estimate))

    # lisätään checkpointit
    checkpoints = np.arange(0, best_dist_estimate + 1, checkpoint_interval)
    additions = []
    for train_num, date in trains:
        for d in checkpoints[1:-1]:
            additions.append(accel_dict(train_num, date, d))
    df = pd.concat([df, pd.DataFrame(additions)])

    # täytetään puuttuvat kiihtyvyysarvot
    df = df.sort_values(["departureDate", "trainNumber", "dist_from_speed"])
    df = df.fillna(method="bfill")

    return df, checkpoints


def get_cluster_df(df, col_name="acceleration"):
    result = pd.pivot_table(df, values=col_name, index=["departureDate", "trainNumber"], columns=["dist_from_speed"], aggfunc=np.mean)
    return result.dropna()


def run_kmeans(df_to_cluster, k, rng=None):
    km = KMeans(n_clusters=k, n_init="auto", random_state=rng)
    km.fit(df_to_cluster)
    cluster_ids = km.predict(df_to_cluster)
    return [km, cluster_ids]


def get_clusters(clustering_results):
    clusters = clustering_results.groupby("cluster_id")

    table = pd.concat([clusters["acceleration_abs"].count(), clusters["acceleration_abs"].min(), clusters["acceleration_abs"].max(), clusters["acceleration_abs"].mean()], axis=1)
    table.columns=["count", "min_mean_abs_accel", "max_mean_abs_accel", "mean_mean_abs_accel"]
    return table.sort_values("count", ascending=False)


# uudelleennimetään klusterit
def get_replacements(cluster_counts, lower_bound):
    clusters = cluster_counts[cluster_counts > lower_bound].sort_values(ascending=False)
    return {c: i for i, c in enumerate(clusters.index)}


def replacement_fcn(cluster_id, cluster_counts, lower_bound):
    replacements = get_replacements(cluster_counts, lower_bound)
    if replacements.get(cluster_id) is None:
        return -1
    return replacements[cluster_id]


def test_clusters_with_knn(df, df_clusters, k_neighbors=5, test_size=0.2, rng=None):
    x_train, x_test, y_train, y_test = train_test_split(df, df_clusters, test_size=0.2, random_state=rng)
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    print(classification_report(y_test, y_pred))

    return knn


def test_clusters_with_rfc(df, df_clusters, test_size=0.2, rng=None):
    x_train, x_test, y_train, y_test = train_test_split(df, df_clusters, test_size=0.2, random_state=rng)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    y_pred = rfc.predict(x_test)

    print(f"Accuracy: {rfc.score(x_test, y_test)}\n")
    print(classification_report(y_test, y_pred))

    return rfc


# clusters muotoa km.predict()
def draw_kmeans_centroids(kmeans, checkpoints, clusters, max_plots=5):
    fig, ax = plt.subplots(figsize=(14, 5))
    c = clusters.value_counts().reset_index().rename({"index": "cluster_id", "cluster_id": "count"}, axis=1)
    # decrease = 0
    for i in range(max_plots):
        label_text = f"{c.loc[i, 'cluster_id']}: {c.loc[i, 'count']}"
        ax.plot(checkpoints / 1000, kmeans.cluster_centers_[c.loc[i, "cluster_id"], :], alpha=0.5, label=label_text)
        # ax.plot(checkpoints / 1000, kmeans.cluster_centers_[c.loc[i, "cluster_id"], :], alpha=0.8 - 0.4 * np.sqrt(decrease))
        # decrease += 1
    
    ax.set_title("Acceleration cluster centroids")
    ax.set_ylabel("acceleration ($m/s^2$)")
    ax.set_xlabel("distance ($km$)")
    ax.set_ylim(-0.5, 0.5)
    ax.legend(c["count"].head(max_plots))
    ax.grid()
    plt.show()
