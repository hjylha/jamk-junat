
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# piirretään kuvaajia
def draw_distance_graphs(df):
    train_num = df['trainNumber'][0]
    date = df["departureDate"][0]
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


def get_y_label(y_name):
    if y_name == "acceleration":
        unit = "$(m/s^2$)"
    elif y_name == "speed":
        unit = "($km/h$)"
    else:
        unit = ""
    ylabel = f"{y_name} {unit}"


def draw_graphs(durations, distances, y, y_name, train_num, date, y_limits=(-1, 1), graph_type="plot"):
    title = f"{y_name.capitalize()} of train {train_num} on {date}"
    ylabel = get_y_label(y_name)
    
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title)
    ax1 = fig.add_subplot(121)
    if graph_type == "scatter":
        ax1.scatter(durations / 3600, y, s=2)
    elif graph_type == "plot":
        ax1.plot(durations / 3600, y)
    ax1.set_xlabel("duration ($h$)")
    ax1.set_ylabel(ylabel)
    ax1.set_ylim(*y_limits)
    ax1.grid()
    # plt.show()

    ax2 = fig.add_subplot(122)
    if graph_type == "scatter":
        ax2.scatter(distances / 1000, y, s=2)
    elif graph_type == "plot":
        ax2.plot(distances / 1000, y)
    # plt.title(f"Acceleration of train {train_num} on {date}")
    ax2.set_xlabel("distance travelled ($km$)")
    # ax2.set_ylabel("acceleration ($m/s^2$)")
    ax2.set_ylim(*y_limits)
    ax2.grid()
    plt.show()


def draw_speed_graphs(df, graph_type="plot"):
    train_num = df['trainNumber'][0]
    date = df["departureDate"][0]
    draw_graphs(df["duration"], df["dist_from_speed"], df["speed"], "speed", train_num, date, (0, df["speed"].max() + 5), graph_type)


def draw_acceleration_graphs(df, graph_type="scatter", limit=2):
    train_num = df['trainNumber'][0]
    date = df["departureDate"][0]
    draw_graphs(df["duration"], df["dist_from_speed"], df["acceleration"], "acceleration", train_num, date, (-limit, limit), graph_type)
    # fig = plt.figure(figsize=(14, 6))
    # fig.suptitle(f"Acceleration of train {train_num} on {date}")
    # ax1 = fig.add_subplot(121)
    # if graph_type == "scatter":
    #     ax1.scatter(df["duration"] / 3500, df[col_name], s=2)
    # elif graph_type == "plot":
    #     ax1.plot(df["duration"] / 3500, df[col_name])
    # ax1.set_xlabel("duration ($h$)")
    # ax1.set_ylabel("acceleration ($m/s^2$)")
    # ax1.set_ylim(-limit, limit)
    # ax1.grid()
    # # plt.show()

    # ax2 = fig.add_subplot(122)
    # if graph_type == "scatter":
    #     ax2.scatter(df["dist_from_speed"] / 1000, df[col_name], s=2)
    # elif graph_type == "plot":
    #     ax2.plot(df["dist_from_speed"] / 1000, df[col_name])
    # # plt.title(f"Acceleration of train {train_num} on {date}")
    # ax2.set_xlabel("distance travelled ($km$)")
    # # ax2.set_ylabel("acceleration ($m/s^2$)")
    # ax2.set_ylim(-limit, limit)
    # ax2.grid()
    # plt.show()


def draw_graph(distances, y, y_name, train_num, date, y_limits=None, graph_type="plot"):
    title = f"{y_name.capitalize()} of train {train_num} on {date}"
    ylabel = get_y_label(y_name)

    fig, ax = plt.subplots(figsize=(14, 5))

    if graph_type == "plot":
        ax.plot(distances / 1000, y, alpha=0.6)
    if graph_type == "scatter":
        ax.scatter(distances / 1000, y, s=3)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("distance ($km$)")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid()
    plt.show()


def draw_speed_graph(df):
    train_num = df['trainNumber'][0]
    date = df["departureDate"][0]
    draw_graph(df["dist_from_speed"], df["speed"], "speed", train_num, date, (0, df["speed"].max() + 5))


def draw_acceleration_graph(df, graph_type="plot", limit=None):
    train_num = df['trainNumber'][0]
    date = df["departureDate"][0]
    limits = (-limit, limit) if limit is not None else None
    draw_graph(df["dist_from_speed"], df["acceleration"], "acceleration", train_num, date, limits, graph_type)


# clusters muotoa km.predict()
def draw_kmeans_centroids(kmeans, checkpoints, clusters, max_plots=5, limit=0.5):
    # jos on liikaa yritystä, pitäisikö siitä ilmoittaa?
    max_plots = min(kmeans.cluster_centers_.shape[0], max_plots)

    fig, ax = plt.subplots(figsize=(14, 5))
    c = clusters.value_counts().reset_index().rename({"cluster_id": "count", "index": "cluster_id"}, axis=1)
    # decrease = 0
    for i in range(max_plots):
        label_text = f"{c.loc[i, 'cluster_id']}: {c.loc[i, 'count']}"
        ax.plot(checkpoints / 1000, kmeans.cluster_centers_[c.loc[i, "cluster_id"], :], alpha=0.5, label=label_text)
        # ax.plot(checkpoints / 1000, kmeans.cluster_centers_[c.loc[i, "cluster_id"], :], alpha=0.8 - 0.4 * np.sqrt(decrease))
        # decrease += 1
    
    ax.set_title("Acceleration cluster centroids")
    ax.set_ylabel("acceleration ($m/s^2$)")
    ax.set_xlabel("distance ($km$)")
    ax.set_ylim(-limit, limit)
    ax.legend()
    ax.grid()
    plt.show()