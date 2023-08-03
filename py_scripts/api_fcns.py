
import time

import requests


def form_url(base_url, rajapinta, *, date="latest", train_num=None, start_station=None, end_station=None):
    if rajapinta == "trains" and train_num:
        return f"{base_url}{rajapinta}/{date}/{train_num}"
    if rajapinta == "trains":
        return f"{base_url}{rajapinta}/{date}"
    if rajapinta == "train-locations" and train_num:
        return f"{base_url}{rajapinta}/{date}/{train_num}"
    if rajapinta == "live-trains":
        return f"{base_url}{rajapinta}/station/{start_station}/{end_station}?departure_date={date}"
    raise Exception(f"Bad url or something ({base_url}/{rajapinta})")


def request_data(rajapinta, **kwargs):
    base_url = "https://rata.digitraffic.fi/api/v1/"
    url = form_url(base_url, rajapinta, **kwargs)

    req = requests.get(url)
    data = req.json()
    if req.status_code == 200 and isinstance(data, list) and data:
        return data
    if req.status_code == 200:
        return
    if req.status_code == 400:
        print(f"Error: status code {req.status_code} Bad Request, {url=}")
        return
    if req.status_code == 500:
        print(f"Error: status code {req.status_code} Internal Server Error, {url=}")
        time.sleep(1)
        return
    raise Exception(f"Error: status code {req.status_code}")
