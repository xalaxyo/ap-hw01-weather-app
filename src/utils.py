import os
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import aiohttp
from dotenv import load_dotenv

from src.paths import BASE_DIR


load_dotenv(BASE_DIR / ".env")


API_KEY = os.environ["API_KEY"]
ENDPOINT_URL = os.environ["ENDPOINT_URL"]


month_to_season = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
}


def add_features_sequential(data: pd.DataFrame) -> pd.DataFrame:

    # Calculate the rolling average with a window of 30 days
    data["rolling_mean"] = data.groupby("city")["temperature"].transform(
        lambda group: group.rolling(window=30, min_periods=1).mean()
    )

    # Calculate mean and std for each season and city
    data["seasonal_mean"] = data.groupby(["city", "season"])["temperature"].transform(
        lambda group: group.mean()
    )
    data["seasonal_std"] = data.groupby(["city", "season"])["temperature"].transform(
        lambda group: group.std()
    )

    # Detect outliers
    data["outlier"] = (
        np.abs(data["temperature"] - data["seasonal_mean"]) > 2 * data["seasonal_std"]
    )

    return data


def add_features_parallelized(
    data: pd.DataFrame,
    num_processes: int = 1,
) -> pd.DataFrame:

    cities = data["city"].unique()

    # Split dataframe by city
    inputs = [data[data["city"] == city] for city in cities]

    with mp.Pool(processes=num_processes) as pool:
        outputs = pool.map(add_features_by_city, inputs)

    return pd.concat(outputs, axis=0)


def add_features_by_city(data: pd.DataFrame) -> pd.DataFrame:

    # Calculate the rolling average with a window of 30 days
    data["rolling_mean"] = data["temperature"].rolling(window=30, min_periods=1).mean()

    # Calculate mean and std for each season and city
    data["seasonal_mean"] = data.groupby("season")["temperature"].transform(
        lambda group: group.mean()
    )
    data["seasonal_std"] = data.groupby("season")["temperature"].transform(
        lambda group: group.std()
    )

    # Detect outliers
    data["outlier"] = (
        np.abs(data["temperature"] - data["seasonal_mean"]) > 2 * data["seasonal_std"]
    )

    return data


def get_current_temperature_sync(city: str) -> float:

    url = f"{ENDPOINT_URL}?appid={API_KEY}&q={city}&units=metric"

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    return data["main"]["temp"]


async def get_current_temperature_async(city: str) -> float:

    url = f"{ENDPOINT_URL}?appid={API_KEY}&q={city}&units=metric"

    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.get(url) as response:
            data = await response.json()

    return data["main"]["temp"]


def check_current_temperature(
    data: pd.DataFrame,
    city: str,
    current_temperature: float,
) -> bool:

    current_month = datetime.now().month
    current_season = month_to_season[current_month]

    data_city = data[(data["city"] == city) & (data["season"] == current_season)]
    data_city = data_city.iloc[0]

    mean_temperature = data_city["seasonal_mean"]
    std_temperature = data_city["seasonal_std"]

    return abs(current_temperature - mean_temperature) <= 2 * std_temperature
