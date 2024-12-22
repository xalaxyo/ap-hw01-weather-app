import streamlit as st
import numpy as np
import pandas as pd
import requests
from requests.exceptions import HTTPError
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


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


def main():

    st.title("Weather Data Analysis")

    st.header("1. Upload historical temperature data")

    uploaded_file = st.file_uploader("Upload historical data (csv)", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        st.write(data.head())

        st.header("2. Choose city to analyze")

        cities = data["city"].unique().tolist()
        city = st.pills("Cities", cities, selection_mode="single")

        if city is not None:
            st.write(f"Your selected city: {city}.")

            data_city = data[data["city"] == city]
            data_city = _add_features(data_city)

            st.subheader("Descriptive statistics")
            st.write(data_city["temperature"].describe())

            st.subheader("Seasonal statistics")
            seasons = ["winter", "spring", "summer", "autumn"]
            data_seasonal = [
                {
                    "season": season,
                    "seasonal_mean": data_city.loc[
                        data_city["season"] == season, "seasonal_mean"
                    ].mean(),
                    "seasonal_std": data_city.loc[
                        data_city["season"] == season, "seasonal_std"
                    ].mean(),
                }
                for season in seasons
            ]
            data_seasonal = pd.DataFrame(data_seasonal)
            st.write(data_seasonal)

            st.subheader("Historical temperature plot")
            fig, ax = plt.subplots()
            sns.scatterplot(
                data=data_city,
                x="timestamp",
                y="temperature",
                hue="outlier",
                style="outlier",
                s=10,
                ax=ax,
            )
            st.pyplot(fig)

            st.header("3. Get current temperature")

            api_key = st.text_input("Enter OpenWeatherAPI key", type="password")

            if len(api_key) > 0:
                st.write("API key is saved.")

                if st.button("Request temperature"):

                    try:
                        current_temperature = _get_current_temperature(
                            api_key=api_key,
                            city=city,
                        )

                        normal = _check_current_temperature(
                            data_city=data_city,
                            current_temperature=current_temperature,
                        )

                        st.metric(
                            f"Current temperature in {city}",
                            f"{current_temperature}°C",
                        )

                        st.metric(
                            "Is current temperature normal?",
                            "Yes." if normal else "No.",
                        )

                    except HTTPError as e:
                        if e.response.status_code == 401:
                            st.error(
                                {
                                    "cod": 401,
                                    "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info.",
                                }
                            )
                        else:
                            st.error(e)


def _add_features(data_city: pd.DataFrame) -> pd.DataFrame:

    # Calculate mean and std for each season and city
    data_city["seasonal_mean"] = data_city.groupby("season")["temperature"].transform(
        lambda group: group.mean()
    )
    data_city["seasonal_std"] = data_city.groupby("season")["temperature"].transform(
        lambda group: group.std()
    )

    # Detect outliers
    data_city["outlier"] = (
        np.abs(data_city["temperature"] - data_city["seasonal_mean"])
        > 2 * data_city["seasonal_std"]
    )

    return data_city


def _get_current_temperature(api_key: str, city: str) -> float:

    endpoint_url = "https://api.openweathermap.org/data/2.5/weather"
    url = f"{endpoint_url}?appid={api_key}&q={city}&units=metric"

    response = requests.get(url)
    response.raise_for_status()

    return response.json()["main"]["temp"]


def _check_current_temperature(
    data_city: pd.DataFrame,
    current_temperature: float,
) -> bool:

    current_month = datetime.now().month
    current_season = month_to_season[current_month]

    data_season = data_city[data_city["season"] == current_season].iloc[0]

    mean_temperature = data_season["seasonal_mean"]
    std_temperature = data_season["seasonal_std"]

    return abs(current_temperature - mean_temperature) <= 2 * std_temperature


if __name__ == "__main__":
    main()
