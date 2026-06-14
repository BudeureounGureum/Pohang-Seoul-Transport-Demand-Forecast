from pathlib import Path
import pandas as pd
import numpy as np
import holidays


def aggregate_and_date_features():
    # Load datasets
    weather_path = Path("data/processed/weather.parquet")
    p2s_path = Path("data/processed/pohang_to_seoul.parquet")
    s2p_path = Path("data/processed/seoul_to_pohang.parquet")
    
    df_weather = pd.read_parquet(weather_path)

    # Flatten the weather dataset to have one line per date (adding columns for every locations)
    df_weather = df_weather.pivot(
        index='date',
        columns='location_name'
    )
    df_weather.columns = [
        f"{var}_{city.lower()}"
        for var, city in df_weather.columns
    ]
    df_weather.reset_index(inplace=True)
    
    # Rename columns of transportation datasets to remove origin_city and destination_city
    df_p2s = pd.read_parquet(p2s_path)
    df_p2s = df_p2s.pivot(
        index='date',
        columns=['origin_city', 'destination_city']
    )
    df_p2s.columns = [
        f"{var}_{origin.lower()}_to_{destination.lower()}"
        for var, origin, destination in df_p2s.columns
    ]
    df_p2s.reset_index(inplace=True)

    df_s2p = pd.read_parquet(s2p_path)
    df_s2p = df_s2p.pivot(
        index='date',
        columns=['origin_city', 'destination_city']
    )
    df_s2p.columns = [
        f"{var}_{origin.lower()}_to_{destination.lower()}"
        for var, origin, destination in df_s2p.columns
    ]
    df_s2p.reset_index(inplace=True)

    # Merge the datasets
    df = df_weather.merge(df_p2s, on='date', how='outer')
    df = df.merge(df_s2p, on='date', how='outer')
    
    # Extract day, month, year from date
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Add day_of_week and is_weekend features
    df['day_of_week'] = df['date'].dt.dayofweek+1
    df['is_weekend'] = df['day_of_week'].isin([6, 7]).astype(int)

    # Add cyclical encoded features for month and day
    # cos_month = cos(month * pi / 6), sin_month = sin(month * pi / 6)
    # cos_day = cos(day_of_week * pi / 3.5), sin_day = sin(day_of_week * pi / 3.5)
    df['cos_month'] = np.cos(df['month'] * np.pi / 6)
    df['sin_month'] = np.sin(df['month'] * np.pi / 6)
    
    df['cos_day'] = np.cos(df['day_of_week'] * np.pi / 3.5)
    df['sin_day'] = np.sin(df['day_of_week'] * np.pi / 3.5)
    
    # Holiday/Vacation data loading
    kr_holidays = holidays.KR(years=range(2020, 2027))
    kr_holidays = pd.to_datetime([d for d in kr_holidays.keys() if pd.Timestamp(2020,10,9) <= pd.Timestamp(d) <= pd.Timestamp(2026,5,24)]).sort_values()

    # Add holidays distances
    dates = df['date']

    # Position of each date among the holidays
    idx = kr_holidays.searchsorted(dates)

    # Previous holiday
    prev_holiday = pd.Series(pd.NaT, index=df.index)
    mask = idx > 0
    prev_holiday[mask] = kr_holidays[idx[mask] - 1]

    # Next holiday
    next_holiday = pd.Series(pd.NaT, index=df.index)
    mask = idx < len(kr_holidays)
    next_holiday[mask] = kr_holidays[idx[mask]]

    # Days since previous holiday
    df['days_since_last_holiday'] = (
        dates - prev_holiday
    ).dt.days

    # Days until next holiday
    df['days_until_next_holiday'] = (
        next_holiday - dates
    ).dt.days

    def count_non_working_days(start_date):
        count = 0
        current = start_date + pd.Timedelta(days=1)

        while (
            current.weekday() >= 5      # Saturday/Sunday
            or current in kr_holidays
        ):
            count += 1
            current += pd.Timedelta(days=1)

        return count

    df['upcoming_non_working_days'] = (
        df['date']
        .apply(count_non_working_days)
    )
    
    # Drop the date column
    df = df.drop(columns=['date'])
    
    # Save aggregated dataset
    output_path = Path("data/processed/aggregated_dataset.parquet")
    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )


if __name__ == "__main__":
    aggregate_and_date_features()
