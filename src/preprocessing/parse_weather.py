from pathlib import Path
import pandas as pd
import numpy as np

def parse_weather_csv():
    input_file = Path("data/raw/weather_20201101_20260519.csv")
    output_file = Path("data/processed/weather.parquet")
    
    df = pd.read_csv(input_file, encoding="cp949")
    
    print(f"Original shape: {df.shape}")

    # Drop unnecessary columns
    df = df.drop(columns=['지점', '기사'])
    # Drop time columns since we aggregate daily and they frequently have NaNs
    time_cols = [col for col in df.columns if col.endswith('시각(hhmi)')]
    df = df.drop(columns=time_cols)
    # Keep features common to the historical dataset and the forecast data.
    common_features = [
        '지점명',
        '일시',
        '평균기온(°C)',
        '최저기온(°C)',
        '최고기온(°C)',
        '강수 계속시간(hr)',
        '일강수량(mm)',
        '최대 순간 풍속(m/s)',
        '최대 풍속(m/s)',
        '평균 풍속(m/s)',
        '최다풍향(16방위)',
        '평균 이슬점온도(°C)',
        '최소 상대습도(%)',
        '평균 상대습도(%)',
        '평균 현지기압(hPa)',
        '최고 해면기압(hPa)',
        '최저 해면기압(hPa)',
        '평균 해면기압(hPa)',
        '가조시간(hr)',
        '합계 일조시간(hr)',
        '합계 일사량(MJ/m2)',
        '일 최심신적설(cm)',
        '평균 전운량(1/10)',
        '안개 계속시간(hr)',
    ]
    df = df[common_features]

    korean_to_english = {
        '지점명': 'location_name',
        '일시': 'date',
        '평균기온(°C)': 'avg_temperature(°C)',
        '최저기온(°C)': 'min_temperature(°C)',
        '최고기온(°C)': 'max_temperature(°C)',
        '강수 계속시간(hr)': 'precipitation_duration(hr)',
        '일강수량(mm)': 'daily_precipitation(mm)',
        '최대 순간 풍속(m/s)': 'max_gust_wind_speed(m/s)',
        '최대 풍속(m/s)': 'max_wind_speed(m/s)',
        '평균 풍속(m/s)': 'avg_wind_speed(m/s)',
        '최다풍향(16방위)': 'most_frequent_wind_direction',
        '평균 이슬점온도(°C)': 'avg_dew_point(°C)',
        '최소 상대습도(%)': 'min_relative_humidity(%)',
        '평균 상대습도(%)': 'avg_humidity(%)',
        '평균 현지기압(hPa)': 'avg_local_pressure(hPa)',
        '최고 해면기압(hPa)': 'max_sea_level_pressure(hPa)',
        '최저 해면기압(hPa)': 'min_sea_level_pressure(hPa)',
        '평균 해면기압(hPa)': 'avg_sea_level_pressure(hPa)',
        '가조시간(hr)': 'sunshine_possible_hours',
        '합계 일조시간(hr)': 'total_sunshine_hours',
        '합계 일사량(MJ/m2)': 'total_solar_radiation(MJ/m2)',
        '일 최심신적설(cm)': 'daily_max_new_snow_depth(cm)',
        '평균 전운량(1/10)': 'avg_cloud_cover(1/10)',
        '안개 계속시간(hr)': 'fog_duration(hr)',
    }
    
    df = df.rename(columns=korean_to_english)
    
    # Convert fog_duration_hr to boolean indicating presence of fog to match the forecast data.
    df['fog_duration(hr)'] = df['fog_duration(hr)'].notna()
    df.rename(columns={'fog_duration(hr)': 'fog_occured'}, inplace=True)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Sort by date
    df = df.sort_values(['date']).reset_index(drop=True)
    
    # Convert all numeric columns (exclude string columns)
    for col in df.columns:
        if col not in ['location_name', 'date', 'fog_occured']:
            # Try to convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Translate location names to English
    location_translation = {
        '포항': 'pohang',
        '서울': 'seoul',
    }
    df['location_name'] = df['location_name'].map(location_translation)

    # Print nulls before processing
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            print(f"{col}: {null_count} missing values")

    # Fill event columns where NaN means "did not happen"
    zero_fill_cols = [
        'precipitation_duration(hr)', 'daily_precipitation(mm)', 'daily_max_new_snow_depth(cm)'
    ]

    for col in zero_fill_cols:
        if col in df.columns:
            # Existing 0.0 means "Trace amount". 
            # The smallest non-zero values in these columns are 0.08 to 0.17 depending on the column.
            # We change "trace amount" 0s to 0.01 so that the algorithm can distinguish it from 0.0 (no event).
            mask_zero = df[col] == 0
            df.loc[mask_zero, col] = 0.01
            
            # Fill NaN (nothing happened) with 0.0
            df[col] = df[col].fillna(0)

    # For the wind direction, we use the previous day's value (or the next day's value when at the begining of the list)
    df['most_frequent_wind_direction'] = (df['most_frequent_wind_direction'].ffill().bfill())

    radians = np.deg2rad(df['most_frequent_wind_direction'])

    df['most_frequent_wind_direction_sin'] = np.sin(radians)
    df['most_frequent_wind_direction_cos'] = np.cos(radians)

    df.drop(columns=['most_frequent_wind_direction'], inplace=True)
            
    # For continuous numeric columns (temperature, wind, pressure, evaporation, etc.), 
    # we use 'pchip' (Piecewise Cubic Hermite Interpolating Polynomial) which is more
    # realistic for weather data than linear interpolation, avoiding wild overshoots.
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df.groupby('location_name')[numeric_cols].transform(
        lambda x: x.interpolate(method='pchip', limit_direction='both')
    )
    # Fallback to linear if any remaining NaNs exist due to too few datapoints for pchip
    df[numeric_cols] = df.groupby('location_name')[numeric_cols].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    
    # Save to parquet
    df.to_parquet(
        output_file,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

if __name__ == "__main__":
    parse_weather_csv()
