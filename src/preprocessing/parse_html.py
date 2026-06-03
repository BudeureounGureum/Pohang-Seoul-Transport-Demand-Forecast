from pathlib import Path
from io import StringIO
import pandas as pd
from pathlib import Path
from io import StringIO
import pandas as pd


def parse_stcis_html(path, origin, destination, origin_korean, destination_korean):

    html = Path(path).read_text(encoding="utf-8")

    df_raw = pd.read_html(StringIO(html))[0]

    # Build column names from the two header rows
    header1 = df_raw.iloc[0]
    header2 = df_raw.iloc[1]

    columns = []

    for h1, h2 in zip(header1, header2):

        h1 = str(h1).strip()
        h2 = str(h2).strip()

        if h1 == h2:
            columns.append(h1)
        else:
            columns.append(f"{h1}_{h2}")

    # Skip header and total rows
    df = df_raw.iloc[3:].copy()

    df.columns = columns

    df.reset_index(drop=True, inplace=True)

    # Rename columns to English
    df = df.rename(
        columns={
            "일자": "date",
            "출발지_시도": "origin_province",
            "출발지_시군구": "origin_city",
            "도착지_시도": "destination_province",
            "도착지_시군구": "destination_city",
            "철도_통행량": "rail_passengers",
            "철도_통행시간": "rail_time",
            "고속_통행량": "express_bus_passengers",
            "고속_통행시간": "express_bus_time",
            "시외버스_통행량": "intercity_bus_passengers",
        }
    )

    # Validate and replace cities
    # Treat '-' as missing value
    df["origin_city"] = df["origin_city"].replace("-", None)
    df["destination_city"] = df["destination_city"].replace("-", None)
    # Drop rows where cities don't match expected Korean value (unless missing)
    df = df[
        (df["origin_city"].str.strip() == origin_korean) |
        (df["origin_city"].isna())
    ].copy()    
    df = df[
        (df["destination_city"].str.strip() == destination_korean) |
        (df["destination_city"].isna())
    ].copy()
    
    df["origin_city"] = origin
    df["destination_city"] = destination

    # Convert date column to datetime and extract weekday/weekend information
    df["date"] = (
        df["date"]
        .astype(str)
        .str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
    )

    df["date"] = pd.to_datetime(df["date"])

    # Convert passenger and time columns to numeric, removing commas and handling missing values
    passenger_cols = [
        "rail_passengers",
        "express_bus_passengers",
        "intercity_bus_passengers",
    ]

    for col in passenger_cols:
        df[col] = (
            df[col]
            .str.replace(",", "", regex=False)
            .astype(int)
        )

    # Drop unnecessary columns
    df = df.drop(
        columns=[
            "origin_province",
            "destination_province",
            "rail_time",
            "express_bus_time",
        ]
    )

    return df

def parse_html_convert_to_parquet(origin, destination, origin_korean, destination_korean):
    dfs = []

    for file in Path(f"data/raw/{origin}_to_{destination}").glob("*.html"):

        dfs.append(
            parse_stcis_html(file, origin, destination, origin_korean, destination_korean)
        )

    final_df = pd.concat(dfs).sort_values("date").reset_index(drop=True)
    final_df.to_parquet(
        f"data/processed/{origin}_to_{destination}.parquet",
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

if __name__ == "__main__":
    parse_html_convert_to_parquet("pohang", "seoul", "포항시", "서울")
    parse_html_convert_to_parquet("seoul", "pohang", "서울", "포항시")