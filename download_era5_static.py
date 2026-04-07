import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from dateutil.relativedelta import relativedelta
import cdsapi
import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed


# Select the variable you want to download

var_sfc = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
]
var_upper = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "geopotential",
    "vertical_velocity", # w
]

var_static = [
    "geopotential", # Geopotential # z
    "land_sea_mask", # Land-sea mask # lsm # 0 (Sea) ~ 1 (Land) Float In-Land Water : 0.5
    "soil_type", #  Soil type # slt 0 ~ 8 INT
]


### download with n retry time
def retrieve_with_retry(
    client: cdsapi.Client,
    dataset: str,
    request: dict,
    file_path: Path,
    max_retries: int = 10,
) -> float | None:
    """Download a single request and report the elapsed time."""
    retries = 0
    while retries < max_retries:
        start_time = time.perf_counter()
        try:
            client.retrieve(dataset, request, str(file_path))
            elapsed = time.perf_counter() - start_time
            print(f"Data saved to {file_path} in {elapsed:.1f}s.")
            return elapsed
        except ValueError as e:
            elapsed = time.perf_counter() - start_time
            retries += 1
            if retries >= max_retries:
                print(f"Failed to retrieve data after {max_retries} attempts. Last error: {e}")
                break
            print(f"Attempt {retries} failed after {elapsed:.1f}s: {e}.")
            print(f"Retrying in {10 * retries} seconds...")
            time.sleep(10 * retries)
    else:
        print(f"Failed to retrieve data after {max_retries} attempts.")

    return None


def static(
    start: str | datetime,
    end: str | datetime,
    save_dir: str,
    region: str,
    levels: str,
) -> None:
    grid = [0.25, 0.25]
    if region == "local":
        area = [40, 100, 5, 145]
    elif region == "global":
        area = [90, 0, -90, 360]
    elif region == "hrrr":
        area = [48.0, 237.25, 21.0, 299.25]
    elif region == "regrid_hrrr":
        area = [51, 235, 23, 290]
    elif region == "era5_us":
        area = [51, 236, 24, 289]
    else:
        raise ValueError(f"Unsupported region '{region}'.")

    if levels == "13_levels":
        pressure_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    elif levels == "8_levels":
        pressure_level = [1000, 925, 850, 700, 500, 300, 150, 50]
    else:
        raise ValueError(f"Unsupported level set '{levels}'.")

    print("***** Downloading [static.nc]. *****")
    output_dir = Path(save_dir) / "static"
    if output_dir.exists():
        print(f"Warning: Directory {output_dir} already exists.", file=sys.stderr)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    retrieve_with_retry(
        client,
        "reanalysis-era5-single-levels",
        {
            "variable": var_static,
            "product_type": "reanalysis",
            "year": 2013,
            "month": 1,
            "day": 1,
            "time": 0,
            "area": area,
            "grid": grid,
            "format": "netcdf",
        },
        output_dir / "static.nc",
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date in the format of YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (inclusive) in the format of YYYY-MM-DD.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/mydata/weather_project",
        help="Directory to save the downloaded data.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="hrrr",
        choices=["global", "local", "hrrr", "regrid_hrrr", "era5_us"],
        help="Specify the region to download data.",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="13_levels",
        choices=["13_levels", "8_levels"],
        help="Specify the pressure levels to download data.",
    )
    parser.add_argument(
        "--no_batch_hours",
        action="store_true",
        help="Disable batching hours into a single request (falls back to one file per hour).",
    )
    parser.add_argument(
        "--chunk_days",
        type=int,
        default=3,
        help="Number of days per batched request when batching is enabled (default: 7).",
    )
    args = parser.parse_args()


    #Download static data
    static(start=args.start, end=args.end, save_dir=args.save_dir, region=args.region, levels=args.levels)
