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
    "geopotential"
]

var_static = [
    "geopotential",
    "land_sea_mask",
    "soil_type",
]

"""Split a multi-time NetCDF into hourly files and delete the source."""
def _split_to_hourly(
    src_path: Path,
    base_dir: Path,
    suffix: str,
    time_coord_candidates: Iterable[str] = ("time", "valid_time"),
) -> None:

    if not src_path.exists():
        return

    ds = xr.open_dataset(src_path)
    try:
        time_coord_name = next(
            (name for name in time_coord_candidates if name in ds.coords),
            None,
        )
        if time_coord_name is None:
            print(f"No time coordinate found in {src_path.name}, skipping split.")
            return

        time_coord = ds[time_coord_name]
        if time_coord.size == 0:
            print(f"Empty time coordinate in {src_path.name}, skipping split.")
            return

        time_attrs = time_coord.attrs
        time_dtype = time_coord.dtype
        epoch = np.datetime64("1970-01-01T00:00:00")

        for idx in range(time_coord.size):
            time_values = time_coord.values[idx : idx + 1]
            if time_values.size == 0:
                continue
            raw_value = time_values[0]
            if np.issubdtype(time_dtype, np.datetime64):
                timestamp = np.datetime64(raw_value, "ns")
            else:
                # assume integer nanoseconds since epoch as used by ERA5 files
                timestamp = np.datetime64(np.int64(raw_value), "ns")

            seconds_since_epoch = int(((timestamp - epoch) / np.timedelta64(1, "s")).astype(np.int64))
            py_dt = datetime.utcfromtimestamp(seconds_since_epoch)
            year = py_dt.year
            month = py_dt.month
            day = py_dt.day
            hour = py_dt.hour

            day_dir = (
                base_dir / f"{year:04d}" / f"{year:04d}{month:02d}" / f"{year:04d}{month:02d}{day:02d}"
            )
            day_dir.mkdir(parents=True, exist_ok=True)

            prefix = f"{year:04d}{month:02d}{day:02d}"
            hour_tag = f"{hour:02d}"
            out_path = day_dir / f"{prefix}{hour_tag}{suffix}"

            base_sel = ds.isel({time_coord_name: idx}, drop=True)
            expanded = base_sel.expand_dims({time_coord_name: time_values})
            coord_array = xr.DataArray(
                np.array(time_values, dtype=time_dtype),
                dims=(time_coord_name,),
                attrs=time_attrs,
            )
            expanded = expanded.assign_coords({time_coord_name: coord_array})
            expanded.to_netcdf(out_path)
            print(f"  Saved {out_path.relative_to(base_dir)}")

    finally:
        ds.close()

    try:
        src_path.unlink()
        print(f"Removed batch file {src_path.name}.")
    except OSError as exc:
        print(f"Warning: could not delete {src_path}: {exc}", file=sys.stderr)


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


def main(
    start: str | datetime,
    end: str | datetime,
    save_dir: str,
    region: str,
    levels: str,
    batch_hours: bool = True,
    chunk_days: int = 7,
) -> None:
    if isinstance(start, str):
        start = datetime.strptime(start, r"%Y-%m-%d")
    if isinstance(end, str):
        end = datetime.strptime(end, r"%Y-%m-%d")
    if start > end:
        raise ValueError("Start date must be before end date.")
    if not isinstance(save_dir, str):
        raise TypeError("save_dir must be a string.")

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    end_inclusive = end
    end_exclusive = end_inclusive + relativedelta(days=1)

    client = cdsapi.Client()
    grid = [0.25, 0.25]
    if region == "local": # latitude and lontitude
        area = [40, 100, 5, 145]
    elif region == "global":
        area = [90, 0, -90, 360]
    elif region == "hrrr":
        area = [51, 236, 24, 289]
    else:
        raise ValueError(f"Unsupported region '{region}'.")

    if levels == "13_levels":
        pressure_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    elif levels == "8_levels":
        pressure_level = [1000, 925, 850, 700, 500, 300, 150, 50]
    else:
        raise ValueError(f"Unsupported level set '{levels}'.")

    if chunk_days <= 0:
        raise ValueError("chunk_days must be a positive integer.")

    hour_list = [f"{h:02d}:00" for h in range(24)]
    if batch_hours:
        current = start
        while current < end_exclusive:
            year = current.year
            month = current.month

            days = []
            cursor = current
            while cursor < end_exclusive and cursor.month == month:
                days.append(cursor.day)
                cursor += relativedelta(days=1)

            day_strs = [f"{day:02d}" for day in days]
            Y = f"{year:04d}"
            M = f"{month:02d}"

            month_dir = save_dir_path / f"{Y}/{Y}{M}"
            month_dir.mkdir(parents=True, exist_ok=True)

            print(f"***** Downloading [{year}/{month}] ({len(days)} days). *****")

            for offset in range(0, len(day_strs), chunk_days):
                chunk = day_strs[offset : offset + chunk_days]
                label_start = chunk[0]
                label_end = chunk[-1]
                chunk_label = f"{label_start}-{label_end}"

                sfc_file = month_dir / f"{Y}{M}_{chunk_label}_sfc.nc"
                upper_file = month_dir / f"{Y}{M}_{chunk_label}_upper.nc"

                print(f"  -> Requesting days {chunk_label} ({len(chunk)} days).")

                sfc_time = retrieve_with_retry(
                    client,
                    "reanalysis-era5-single-levels",
                    {
                        "variable": var_sfc,
                        "product_type": "reanalysis",
                        "year": Y,
                        "month": M,
                        "day": chunk,
                        "time": hour_list,
                        "area": area,
                        "grid": grid,
                        "format": "netcdf",
                    },
                    sfc_file,
                )

                upper_time = retrieve_with_retry(
                    client,
                    "reanalysis-era5-pressure-levels",
                    {
                        "variable": var_upper,
                        "pressure_level": pressure_level,
                        "product_type": "reanalysis",
                        "year": Y,
                        "month": M,
                        "day": chunk,
                        "time": hour_list,
                        "area": area,
                        "grid": grid,
                        "format": "netcdf",
                    },
                    upper_file,
                )

                hours_requested = len(chunk) * len(hour_list)
                if sfc_time is not None:
                    print(
                        f"    Surface batch ({hours_requested} hours) saved to {sfc_file.name} in {sfc_time:.1f}s."
                    )
                    _split_to_hourly(
                        src_path=sfc_file,
                        base_dir=save_dir_path,
                        suffix="_sfc.nc",
                    )
                if upper_time is not None:
                    print(
                        f"    Upper-air batch ({hours_requested} hours) saved to {upper_file.name} in {upper_time:.1f}s."
                    )
                    _split_to_hourly(
                        src_path=upper_file,
                        base_dir=save_dir_path,
                        suffix="_upper.nc",
                    )

            current = cursor
    else:
        dt = start
        while dt < end_exclusive:
            Y = f"{dt.year:04d}"
            M = f"{dt.month:02d}"
            D = f"{dt.day:02d}"
            output_dir = save_dir_path / f"{Y}/{Y}{M}/{Y}{M}{D}"
            if output_dir.exists():
                print(f"Warning: Directory {output_dir} already exists.", file=sys.stderr)

            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"***** Downloading [{dt.year}/{dt.month}/{dt.day}]. *****")

            def send_req(hour: str):
                hour_for_filename = f"{int(hour):02d}"
                sfc_time = retrieve_with_retry(
                    client,
                    "reanalysis-era5-single-levels",
                    {
                        "variable": var_sfc,
                        "product_type": "reanalysis",
                        "year": Y,
                        "month": M,
                        "day": D,
                        "time": hour,
                        "area": area,
                        "grid": grid,
                        "format": "netcdf",
                    },
                    output_dir / f"{Y}{M}{D}{hour_for_filename}_sfc.nc",
                )

                upper_time = retrieve_with_retry(
                    client,
                    "reanalysis-era5-pressure-levels",
                    {
                        "variable": var_upper,
                        "pressure_level": pressure_level,
                        "product_type": "reanalysis",
                        "year": Y,
                        "month": M,
                        "day": D,
                        "time": hour,
                        "area": area,
                        "grid": grid,
                        "format": "netcdf",
                    },
                    output_dir / f"{Y}{M}{D}{hour_for_filename}_upper.nc",
                )

                return hour_for_filename, sfc_time, upper_time

            hours_a_day = range(24)
            day_stats: list[tuple[str, float | None, float | None]] = []
            with ThreadPoolExecutor(max_workers=4) as executor: # 最多 4 條 thread 同時下載
                futures = [executor.submit(send_req, hour) for hour in hours_a_day]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            day_stats.append(result)
                    except Exception as e:
                        print(f"Error in thread: {e}")

            if day_stats:
                sfc_times = [t for _, t, _ in day_stats if t is not None]
                upper_times = [t for _, _, t in day_stats if t is not None]
                if sfc_times:
                    avg_sfc = sum(sfc_times) / len(sfc_times)
                    print(f"Average surface download time: {avg_sfc:.1f}s over {len(sfc_times)} files.")
                if upper_times:
                    avg_upper = sum(upper_times) / len(upper_times)
                    print(f"Average upper-air download time: {avg_upper:.1f}s over {len(upper_times)} files.")

            dt += relativedelta(days=1)


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
        default="./",
        help="Directory to save the downloaded data.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="hrrr",
        choices=["global", "local", "hrrr"],
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

    # Download surface and atmos data
    main(
        start=args.start,
        end=args.end,
        save_dir=args.save_dir,
        region=args.region,
        levels=args.levels,
        batch_hours=not args.no_batch_hours,
        chunk_days=args.chunk_days,
    )