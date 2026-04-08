import glob
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import csv
import re
import sys

def crop_helper(arr, patch_size):

    """
        crop_byzone_bymode_bypatchsize
        Crop array to multiple-of-4 and optionally to fixed (H,W).
        Determin slice first, cut later

        1D data : return lat_slice, lon_slice
        2D, 3D, 4D data : return arr
    """

    # cut to patch size
    if arr.ndim == 1 :
        l = len(arr)
        new_l = (l // patch_size) * patch_size
        return arr[:new_l]
    elif arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        c, h, w = arr.shape
    elif arr.ndim == 4:
        b, c, h, w = arr.shape

    new_h, new_w = (h // patch_size) * patch_size, (w // patch_size) * patch_size
    lat_slice = slice(0, new_h)
    lon_slice = slice(0, new_w)

    if arr.ndim == 1:
        pass
    elif arr.ndim == 2:
        h, w = arr.shape # 1059, 1799 265, 450
        arr = arr[lat_slice, lon_slice]
    elif arr.ndim == 3:
        c, h, w = arr.shape
        arr = arr[:, lat_slice, lon_slice]
    elif arr.ndim == 4:
        b, c, h, w = arr.shape
        arr = arr[:, :, lat_slice, lon_slice]

    return arr


def flip_south_to_north(arr):
    """Flip SOUTH→NORTH (CWA) to NORTH→SOUTH for Aurora."""
    if arr.ndim == 2:
        return np.flipud(arr).copy()  # <<< FIX HERE
    elif arr.ndim == 3:
        return np.flip(arr, axis=1).copy()  # <<< FIX HERE
    return arr.copy()


def regrid_nc_var(xr_field, weight_array, idx_array, patch_size):

    """
        Load xarray var → flip SOUTH to NORTH → crop -> regrid → torch tensor
    """

    arr = xr_field.values[0]

    arr = flip_south_to_north(arr)


    _, k = weight_array.shape

    if arr.ndim == 2 :

        H, W = arr.shape
        arr = arr.flatten()

        neighbor_values = arr[idx_array] # HW * K
        neighbor_values = neighbor_values.reshape(450, 450, k)
        neighbor_values = neighbor_values.transpose(2, 0, 1)
        neighbor_values = crop_helper(
            neighbor_values, patch_size
        )
        neighbor_values = neighbor_values.transpose(1, 2, 0)
        new_H, new_W, k = neighbor_values.shape
        neighbor_values = neighbor_values.reshape(-1, k)

        interpolated_grid = (neighbor_values * weight_array).sum(axis=1).reshape(new_H, new_W).astype(np.float32)


    elif arr.ndim == 3 :
        C, H, W = arr.shape
        interpolated_grid = []
        for c_idx in range(C) :

            arr_c = arr[c_idx].flatten()
            neighbor_values = arr_c[idx_array]
            neighbor_values = neighbor_values.reshape(H, W, k)
            neighbor_values = neighbor_values.transpose(2, 0, 1)
            neighbor_values = crop_helper(
                neighbor_values, patch_size
            )
            neighbor_values = neighbor_values.transpose(1, 2, 0)
            new_H, new_W, k = neighbor_values.shape
            neighbor_values = neighbor_values.reshape(-1, k)
            this_interpolated_grid = (neighbor_values * weight_array).sum(axis=1).reshape(new_H, new_W).astype(np.float32)
            interpolated_grid.append(this_interpolated_grid)

        interpolated_grid = np.stack(interpolated_grid, axis=0)

    else :
        raise "arr.shape error"

    return torch.tensor(interpolated_grid)


class CWA_ignore_missing(Dataset):
    def __init__(self, data_path, leadtime, step, dataset_time_range, dev=False, dataset_time_type="use_year", patch_size=4, whether_regrid=True, divergence_mode="none"):
        self.G0 = 9.80665  # m/s²
        self.csv_file_path = "./leadtime6_valid_folders.csv"
        self.data_path = data_path
        self.leadtime = leadtime
        self.step = step
        self.patch_size = patch_size
        self.whether_regrid = whether_regrid
        self.divergence_mode = divergence_mode

        self.tranditional_model_leadtime = 4

        """
            Collect all *_sfc.nc files
        """

        if dataset_time_type == "use_hour":
            start_year = int(dataset_time_range[0][:4])
            start_month = int(dataset_time_range[0][4:6]) # 01
            start_date = int(dataset_time_range[0][6:8]) # 01
            start_hour = int(dataset_time_range[0][8:10]) # 00
            start = datetime(start_year, start_month, start_date, start_hour)
            start = start - timedelta(hours=self.tranditional_model_leadtime) # 2022 12 31 20

            end_year = int(dataset_time_range[-1][:4])
            end_month = int(dataset_time_range[-1][4:6])
            end_date = int(dataset_time_range[-1][6:8]) # 01
            end_hour = int(dataset_time_range[-1][8:10]) # 00
            end = datetime(end_year, end_month, end_date, end_hour) + relativedelta(hours=1) # 2023 04 01
            end = end - timedelta(hours=self.tranditional_model_leadtime) # 2023 03 31 20


        elif dataset_time_type == "use_month":
            start_year = int(dataset_time_range[0][:4])
            start_month = int(dataset_time_range[0][4:6])
            start = datetime(start_year, start_month, 1)
            start = start - timedelta(hours=self.tranditional_model_leadtime) # 2022 12 31 20

            end_year = int(dataset_time_range[-1][:4])
            end_month = int(dataset_time_range[-1][4:6])
            end = datetime(end_year, end_month, 1) + relativedelta(months=1) # 2023 04 01
            end = end - timedelta(hours=self.tranditional_model_leadtime) # 2023 03 31 20


        else:
            start = datetime(int(dataset_time_range[0]), 1, 1)
            start = start - timedelta(hours=self.tranditional_model_leadtime) # 2020 12 31 20

            end = datetime(int(dataset_time_range[-1]) + 1, 1, 1)
            end = end - timedelta(hours=self.tranditional_model_leadtime) # 2023 12 31 20

        print(f"start = {start}, end = {end}")

        current = start # 2022 12 31 20 # 2020 12 31 20
        ideal_hours_count = 0

        while current < end:
            ideal_hours_count += 1

            current += timedelta(hours=1)

            if dev and ideal_hours_count == 20 :
                break

        self.folder_list = [] # store the folder time

        # check csv step/leadtime vs self.step / self.leadtime
        # 支援 step{N}_leadtime{M}_valid_folders.csv 或 leadtime{M}_valid_folders.csv（檔名無 step）
        filename = os.path.basename(self.csv_file_path)

        match_full = re.match(r"step(\d+)_leadtime(\d+)_valid_folders\.csv", filename)
        match_leadonly = re.match(r"leadtime(\d+)_valid_folders\.csv", filename)

        if match_full:
            csv_step, csv_leadtime = int(match_full.group(1)), int(match_full.group(2))
            if csv_step != self.step or csv_leadtime != self.leadtime:
                print("ERROR: CSV step/leadtime mismatch")
                print(f"CSV step={csv_step}, leadtime={csv_leadtime}")
                print(f"Dataset step={self.step}, leadtime={self.leadtime}")
                sys.exit(1)
        elif match_leadonly:
            csv_leadtime = int(match_leadonly.group(1))
            if csv_leadtime != self.leadtime:
                print("ERROR: CSV leadtime mismatch")
                print(f"CSV leadtime={csv_leadtime}")
                print(f"Dataset leadtime={self.leadtime}")
                sys.exit(1)
        else:
            print(f"CSV filename format error: {filename}")
            sys.exit(1)



        with open(self.csv_file_path) as f:

            reader = csv.reader(f)

            for row in reader:

                t = datetime.strptime(row[0], "%Y-%m-%d_%H")

                if start <= t < end:
                    self.folder_list.append(t)

                if dev and len(self.folder_list) == 20 :
                    break

        self.folder_list.sort()

        # for folder in self.folder_list:
        #     print(folder)


        print(f"In dataset CWA, # hour = {len(self.folder_list)} / {ideal_hours_count}, valid hour rate = {(len(self.folder_list) / ideal_hours_count * 100):.2f}")
        print(f"Use years : {dataset_time_range}")


        # """
        #     prepare weight array and idx array for on the fly regrid
        # """

        if self.whether_regrid == True : # original_data (on the fly regrid)

            weight_array = np.load("./CWA_Dataset/regrid/reusable_resource/cwa_regrid_weight_array.npyy")  # shape: (1059,)

            H, W, k = weight_array.shape
            weight_array = weight_array.transpose(2, 0, 1)
            weight_array = crop_helper(
                weight_array, self.patch_size
            )
            k, new_H, new_W = weight_array.shape
            weight_array = weight_array.transpose(1, 2, 0)
            self.weight_array = weight_array.reshape(-1, k)


            self.idx_array = np.load("./CWA_Dataset/regrid/reusable_resource/cwa_regrid_nearst_location.npyy") # shape: (1799,)

            new_lat = np.load("./CWA_Dataset/regrid/reusable_resource/cwa_regrid_latitude.npyy") # H
            new_lon = np.load("./CWA_Dataset/regrid/reusable_resource/cwa_regrid_longitude.npyy") # W

            self.new_lat = crop_helper(new_lat, self.patch_size)
            self.new_lon = crop_helper(new_lon, self.patch_size)


        # """
        #     Read Static vars (regrided data)
        # """
        static_ds = xr.open_mfdataset(f"./regrid_static/cwa_static_regrided.nc")

        self.static_vars = {}
        for varname in ["slt"]:
            arr = static_ds[varname].values[0]

            arr = crop_helper(arr, self.patch_size).astype(np.float32)

            self.static_vars[varname] = torch.tensor(arr)

        lsm_z_ds = xr.open_mfdataset(f"./regrid_static/lsm_z_regrided.nc")

        self.static_vars["lsm"] = self.load_field(lsm_z_ds["LANDMASK"])
        self.static_vars["z"] = self.load_field(lsm_z_ds["HGT"]) * self.G0

    def load_field(self, xr_field):
        arr = xr_field.values[0]
        return torch.tensor(arr)

    # the first (self.step + 1) * self.leadtime don't have t-1 and t, so they are not data
    def __len__(self):
        return len(self.folder_list)


    def __getitem__(self, index):

        start_time = self.folder_list[index]

        # ---------------- time steps ----------------
        time_list = [
            start_time + timedelta(hours=i * self.leadtime)
            for i in range(self.step + 2)
        ] # folder time

        #
        ncfile_list = []
        for t in time_list :
            folder_name = t.strftime("%Y-%m-%d_%H")
            ncfile_name_t = t + timedelta(hours=self.tranditional_model_leadtime)
            ncfile_name = f"wrfout_d01_{ncfile_name_t.strftime('%Y-%m-%d_%H')}_interp.nc"

            if t.year <= 2023:
                ncfile_path = os.path.join(self.data_path, "rwrf")
            else:
                ncfile_path = os.path.join(self.data_path, "rwrf_data")

            folder_path = os.path.join(ncfile_path, folder_name, ncfile_name)

            ncfile_list.append(xr.open_dataset(folder_path))


        atmos_levels = torch.tensor(ncfile_list[0]["pres_levels"].values.astype(np.int32))


        # process input by on the fly regrid and dieractly load data

        if self.whether_regrid == True :

            lat = torch.tensor(self.new_lat)
            lon = torch.tensor(self.new_lon)

            input_surf = {
                "2t":  torch.stack([regrid_nc_var(ncfile_list[0]["T2"], self.weight_array, self.idx_array, self.patch_size), regrid_nc_var(ncfile_list[1]["T2"], self.weight_array, self.idx_array, self.patch_size)], dim=0),
                "10u": torch.stack([regrid_nc_var(ncfile_list[0]["umet10"], self.weight_array, self.idx_array, self.patch_size), regrid_nc_var(ncfile_list[1]["umet10"], self.weight_array, self.idx_array, self.patch_size)], dim=0),
                "10v": torch.stack([regrid_nc_var(ncfile_list[0]["vmet10"], self.weight_array, self.idx_array, self.patch_size), regrid_nc_var(ncfile_list[1]["vmet10"], self.weight_array, self.idx_array, self.patch_size)], dim=0),
                "msl": torch.stack([regrid_nc_var(ncfile_list[0]["PSFC"], self.weight_array, self.idx_array, self.patch_size), regrid_nc_var(ncfile_list[1]["PSFC"], self.weight_array, self.idx_array, self.patch_size)], dim=0),
            }

            input_atmos = {
                "t": torch.stack([regrid_nc_var(ncfile_list[0]["tk_p"], self.weight_array, self.idx_array, self.patch_size), regrid_nc_var(ncfile_list[1]["tk_p"], self.weight_array, self.idx_array, self.patch_size)], dim=0),
                "u": torch.stack([regrid_nc_var(ncfile_list[0]["umet_p"], self.weight_array, self.idx_array, self.patch_size), regrid_nc_var(ncfile_list[1]["umet_p"], self.weight_array, self.idx_array, self.patch_size)], dim=0),
                "v": torch.stack([regrid_nc_var(ncfile_list[0]["vmet_p"], self.weight_array, self.idx_array, self.patch_size), regrid_nc_var(ncfile_list[1]["vmet_p"], self.weight_array, self.idx_array, self.patch_size)], dim=0),
                "q": torch.stack([regrid_nc_var(ncfile_list[0]["QVAPOR_p"], self.weight_array, self.idx_array, self.patch_size), regrid_nc_var(ncfile_list[1]["QVAPOR_p"], self.weight_array, self.idx_array, self.patch_size)], dim=0),
                "z": torch.stack([regrid_nc_var(ncfile_list[0]["z_p"], self.weight_array, self.idx_array, self.patch_size), regrid_nc_var(ncfile_list[1]["z_p"], self.weight_array, self.idx_array, self.patch_size)], dim=0) * self.G0,
            }

            if self.divergence_mode == "vertical":
                input_atmos["w"] = torch.stack([regrid_nc_var(ncfile_list[0]["wa_p"], self.weight_array, self.idx_array, self.patch_size), regrid_nc_var(ncfile_list[1]["wa_p"], self.weight_array, self.idx_array, self.patch_size)], dim=0)


        else :

            # IMPORTANT : need to be check whether the numeric is right
            lat_np = ncfile_list[0]["XLAT"].values.astype(np.float32)
            lon_np = ncfile_list[0]["XLONG"].values.astype(np.float32)

            lat = torch.tensor(lat_np)
            lon = torch.tensor(lon_np)


            input_surf = {
                "2t":  torch.stack([self.load_field(ncfile_list[0]["T2"]), self.load_field(ncfile_list[1]["T2"])], dim=0),
                "10u": torch.stack([self.load_field(ncfile_list[0]["umet10"]), self.load_field(ncfile_list[1]["umet10"])], dim=0),
                "10v": torch.stack([self.load_field(ncfile_list[0]["vmet10"]), self.load_field(ncfile_list[1]["vmet10"])], dim=0),
                "msl": torch.stack([self.load_field(ncfile_list[0]["PSFC"]), self.load_field(ncfile_list[1]["PSFC"])], dim=0),
            }

            input_atmos = {
                "t": torch.stack([self.load_field(ncfile_list[0]["tk_p"]), self.load_field(ncfile_list[1]["tk_p"])], dim=0),
                "u": torch.stack([self.load_field(ncfile_list[0]["umet_p"]), self.load_field(ncfile_list[1]["umet_p"])], dim=0),
                "v": torch.stack([self.load_field(ncfile_list[0]["vmet_p"]), self.load_field(ncfile_list[1]["vmet_p"])], dim=0),
                "q": torch.stack([self.load_field(ncfile_list[0]["QVAPOR_p"]), self.load_field(ncfile_list[1]["QVAPOR_p"])], dim=0),
                "z": torch.stack([self.load_field(ncfile_list[0]["z_p"]), self.load_field(ncfile_list[1]["z_p"])], dim=0) * self.G0,
            }



        # ===================== INPUT =====================
        input = {
            "surf_vars": input_surf,
            "atmos_vars": input_atmos,
            "static_vars": self.static_vars,
            "lat": lat,
            "lon": lon,
            "time": time_list[1].strftime("%Y%m%d%H"),
            "atmos_levels": atmos_levels,
        }

        # process labels

        labels = []

        for i in range(self.step):
            tstep = {}

            if self.whether_regrid == True :

                tstep["surf_vars"] = {
                    "2t":  regrid_nc_var(ncfile_list[i+2]["T2"], self.weight_array, self.idx_array, self.patch_size).unsqueeze(0),
                    "10u": regrid_nc_var(ncfile_list[i+2]["umet10"], self.weight_array, self.idx_array, self.patch_size).unsqueeze(0),
                    "10v": regrid_nc_var(ncfile_list[i+2]["vmet10"], self.weight_array, self.idx_array, self.patch_size).unsqueeze(0),
                    "msl": regrid_nc_var(ncfile_list[i+2]["PSFC"], self.weight_array, self.idx_array, self.patch_size).unsqueeze(0),
                }

                tstep["atmos_vars"] = {
                    "t": regrid_nc_var(ncfile_list[i+2]["tk_p"], self.weight_array, self.idx_array, self.patch_size).unsqueeze(0),
                    "u": regrid_nc_var(ncfile_list[i+2]["umet_p"], self.weight_array, self.idx_array, self.patch_size).unsqueeze(0),
                    "v": regrid_nc_var(ncfile_list[i+2]["vmet_p"], self.weight_array, self.idx_array, self.patch_size).unsqueeze(0),
                    "q": regrid_nc_var(ncfile_list[i+2]["QVAPOR_p"], self.weight_array, self.idx_array, self.patch_size).unsqueeze(0),
                    "z": regrid_nc_var(ncfile_list[i+2]["z_p"], self.weight_array, self.idx_array, self.patch_size).unsqueeze(0) * self.G0,
                }

                if self.divergence_mode == "vertical" or self.divergence_mode == "horizontal" :
                    tstep["atmos_vars"]["w"] = regrid_nc_var(ncfile_list[i+2]["wa_p"], self.weight_array, self.idx_array, self.patch_size).unsqueeze(0)

            else :
                tstep["surf_vars"] = {
                    "2t":  self.load_field(ncfile_list[i+2]["T2"]).unsqueeze(0),
                    "10u": self.load_field(ncfile_list[i+2]["umet10"]).unsqueeze(0),
                    "10v": self.load_field(ncfile_list[i+2]["vmet10"]).unsqueeze(0),
                    "msl": self.load_field(ncfile_list[i+2]["PSFC"]).unsqueeze(0),
                }

                tstep["atmos_vars"] = {
                    "t": self.load_field(ncfile_list[i+2]["tk_p"]).unsqueeze(0),
                    "u": self.load_field(ncfile_list[i+2]["umet_p"]).unsqueeze(0),
                    "v": self.load_field(ncfile_list[i+2]["vmet_p"]).unsqueeze(0),
                    "q": self.load_field(ncfile_list[i+2]["QVAPOR_p"]).unsqueeze(0),
                    "z": self.load_field(ncfile_list[i+2]["z_p"]).unsqueeze(0) * self.G0,
                }

                if self.divergence_mode == "vertical" or self.divergence_mode == "horizontal" :
                    tstep["atmos_vars"]["w"] = self.load_field(ncfile_list[i+2]["wa_p"]).unsqueeze(0)



            tstep["static_vars"] = self.static_vars
            tstep["lat"] = lat
            tstep["lon"] = lon
            tstep["time"] = time_list[2].strftime("%Y%m%d%H")
            tstep["atmos_levels"] = atmos_levels

            labels.append(tstep)

        return input, labels


class Global_ERA5(Dataset):
    def __init__(self, split, root, leadtime, step):
        splits = {
            'train': ['2021'],
            'val': ['2024'],
            'test': ['2024'],
        }
        years = splits[split]
        self.root = root
        self.sfc_dir = []
        for year in years:
            sfc_path = f'{root}/{year}/*/*/*_sfc.nc'
            self.sfc_dir += sorted(glob.glob(sfc_path))
        start_file_name = self.sfc_dir[0].split("/")[-1].split("_")[0]
        self.start_time = datetime.strptime(start_file_name,"%Y%m%d%H")

        self.static_dir = glob.glob(f"{root}/static/*.nc")
        self.static_ds = xr.open_mfdataset(self.static_dir)

        self.leadtime=leadtime
        self.step=step if (split=='test') else 1

    def __len__(self):
        return len(self.sfc_dir)-(self.step+1)*self.leadtime

    def __getitem__(self, index):
        time_list = [self.start_time + timedelta(hours=index+i*self.leadtime) for i in range(self.step+2)]
        sfc_list = [xr.open_dataset(f'{self.root}/{time_list[i].strftime("%Y")}/{time_list[i].strftime("%Y%m")}/{time_list[i].strftime("%Y%m%d")}/{time_list[i].strftime("%Y%m%d%H")}_sfc.nc') for i in range(self.step+2)]
        upper_list = [xr.open_dataset(f'{self.root}/{time_list[i].strftime("%Y")}/{time_list[i].strftime("%Y%m")}/{time_list[i].strftime("%Y%m%d")}/{time_list[i].strftime("%Y%m%d%H")}_upper.nc') for i in range(self.step+2)]

        # get lat and lon
        lat = torch.tensor(upper_list[0]["latitude"].values[:-1].astype(np.float32))
        lon = torch.tensor(upper_list[0]["longitude"].values.astype(np.float32))
        atmos_levels = torch.tensor(upper_list[0]["pressure_level"].values.astype(np.int32))

        input = {
            "surf_vars" :{
                "2t" : torch.stack([torch.tensor(sfc_list[0]["t2m"].values[0,:-1,:].astype(np.float32)), \
                                    torch.tensor(sfc_list[1]["t2m"].values[0,:-1,:].astype(np.float32))], dim=0),
                "10u": torch.stack([torch.tensor(sfc_list[0]["u10"].values[0,:-1,:].astype(np.float32)), \
                                    torch.tensor(sfc_list[1]["u10"].values[0,:-1,:].astype(np.float32))], dim=0),
                "10v": torch.stack([torch.tensor(sfc_list[0]["v10"].values[0,:-1,:].astype(np.float32)), \
                                    torch.tensor(sfc_list[1]["v10"].values[0,:-1,:].astype(np.float32))], dim=0),
                "msl": torch.stack([torch.tensor(sfc_list[0]["msl"].values[0,:-1,:].astype(np.float32)), \
                                    torch.tensor(sfc_list[1]["msl"].values[0,:-1,:].astype(np.float32))], dim=0),
            },
            "atmos_vars" : {
                "t": torch.stack([torch.tensor(upper_list[0]["t"].values[0,:,:-1,:].astype(np.float32)), \
                                  torch.tensor(upper_list[1]["t"].values[0,:,:-1,:].astype(np.float32))], dim=0),
                "u": torch.stack([torch.tensor(upper_list[0]["u"].values[0,:,:-1,:].astype(np.float32)), \
                                  torch.tensor(upper_list[1]["u"].values[0,:,:-1,:].astype(np.float32))], dim=0),
                "v": torch.stack([torch.tensor(upper_list[0]["v"].values[0,:,:-1,:].astype(np.float32)), \
                                  torch.tensor(upper_list[1]["v"].values[0,:,:-1,:].astype(np.float32))], dim=0),
                "q": torch.stack([torch.tensor(upper_list[0]["q"].values[0,:,:-1,:].astype(np.float32)), \
                                  torch.tensor(upper_list[1]["q"].values[0,:,:-1,:].astype(np.float32))], dim=0),
                "z": torch.stack([torch.tensor(upper_list[0]["z"].values[0,:,:-1,:].astype(np.float32)), \
                                  torch.tensor(upper_list[1]["z"].values[0,:,:-1,:].astype(np.float32))], dim=0),
            },
            "static_vars" : {
                "z"  : torch.tensor(self.static_ds["z"  ].values[:,:-1,:].astype(np.float32)).squeeze(0),
                "slt": torch.tensor(self.static_ds["slt"].values[:,:-1,:].astype(np.float32)).squeeze(0),
                "lsm": torch.tensor(self.static_ds["lsm"].values[:,:-1,:].astype(np.float32)).squeeze(0),
            },
            "lat" : lat,
            "lon" : lon,
            "time": time_list[1].strftime("%Y%m%d%H"),
            "atmos_levels":atmos_levels
        }

        label = [{
            "surf_vars" :{
                "2t" : torch.tensor(sfc_list[i+2]["t2m"].values[0,:-1,:].astype(np.float32)).unsqueeze(0),
                "10u": torch.tensor(sfc_list[i+2]["u10"].values[0,:-1,:].astype(np.float32)).unsqueeze(0),
                "10v": torch.tensor(sfc_list[i+2]["v10"].values[0,:-1,:].astype(np.float32)).unsqueeze(0),
                "msl": torch.tensor(sfc_list[i+2]["msl"].values[0,:-1,:].astype(np.float32)).unsqueeze(0),
            },
            "atmos_vars" : {
                "t": torch.tensor(upper_list[i+2]["t"].values[0,:,:-1,:].astype(np.float32)).unsqueeze(0),
                "u": torch.tensor(upper_list[i+2]["u"].values[0,:,:-1,:].astype(np.float32)).unsqueeze(0),
                "v": torch.tensor(upper_list[i+2]["v"].values[0,:,:-1,:].astype(np.float32)).unsqueeze(0),
                "q": torch.tensor(upper_list[i+2]["q"].values[0,:,:-1,:].astype(np.float32)).unsqueeze(0),
                "z": torch.tensor(upper_list[i+2]["z"].values[0,:,:-1,:].astype(np.float32)).unsqueeze(0),
            },
            "static_vars" : {
                "z"  : torch.tensor(self.static_ds["z"  ].values[:,:-1,:].astype(np.float32)).squeeze(0),
                "slt": torch.tensor(self.static_ds["slt"].values[:,:-1,:].astype(np.float32)).squeeze(0),
                "lsm": torch.tensor(self.static_ds["lsm"].values[:,:-1,:].astype(np.float32)).squeeze(0),
            },
            "lat" : lat,
            "lon" : lon,
            "time": time_list[2].strftime("%Y%m%d%H"), # don't care
            "atmos_levels":atmos_levels
        } for i in range(self.step)]

        return input, label


class ERA5_Global_Crop(Dataset):
    def __init__(self, split, root, leadtime, step,
                 lat_min=None, lat_max=None, lon_min=None, lon_max=None,
                 use_month=False, month_splits=None):
        """
        Args:
            split: 'train', 'val', or 'test'
            root: Root directory of the dataset
            leadtime: Time interval between input timesteps
            step: Number of prediction steps
            lat_min, lat_max, lon_min, lon_max: Spatial crop boundaries (optional)
            use_month: If True, use month-based splits (like Global_ERA5_month),
                      otherwise use year-based splits (default)
            month_splits: Dictionary for month-based splits, e.g.,
                         {'train': ['202101', '202102'], 'val': ['202401'], 'test': ['202401']}
                         If None and use_month=True, uses default month splits
        """
        # Default spatial boundaries (US Central region)
        self.lat_min = lat_min if lat_min is not None else 5.0
        self.lat_max = lat_max if lat_max is not None else 40.0
        self.lon_min = lon_min if lon_min is not None else 100.0
        self.lon_max = lon_max if lon_max is not None else 145.0

        self.use_month = use_month

        if use_month:
            # Month-based splits (like Global_ERA5_month)
            if month_splits is None:
                month_splits = {
                    'train': ['202301'],
                    'val': ['202401'],
                    'test': ['202401'],
                }
            time_periods = month_splits[split]
            self.root = root
            self.sfc_dir = []
            for month in time_periods:
                sfc_path = f'{root}/{month[:-2]}/{month}/*/*_sfc.nc'
                self.sfc_dir += sorted(glob.glob(sfc_path))
        else:
            # Year-based splits
            splits = {
                'train': ['2023'],
                'val': ['2024'],
                'test': ['2024'],
            }
            years = splits[split]
            self.root = root
            self.sfc_dir = []
            for year in years:
                sfc_path = f'{root}/{year}/*/*/*_sfc.nc'
                self.sfc_dir += sorted(glob.glob(sfc_path))

        if len(self.sfc_dir) == 0:
            raise ValueError(f"No files found for split '{split}' with use_month={use_month}")

        start_file_name = self.sfc_dir[0].split("/")[-1].split("_")[0]
        self.start_time = datetime.strptime(start_file_name,"%Y%m%d%H")

        self.static_dir = glob.glob(f"{root}/static/*.nc")
        self.static_ds = xr.open_mfdataset(self.static_dir)

        self._init_spatial_indices()
        self._preprocess_static_data()

        self.leadtime = leadtime
        self.step = step if (split == 'test') else 1
        self.lat_slice = slice(self.lat_start_idx, self.lat_end_idx)
        self.lon_slice = slice(self.lon_start_idx, self.lon_end_idx)
        self.n_lat = self.lat_end_idx - self.lat_start_idx - 1  # -1 for [:-1]
        self.n_lon = self.lon_end_idx - self.lon_start_idx - 1
        self.spatial_shape = (self.n_lat, self.n_lon)
        self.sfc_var_map = [("2t", "t2m"), ("10u", "u10"), ("10v", "v10"), ("msl", "msl")]
        self.atmos_var_list = ["t", "u", "v", "q", "z"]

    def _init_spatial_indices(self):
        """Initialize spatial indices by reading coordinates from first file."""
        # Read first file to get coordinate structure
        sample_file = self.sfc_dir[0]
        sample_ds = xr.open_dataset(sample_file.replace('_sfc.nc', '_upper.nc'))

        # Get latitude and longitude arrays
        lat_array = sample_ds["latitude"].values
        lon_array = sample_ds["longitude"].values

        # Find latitude indices (5N ~ 40N)
        # Note: ERA5 lat is typically descending (90N to -90S)
        lat_mask = (lat_array >= self.lat_min) & (lat_array <= self.lat_max)
        lat_indices = np.where(lat_mask)[0]

        if len(lat_indices) == 0:
            sample_ds.close()
            raise ValueError(f"No latitude indices found for range [{self.lat_min}, {self.lat_max}]. "
                           f"Available range: [{lat_array.min():.2f}, {lat_array.max():.2f}]")

        # Ensure indices are in correct order (start < end)
        self.lat_start_idx = int(min(lat_indices))
        self.lat_end_idx = int(max(lat_indices)) + 1  # +1 for slice (exclusive end)

        # Find longitude indices (100E ~ 145E)
        # Handle both 0-360 and -180-180 formats
        lon_mask = (lon_array >= self.lon_min) & (lon_array <= self.lon_max)
        lon_indices = np.where(lon_mask)[0]

        if len(lon_indices) == 0:
            # Try alternative: if lon is in 0-360 format, values are already correct
            # If lon is in -180-180 format, we need to check if range wraps around
            if lon_array.min() < 0 and lon_array.max() <= 180:
                # -180 to 180 format: 100-145 is already correct, but might not exist
                # This case should have been caught above, so this is a fallback
                pass
            elif lon_array.max() > 180:
                # 0-360 format: 100-145 should work directly
                # If still no match, there might be an issue with the data
                pass

            sample_ds.close()
            raise ValueError(f"No longitude indices found for range [{self.lon_min}, {self.lon_max}]. "
                           f"Available range: [{lon_array.min():.2f}, {lon_array.max():.2f}]. "
                           f"Format: {'0-360' if lon_array.max() > 180 else '-180-180'}")

        # Handle longitude wrapping: ensure indices are contiguous
        # For 0-360 format, indices should be contiguous
        # For -180-180 format, if range crosses 180, indices might wrap
        if len(lon_indices) > 1 and lon_indices[-1] - lon_indices[0] != len(lon_indices) - 1:
            # Non-contiguous indices - might indicate wrapping issue
            # For now, take the first contiguous block
            lon_indices = np.array(sorted(lon_indices))
            # Find the largest contiguous block
            diff = np.diff(lon_indices)
            breaks = np.where(diff > 1)[0]
            if len(breaks) > 0:
                # Take the first contiguous block
                lon_indices = lon_indices[:breaks[0]+1]

        self.lon_start_idx = int(min(lon_indices))
        self.lon_end_idx = int(max(lon_indices)) + 1  # +1 for slice (exclusive end)

        sample_ds.close()

    def _preprocess_static_data(self):
        """Pre-crop static data to avoid repeated processing in __getitem__."""
        # Crop static variables once during initialization
        static_z_full = self.static_ds["z"].values
        static_z_cropped = static_z_full[:, self.lat_start_idx:self.lat_end_idx, self.lon_start_idx:self.lon_end_idx]
        self.static_z = torch.tensor(static_z_cropped[:, :-1, :-1].astype(np.float32)).squeeze(0)

        static_slt_full = self.static_ds["slt"].values
        static_slt_cropped = static_slt_full[:, self.lat_start_idx:self.lat_end_idx, self.lon_start_idx:self.lon_end_idx]
        self.static_slt = torch.tensor(static_slt_cropped[:, :-1, :-1].astype(np.float32)).squeeze(0)

        static_lsm_full = self.static_ds["lsm"].values
        static_lsm_cropped = static_lsm_full[:, self.lat_start_idx:self.lat_end_idx, self.lon_start_idx:self.lon_end_idx]
        self.static_lsm = torch.tensor(static_lsm_cropped[:, :-1, :-1].astype(np.float32)).squeeze(0)

    def __len__(self):
        return len(self.sfc_dir) - (self.step + 1) * self.leadtime

    def __getitem__(self, index):
        total_start = time.time()
        time_list = [self.start_time + timedelta(hours=index+i*self.leadtime)
                     for i in range(self.step+2)]

        # === FILE OPENING ===
        file_open_start = time.time()
        sfc_list = []
        upper_list = []

        for i in range(self.step+2):
            sfc_path = f'{self.root}/{time_list[i].strftime("%Y")}/{time_list[i].strftime("%Y%m")}/{time_list[i].strftime("%Y%m%d")}/{time_list[i].strftime("%Y%m%d%H")}_sfc.nc'
            upper_path = sfc_path.replace('_sfc.nc', '_upper.nc')

            try:
                sfc_list.append(xr.open_dataset(sfc_path, engine='netcdf4', decode_times=False))
                upper_list.append(xr.open_dataset(upper_path, engine='netcdf4', decode_times=False))
            except:
                sfc_list.append(xr.open_dataset(sfc_path, decode_times=False))
                upper_list.append(xr.open_dataset(upper_path, decode_times=False))
        file_open_time = time.time() - file_open_start

        # === COORDINATE EXTRACTION ===
        coord_start = time.time()
        # latitude and longitude are 1D coordinate variables, no need for isel
        lat = torch.from_numpy(upper_list[0]["latitude"].values[self.lat_start_idx:self.lat_end_idx][:-1].astype(np.float32))
        lon = torch.from_numpy(upper_list[0]["longitude"].values[self.lon_start_idx:self.lon_end_idx][:-1].astype(np.float32))
        atmos_levels = torch.from_numpy(upper_list[0]["pressure_level"].values.astype(np.int32))
        coord_time = time.time() - coord_start

        # === DATA PROCESSING ===
        data_process_start = time.time()

        n_surf_vars = 4
        n_atmos_vars = 5
        n_levels = len(atmos_levels)

        input_surf_vars = torch.empty((n_surf_vars, 2, *self.spatial_shape), dtype=torch.float32)
        input_atmos_vars = torch.empty((n_atmos_vars, 2, n_levels, *self.spatial_shape), dtype=torch.float32)

        # Surface variables
        # Match Raw_ERA5 exactly: first select time (index 0), then crop spatial region, then apply [:-1,:-1]
        # Raw_ERA5: sfc_list[0]["t2m"].values[0,:-1,:-1]
        # This means: select time[0], then apply [:-1,:-1] to lat and lon dimensions
        # CRITICAL: Must match Raw_ERA5's approach: .values[0,:-1,:-1] after spatial cropping
        for var_idx, (_, actual_name) in enumerate(self.sfc_var_map):
            for time_idx in range(2):
                # First, get full data with time dimension: .values[0] selects time dimension
                # Then crop spatial region using numpy slicing (matching Raw_ERA5's approach)
                full_data = sfc_list[time_idx][actual_name].values[0]  # Shape: (lat_full, lon_full)
                # Crop to East Asia region using numpy slicing
                cropped = full_data[self.lat_start_idx:self.lat_end_idx,
                                   self.lon_start_idx:self.lon_end_idx]  # Shape: (lat_size, lon_size)
                # Apply [:-1, :-1] slicing to match Raw_ERA5 format exactly
                data = cropped[:-1, :-1]  # Shape: (lat_size-1, lon_size-1)
                # Convert to float32 and ensure C-contiguous
                input_surf_vars[var_idx, time_idx] = torch.from_numpy(
                    np.ascontiguousarray(data, dtype=np.float32)
                )

        # Atmospheric variables
        # Match Raw_ERA5 exactly: first select time (index 0), then crop spatial region, then apply [:,:-1,:-1]
        # Raw_ERA5: upper_list[0]["t"].values[0,:,:-1,:-1]
        # This means: select time[0], keep all levels [:], then apply [:-1,:-1] to lat and lon dimensions
        for var_idx, var_name in enumerate(self.atmos_var_list):
            for time_idx in range(2):
                # First, get full data with time dimension: .values[0] selects time dimension
                # Then crop spatial region using numpy slicing (matching Raw_ERA5's approach)
                full_data = upper_list[time_idx][var_name].values[0]  # Shape: (n_levels, lat_full, lon_full)
                # Crop to East Asia region using numpy slicing
                cropped = full_data[:, self.lat_start_idx:self.lat_end_idx,
                                   self.lon_start_idx:self.lon_end_idx]  # Shape: (n_levels, lat_size, lon_size)
                # Apply [:, :-1, :-1] slicing to match Raw_ERA5 format exactly
                data = cropped[:, :-1, :-1]  # Shape: (n_levels, lat_size-1, lon_size-1)
                # Convert to float32 and ensure C-contiguous
                input_atmos_vars[var_idx, time_idx] = torch.from_numpy(
                    np.ascontiguousarray(data, dtype=np.float32)
                )

        input = {
            "surf_vars": {k: input_surf_vars[i] for i, (k, _) in enumerate(self.sfc_var_map)},
            "atmos_vars": {k: input_atmos_vars[i] for i, k in enumerate(self.atmos_var_list)},
            "static_vars": {"z": self.static_z, "slt": self.static_slt, "lsm": self.static_lsm},
            "lat": lat, "lon": lon, "atmos_levels": atmos_levels,
            "time": time_list[1].strftime("%Y%m%d%H")
        }
        data_process_time = time.time() - data_process_start

        # === LABEL PROCESSING ===
        label_start = time.time()
        label = []
        label_surf = torch.empty((self.step, n_surf_vars, *self.spatial_shape), dtype=torch.float32)
        label_atmos = torch.empty((self.step, n_atmos_vars, n_levels, *self.spatial_shape), dtype=torch.float32)

        for step_idx in range(self.step):
            ds_idx = step_idx + 2

            for var_idx, (_, actual_name) in enumerate(self.sfc_var_map):
                # Match Raw_ERA5 exactly: sfc_list[i+2]["t2m"].values[0,:-1,:-1]
                # Step 1: Select time dimension (index 0)
                full_data = sfc_list[ds_idx][actual_name].values[0]  # Shape: (lat_full, lon_full)
                # Step 2: Crop to East Asia region using numpy slicing
                cropped = full_data[self.lat_start_idx:self.lat_end_idx,
                                   self.lon_start_idx:self.lon_end_idx]  # Shape: (lat_size, lon_size)
                # Step 3: Apply [:-1, :-1] slicing to match Raw_ERA5 format exactly
                data = cropped[:-1, :-1]  # Shape: (lat_size-1, lon_size-1)
                # Convert to float32 and ensure C-contiguous
                label_surf[step_idx, var_idx] = torch.from_numpy(
                    np.ascontiguousarray(data, dtype=np.float32)
                )

            for var_idx, var_name in enumerate(self.atmos_var_list):
                # Match Raw_ERA5 exactly: upper_list[i+2]["t"].values[0,:,:-1,:-1]
                # Step 1: Select time dimension (index 0)
                full_data = upper_list[ds_idx][var_name].values[0]  # Shape: (n_levels, lat_full, lon_full)
                # Step 2: Crop to East Asia region using numpy slicing
                cropped = full_data[:, self.lat_start_idx:self.lat_end_idx,
                                   self.lon_start_idx:self.lon_end_idx]  # Shape: (n_levels, lat_size, lon_size)
                # Step 3: Apply [:, :-1, :-1] slicing to match Raw_ERA5 format exactly
                data = cropped[:, :-1, :-1]  # Shape: (n_levels, lat_size-1, lon_size-1)
                # Convert to float32 and ensure C-contiguous
                label_atmos[step_idx, var_idx] = torch.from_numpy(
                    np.ascontiguousarray(data, dtype=np.float32)
                )

            label.append({
                "surf_vars": {k: label_surf[step_idx, i] for i, (k, _) in enumerate(self.sfc_var_map)},
                "atmos_vars": {k: label_atmos[step_idx, i] for i, k in enumerate(self.atmos_var_list)},
                "static_vars": {"z": self.static_z, "slt": self.static_slt, "lsm": self.static_lsm},
                "lat": lat, "lon": lon, "atmos_levels": atmos_levels,
                "time": time_list[ds_idx].strftime("%Y%m%d%H")
            })
        label_time = time.time() - label_start

        # === CLEANUP ===
        close_start = time.time()
        for ds in sfc_list + upper_list:
            ds.close()
        close_time = time.time() - close_start

        # === TIMING OUTPUT ===
        total_time = time.time() - total_start
        if index < 3:  # Only print for first 3 samples
            print(f"\n[Sample {index}] Timing breakdown:")
            print(f" File open:     {file_open_time:.2f}s ({file_open_time/total_time*100:.1f}%)")
            print(f" Coord extract:  {coord_time:.2f}s ({coord_time/total_time*100:.1f}%)")
            print(f" Data process:   {data_process_time:.2f}s ({data_process_time/total_time*100:.1f}%)")
            print(f" Label process:  {label_time:.2f}s ({label_time/total_time*100:.1f}%)")
            print(f" File close:     {close_time:.2f}s ({close_time/total_time*100:.1f}%)")
            print(f" Total:          {total_time:.2f}s")

        return input, label