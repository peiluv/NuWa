import glob
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os

# def determin_time_range(args):

#     if args.dataset_time_type == "use_hour":
#         start_year = int(args.selected_time_range[0][:4])
#         start_month = int(args.selected_time_range[0][4:6]) # 01
#         start_date = int(args.selected_time_range[0][6:8]) # 01
#         start_hour = int(args.selected_time_range[0][8:10]) # 00
#         start = datetime(start_year, start_month, start_date, start_hour)
#         start = start - timedelta(hours=args.tranditional_model_leadtime) # 2022 12 31 20

#         end_year = int(args.selected_time_range[-1][:4])
#         end_month = int(args.selected_time_range[-1][4:6]) 
#         end_date = int(args.selected_time_range[-1][6:8]) # 01
#         end_hour = int(args.selected_time_range[-1][8:10]) # 00
#         end = datetime(end_year, end_month, end_date, end_hour) + timedelta(hours=1) # 2023 04 01
#         end = end - timedelta(hours=args.tranditional_model_leadtime) # 2023 03 31 20

#     elif args.dataset_time_type == "use_date":
#         start_year = int(args.selected_time_range[0][:4])
#         start_month = int(args.selected_time_range[0][4:6]) # 01
#         start_date = int(args.selected_time_range[0][6:8]) # 01
#         start = datetime(start_year, start_month, start_date)
#         start = start - timedelta(hours=args.tranditional_model_leadtime) # 2022 12 31 20

#         end_year = int(args.selected_time_range[-1][:4])
#         end_month = int(args.selected_time_range[-1][4:6]) 
#         end_date = int(args.selected_time_range[-1][6:8]) # 01
#         end = datetime(end_year, end_month, end_date) + timedelta(hours=1) # 2023 04 01
#         end = end - timedelta(hours=args.tranditional_model_leadtime) # 2023 03 31 20

#     elif args.dataset_time_type == "use_month":
#         start_year = int(args.selected_time_range[0][:4])
#         start_month = int(args.selected_time_range[0][4:6])
#         start = datetime(start_year, start_month, 1)
#         start = start - timedelta(hours=args.tranditional_model_leadtime) # 2022 12 31 20

#         end_year = int(args.selected_time_range[-1][:4])
#         end_month = int(args.selected_time_range[-1][4:6]) 
#         end = datetime(end_year, end_month, 1) + relativedelta(months=1) # 2023 04 01
#         end = end - timedelta(hours=args.tranditional_model_leadtime) # 2023 03 31 20

#     elif args.dataset_time_type == "use_year":
#         start = datetime(int(args.selected_time_range[0]), 1, 1)
#         start = start - timedelta(hours=args.tranditional_model_leadtime) # 2020 12 31 20

#         end = datetime(int(args.selected_time_range[-1]) + 1, 1, 1)
#         end = end - timedelta(hours=args.tranditional_model_leadtime) # 2023 12 31 20
#     else : 
#         raise f"args.dataset_time_type = {args.dataset_time_type} error"
        

#     return start, end


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



class CWA(Dataset): # 1/4, 1/9, 1/16
    def __init__(self, data_path, leadtime, step, dataset_time_range, dev=False, dataset_time_type="use_year", patch_size=4, whether_regrid=True, divergence_mode="none"):

        self.G0 = 9.80665  # m/s²
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

        # start = start - (self.step + 1) * self.leadtime
        # end = end - (self.step + 1) * self.leadtime

        print(f"start = {start}, end = {end}")
        
        current = start # 2022 12 31 20 # 2020 12 31 20
        self.folder_list = []

        while current < end:
            folder_name = current.strftime("%Y-%m-%d_%H")
            # print(f"folder_name = {folder_name}")


            if current.year <= 2023:
                root = os.path.join(data_path, "rwrf")
            else:
                root = os.path.join(data_path, "rwrf_data")

            folder_path = os.path.join(root, folder_name)

            if os.path.isdir(folder_path):
                self.folder_list.append(folder_name)
            else : 
                print(f"Error : Lack Data {folder_path}")
                # raise Exception(f"Error : Lack Data {folder_path}")

            current += timedelta(hours=1)

            if dev and len(self.folder_list) == 20 : 
                break
        
        self.folder_list.sort()

        self.start_time = datetime.strptime(self.folder_list[0], "%Y-%m-%d_%H")

        

        print(f"In dataset CWA, # hour = {len(self.folder_list)}")
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

        else : 
            pass


        # """
        #     Read Static vars (regrided data)
        # """
        static_ds = xr.open_mfdataset(f"./CWA_Dataset/regrid_static/cwa_static_regrided.nc")        

        self.static_vars = {}
        for varname in ["slt"]:
            arr = static_ds[varname].values[0]

            arr = crop_helper(arr, self.patch_size).astype(np.float32)

            self.static_vars[varname] = torch.tensor(arr)

        lsm_z_ds = xr.open_mfdataset(f"./CWA_Dataset/regrid_static/lsm_z_regrided.nc")        

        self.static_vars["lsm"] = self.load_field(lsm_z_ds["LANDMASK"])
        self.static_vars["z"] = self.load_field(lsm_z_ds["HGT"]) * self.G0


    
    # the first (self.step + 1) * self.leadtime don't have t-1 and t, so they are not data
    def __len__(self):
        return len(self.folder_list) - (self.step + 1) * self.leadtime

    def load_field(self, xr_field):       
        arr = xr_field.values[0]
        return torch.tensor(arr)

    def __getitem__(self, index):

        # ---------------- time steps ----------------
        time_list = [
            self.start_time + timedelta(hours=index + i * self.leadtime)
            for i in range(self.step + 2)
        ]

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

        # need to be here because input and output was seperated by for loop


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


if __name__ == "__main__":
    data_path = "/scratch6"
    # train_dataset = CWA(data_path=data_path, leadtime=6, step=1, dataset_time_range=["2021", "2022", "2023"])
    train_dataset_month = CWA(data_path=data_path, leadtime=6, step=1, dataset_time_range=["202301", "202302", "202303"], dataset_time_type = "use_month", dev=True)
    # print(type(train_dataset[0]))
    print(type(train_dataset_month[0]))
