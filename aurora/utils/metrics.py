import numpy as np
import torch
from scipy import stats
from aurora.batch import Batch
import sys

# Smooth RMSE
def my_smooth_rmse_val(pred: Batch, y: Batch, vars, i: int, eps: float = 1e-6):
    with torch.no_grad():
        loss_dict = {
            'surf_vars': {
                k: torch.sum(torch.sqrt(
                    torch.mean((pred.surf_vars[k] - y.surf_vars[k]) ** 2, dim=(-2, -1)) + eps
                ))
                for k in vars["surf_vars"]
            },
            'atmos_vars': {
                k: torch.sum(torch.sqrt(
                    torch.mean((pred.atmos_vars[k] - y.atmos_vars[k]) ** 2, dim=(-2, -1)) + eps
                ), dim=(0, 1))
                for k in vars["atmos_vars"]
            },
        }

    return {f'my_rmse_{i+1}': loss_dict}

def my_mae_val(pred: Batch, y: Batch, vars):
    # print(pred)
    for k in vars["surf_vars"]:
        # print(f"pred.surf_vars[{k}].shape = {pred.surf_vars[k].shape}")
        if torch.isnan(pred.surf_vars[k]).any() or torch.isinf(pred.surf_vars[k]).any():
            print(f"pred {k} NaN/Inf")

    for k in vars["atmos_vars"]:
        # print(f"pred.atmos_vars[{k}].shape = {pred.atmos_vars[k].shape}")
        if torch.isnan(pred.atmos_vars[k]).any() or torch.isinf(pred.atmos_vars[k]).any():
            print(f"pred {k} NaN/Inf")

    loss_dict = {}
    with torch.no_grad():
        loss_dict={
            'surf_vars' : {k: torch.sum(torch.mean(abs(pred.surf_vars[k].to(torch.float64)-y.surf_vars[k].to(torch.float64)), dim = (-2,-1)),dim=(0,1)) for k in vars["surf_vars"]}, # 1
            'atmos_vars': {k: torch.sum(torch.mean(abs(pred.atmos_vars[k].to(torch.float64)-y.atmos_vars[k].to(torch.float64)), dim = (-2,-1)),dim=(0,1)) for k in vars["atmos_vars"]}, # 13
        }
        # print(loss_dict)
    final_loss_dict = {'my_mae': loss_dict}

    return final_loss_dict

def my_rmse_val(pred: Batch, y: Batch, vars):
    # shape == (b, t, h, w)
    for k in vars["surf_vars"]:
        if torch.isnan(pred.surf_vars[k]).any() or torch.isinf(pred.surf_vars[k]).any():
            print(f"pred {k} NaN/Inf")

    # shape == (b, t, c, h, w)
    for k in vars["atmos_vars"]:
        if torch.isnan(pred.atmos_vars[k]).any() or torch.isinf(pred.atmos_vars[k]).any():
            print(f"pred {k} NaN/Inf")

    loss_dict = {}
    with torch.no_grad():
        loss_dict={
            'surf_vars' : {k: torch.sum(torch.sqrt(torch.mean((pred.surf_vars[k].to(torch.float64)-y.surf_vars[k].to(torch.float64))** 2, dim = (-2,-1)))) for k in vars["surf_vars"]},
            'atmos_vars': {k: torch.sum(torch.sqrt(torch.mean((pred.atmos_vars[k].to(torch.float64)-y.atmos_vars[k].to(torch.float64))** 2, dim = (-2,-1))), dim=(0,1)) for k in vars["atmos_vars"]},
            #============= for global dataset cropped to reginal ===============================
            # 'surf_vars' : {k: torch.sum(torch.sqrt(torch.mean((pred.surf_vars[k][:,:,200:340,400:580].to(torch.float64)-y.surf_vars[k][:,:,200:340,400:580].to(torch.float64))** 2, dim = (-2,-1)))) for k in vars["surf_vars"]},
            # 'atmos_vars': {k: torch.sum(torch.sqrt(torch.mean((pred.atmos_vars[k][:,:,:,200:340,400:580].to(torch.float64)-y.atmos_vars[k][:,:,:,200:340,400:580].to(torch.float64))** 2, dim = (-2,-1))), dim=(0,1)) for k in vars["atmos_vars"]},
        }
    final_loss_dict = {'my_rmse': loss_dict}

    return final_loss_dict


def lat_weighted_rmse(pred: Batch, y: Batch, vars: dict, lat: torch.Tensor):
    # w(i) = cos(lat(i)) / (1/H * sum_H i' cos(lat(i')))
    w_lat = torch.cos(torch.deg2rad(lat.to(torch.float32)))
    w_lat = w_lat / w_lat.mean() # 正規化確保權重平均為1
    w_lat = w_lat.to(device=lat.device)

    loss_dict = {}
    with torch.no_grad():
        # Surface Variables
        surf_losses = {}
        for k in vars["surf_vars"]:
            pred_tensor = pred.surf_vars[k].to(torch.float64)
            y_tensor = y.surf_vars[k].to(torch.float64)
            # 重塑 w_lat 以匹配 (B, T, H, W) 的空間維度
            w_lat_surf = w_lat.reshape(1, 1, pred_tensor.shape[-2], 1).to(pred_tensor.device)

            # 計算加權均方誤差，並在高度(H)和寬度(W)維度上取平均
            # 結果為 (B, T) 形狀的張量
            weighted_mse = torch.mean(((pred_tensor - y_tensor)**2) * w_lat_surf, dim=(-2, -1))
            rmse = torch.sqrt(weighted_mse) # 得到每個批次樣本、每個時間步的空間 RMSE

            # 將加總改為平均，Batch 和 Time 維度取平均，得到該變數的單一 RMSE 值
            surf_losses[k] = torch.mean(rmse)

        # Atmospheric Variables
        atmos_losses = {}
        for k in vars["atmos_vars"]:
            pred_tensor = pred.atmos_vars[k].to(torch.float64)
            y_tensor = y.atmos_vars[k].to(torch.float64)
            # 重塑 w_lat 以匹配 (B, T, C, H, W) 的空間維度
            w_lat_atmos = w_lat.reshape(1, 1, 1, pred_tensor.shape[-2], 1).to(pred_tensor.device)

            # 計算加權均方誤差，並在高度(H)和寬度(W)維度上取平均
            # 結果為 (B, T, C) 形狀的張量
            weighted_mse = torch.mean(((pred_tensor - y_tensor)**2) * w_lat_atmos, dim=(-2, -1))
            rmse = torch.sqrt(weighted_mse) # 得到每個批次樣本、每個時間步、每個壓力層級的空間RMSE

            # 將加總改為平均，並保留壓力層級(C)維度，對 Batch 和時間 Time 維度取平均，得到每個壓力層級的 RMSE 值
            atmos_losses[k] = torch.mean(rmse, dim=(0, 1))

        loss_dict = {
            'surf_vars': surf_losses,
            'atmos_vars': atmos_losses,
        }

    final_loss_dict = {'lat_weighted_rmse': loss_dict}
    return final_loss_dict


def global_cropped_rmse(pred: Batch, y: Batch, vars):
    # print("in my rmse val")
    # print(vars)
    # print(pred.surf_vars["2t"].shape)
    # print(pred.atmos_vars["q"].shape)
    # print("pred is ok")
    # print(y.surf_vars["2t"].shape)
    # print(y.atmos_vars)
    # print("y is ok")
    # print(pred.surf_vars["2t"].shape)
    # print(torch.sum(torch.sqrt(torch.mean((pred.surf_vars["2t"]-y.surf_vars["2t"])** 2, dim = (-2,-1)))))
    # print("====================")
    loss_dict = {}
    with torch.no_grad():
        loss_dict={
            'surf_vars' : {k: torch.sum(torch.sqrt(torch.mean((pred.surf_vars[k].to(torch.float64)-y.surf_vars[k].to(torch.float64))** 2, dim = (-2,-1)))) for k in vars["surf_vars"]},
            'atmos_vars': {k: torch.sum(torch.sqrt(torch.mean((pred.atmos_vars[k].to(torch.float64)-y.atmos_vars[k].to(torch.float64))** 2, dim = (-2,-1))), dim=(0,1)) for k in vars["atmos_vars"]},
        }
    final_loss_dict = {'my_rmse': loss_dict}

    return final_loss_dict

def rmse_val(pred, y, transform, vars, lat, clim, log_postfix):
    '''
    root mean squared error.
    Args:
        y: [B, V, H, W].
        pred: [B, V, H, W].
        vars: list of variable names.
        lat: H.
    '''

    pred = transform(pred)
    y = transform(y)

    # [B, V, H, W].
    error = (pred - y) ** 2

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f'rmse_{var}_{log_postfix}'] = torch.mean(
                torch.sqrt(torch.mean(error[:, i], dim = (-2, -1)))
            )

    loss_dict['rmse'] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def mse(pred, y, vars, lat=None, mask=None):
    """Mean squared error

    Args:
        pred: [B, V, H, W]
        y: [B, V, H, W]
        vars: list of variable names
    """

    loss = (pred - y) ** 2

    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (loss[:, i] * mask).sum() / mask.sum()
            else:
                loss_dict[var] = loss[:, i].mean()

    if mask is not None:
        loss_dict["loss"] = (loss.mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = loss.mean(dim=1).mean()

    return loss_dict


def lat_weighted_mse(pred, y, vars, lat, mask=None):
    """Latitude weighted mean squared error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [N, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean()

    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()

    return loss_dict


def lat_weighted_mse_val(pred, y, transform, vars, lat, clim, log_postfix):
    """Latitude weighted mean squared error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()

    loss_dict["w_mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_acc(pred: Batch, y: Batch, climatology: Batch, vars_config: dict, lat: torch.Tensor) -> dict:
    """
    Args:
        pred (:class:`Batch`): 模型預測的 Batch 物件，包含 surf_vars 和 atmos_vars。
        y (:class:`Batch`): 真實值的 Batch 物件，包含 surf_vars 和 atmos_vars。
        climatology (:class:`Batch`): 氣候值的 Batch 物件，包含 surf_vars 和 atmos_vars。
                                      期望其空間維度與 pred/y 匹配，時間和批次維度可廣播。
                                      若氣候值隨時間變化，應具備與預測/真實值相同的時間/批次維度。
        vars_config (dict): 包含 'surf_vars' 和 'atmos_vars' 鍵的字典，
                            其值為對應變數名稱的列表 (e.g., {"surf_vars": ["2t", "10u"], ...})。
        lat (:class:`torch.Tensor`): 緯度張量 (一維)，用於緯度加權。

    Returns:
        dict: 包含各變數 ACC 值的字典。對於大氣變數，會為每個壓力層級計算 ACC。
    """
    device = lat.device
    pred = pred.to(device).type(torch.float64)
    y = y.to(device).type(torch.float64)
    climatology = climatology.to(device).type(torch.float64)

    # 計算緯度權重 w(i) = cos(lat(i)) / (1/H * sum_H i' cos(lat(i')))
    # 緯度需從度數轉換為弧度
    w_lat = torch.cos(torch.deg2rad(lat.to(torch.float32)))
    # 正規化確保權重平均為1
    w_lat = w_lat / w_lat.mean()
    w_lat = w_lat.to(device=device)

    acc_scores = {} # 用於儲存最終 ACC 值的字典

    with torch.no_grad(): # 評估指標計算應在無梯度環境中執行，以節省資源。
        # 處理表面變數 (surf_vars) 和大氣變數 (atmos_vars)
        for var_type, var_names in vars_config.items():
            for k in var_names:
                pred_tensor = getattr(pred, var_type)[k]
                y_tensor = getattr(y, var_type)[k]
                climatology_tensor = getattr(climatology, var_type)[k]

                # 確定權重張量的重塑形狀，以匹配預測張量的空間維度
                # H_idx 和 W_idx 分別指向緯度(高度)和經度(寬度)的維度索引
                if var_type == "surf_vars":
                    # 表面變數形狀預期為 (B, T, H, W)。
                    H_idx = -2
                    W_idx = -1
                    w_lat_expanded = w_lat.reshape(1, 1, pred_tensor.shape[H_idx], 1)
                    # 氣候值張量需要能廣播到與預測/真實值相同的形狀
                    if climatology_tensor.ndim == 2: # (H, W)
                        climatology_tensor = climatology_tensor.unsqueeze(0).unsqueeze(0).expand_as(pred_tensor)
                    elif climatology_tensor.ndim == 4: # 已是 (B, T, H, W)
                        pass
                    else:
                        raise ValueError(
                            f"表面變數 '{k}' 的氣候值張量形狀不符合預期: {climatology_tensor.shape}。 "
                            f"期望 (H,W) 或 (B,T,H,W) 以便廣播到預測形狀 {pred_tensor.shape}。"
                        )

                elif var_type == "atmos_vars":
                    # 大氣變數形狀預期為 (B, T, C, H, W)
                    H_idx = -2
                    W_idx = -1
                    w_lat_expanded = w_lat.reshape(1, 1, 1, pred_tensor.shape[H_idx], 1)
                    # 氣候值張量需要能廣播到與預測/真實值相同的形狀
                    # 如果氣候值僅為空間和層級數據 (C, H, W)，則擴展其批次和時間維度
                    if climatology_tensor.ndim == 3: # (C, H, W)
                        climatology_tensor = climatology_tensor.unsqueeze(0).unsqueeze(0).expand_as(pred_tensor)
                    elif climatology_tensor.ndim == 5: # 已是 (B, T, C, H, W)
                        pass
                    else:
                        raise ValueError(
                            f"大氣變數 '{k}' 的氣候值張量形狀不符合預期: {climatology_tensor.shape}。 "
                            f"期望 (C,H,W) 或 (B,T,C,H,W) 以便廣播到預測形狀 {pred_tensor.shape}。"
                        )
                w_lat_expanded = w_lat_expanded.to(pred_tensor.device)

                # 計算預測值與真實值相對於氣候值的偏差 (Anomaly)
                pred_anomaly = pred_tensor - climatology_tensor
                y_anomaly = y_tensor - climatology_tensor

                # 計算分子：加權協方差之和。根據公式 F16 的分子部分
                # sum_{i,j} w(i)(X̂t_i,j - Ct_i,j)(Xt_i,j - Ct_i,j)
                numerator_spatial_sum = torch.sum(w_lat_expanded * pred_anomaly * y_anomaly, dim=(H_idx, W_idx))

                # 計算分母的兩個平方和項：加權變異數之和。根據公式 F16 的分母部分
                # (sum_{i,j} w(i)(X̂t_i,j - Ct_i,j)^2)
                pred_var_spatial_sum = torch.sum(w_lat_expanded * pred_anomaly**2, dim=(H_idx, W_idx))
                # (sum_{i,j} w(i)(Xt_i,j - Ct_i,j)^2)
                y_var_spatial_sum = torch.sum(w_lat_expanded * y_anomaly**2, dim=(H_idx, W_idx))

                # 計算分母項的平方根
                denominator_term = torch.sqrt(pred_var_spatial_sum * y_var_spatial_sum)

                # 處理分母為零的情況，避免除以零產生 NaN
                # 如果分母為零 (表示預測或真實的異常值變異為零)，則將 ACC 設為 0
                # 為了數值穩定性，將分母為零的位置暫時替換為 1 進行除法，之後再修正
                safe_denominator = torch.where(
                    denominator_term == 0,
                    torch.tensor(1.0, device=device, dtype=torch.float64),
                    denominator_term
                )

                # 計算每個時間步/批次的 ACC。結果形狀為 (B, T) 或 (B, T, C)
                acc_per_slice = numerator_spatial_sum / safe_denominator
                # 將原始分母為零的 ACC 值設為 0
                acc_per_slice = torch.where(
                    denominator_term == 0,
                    torch.tensor(0.0, device=device, dtype=torch.float64),
                    acc_per_slice
                )

                # 根據公式 F16，對所有樣本資料集 (Batch 和 Time 維度) 求平均
                # 對於大氣變數  U, V, T, Q, Z
                # 會按壓力層級顯示 ACC，因此保留壓力層級 (C) 維度
                if var_type == "surf_vars":
                    acc_scores[k] = torch.mean(acc_per_slice) # 從 (B, T) 聚合為純量
                elif var_type == "atmos_vars":
                    acc_scores[k] = torch.mean(acc_per_slice, dim=(0, 1)) # 從 (B, T, C) 聚合為 (C) 維度張量

    final_acc_dict = {'acc': acc_scores}
    return final_acc_dict


def lat_weighted_nrmses(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)
    y_normalization = clim

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(-1).to(dtype=y.dtype, device=y.device)  # (H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_ = pred[:, i]  # B, H, W
            y_ = y[:, i]  # B, H, W
            error = (torch.mean(pred_, dim=0) - torch.mean(y_, dim=0)) ** 2  # H, W
            error = torch.mean(error * w_lat)
            loss_dict[f"w_nrmses_{var}"] = torch.sqrt(error) / y_normalization

    return loss_dict


def lat_weighted_nrmseg(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)
    y_normalization = clim

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=y.dtype, device=y.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_ = pred[:, i]  # B, H, W
            pred_ = torch.mean(pred_ * w_lat, dim=(-2, -1))  # B
            y_ = y[:, i]  # B, H, W
            y_ = torch.mean(y_ * w_lat, dim=(-2, -1))  # B
            error = torch.mean((pred_ - y_) ** 2)
            loss_dict[f"w_nrmseg_{var}"] = torch.sqrt(error) / y_normalization

    return loss_dict


def lat_weighted_nrmse(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    nrmses = lat_weighted_nrmses(pred, y, transform, vars, lat, clim, log_postfix)
    nrmseg = lat_weighted_nrmseg(pred, y, transform, vars, lat, clim, log_postfix)
    loss_dict = {}
    for var in vars:
        loss_dict[f"w_nrmses_{var}"] = nrmses[f"w_nrmses_{var}"]
        loss_dict[f"w_nrmseg_{var}"] = nrmseg[f"w_nrmseg_{var}"]
        loss_dict[f"w_nrmse_{var}"] = nrmses[f"w_nrmses_{var}"] + 5 * nrmseg[f"w_nrmseg_{var}"]
    return loss_dict


def remove_nans(pred: torch.Tensor, gt: torch.Tensor):
    # pred and gt are two flattened arrays
    pred_nan_ids = torch.isnan(pred) | torch.isinf(pred)
    pred = pred[~pred_nan_ids]
    gt = gt[~pred_nan_ids]

    gt_nan_ids = torch.isnan(gt) | torch.isinf(gt)
    pred = pred[~gt_nan_ids]
    gt = gt[~gt_nan_ids]

    return pred, gt


def pearson(pred, y, transform, vars, lat, log_steps, log_days, clim):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                pred_, y_ = pred[:, step - 1, i].flatten(), y[:, step - 1, i].flatten()
                pred_, y_ = remove_nans(pred_, y_)
                loss_dict[f"pearsonr_{var}_day_{day}"] = stats.pearsonr(pred_.cpu().numpy(), y_.cpu().numpy())[0]

    loss_dict["pearsonr"] = np.mean([loss_dict[k] for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_mean_bias(pred, y, transform, vars, lat, log_steps, log_days, clim):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                pred_, y_ = pred[:, step - 1, i].flatten(), y[:, step - 1, i].flatten()
                pred_, y_ = remove_nans(pred_, y_)
                loss_dict[f"mean_bias_{var}_day_{day}"] = pred_.mean() - y_.mean()

                # pred_mean = torch.mean(w_lat * pred[:, step - 1, i])
                # y_mean = torch.mean(w_lat * y[:, step - 1, i])
                # loss_dict[f"mean_bias_{var}_day_{day}"] = y_mean - pred_mean

    loss_dict["mean_bias"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict
