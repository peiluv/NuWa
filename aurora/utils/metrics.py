import numpy as np
import torch
from scipy import stats
from aurora.batch import Batch
import sys

def mae(pred: Batch, y: Batch, vars):
    for k in vars["surf_vars"]:
        if torch.isnan(pred.surf_vars[k]).any() or torch.isinf(pred.surf_vars[k]).any():
            print(f"pred {k} NaN/Inf")

    for k in vars["atmos_vars"]:
        if torch.isnan(pred.atmos_vars[k]).any() or torch.isinf(pred.atmos_vars[k]).any():
            print(f"pred {k} NaN/Inf")

    loss_dict = {}
    with torch.no_grad():
        loss_dict={
            'surf_vars' : {k: torch.sum(torch.mean(abs(pred.surf_vars[k].to(torch.float64)-y.surf_vars[k].to(torch.float64)), dim = (-2,-1)),dim=(0,1)) for k in vars["surf_vars"]}, # 1
            'atmos_vars': {k: torch.sum(torch.mean(abs(pred.atmos_vars[k].to(torch.float64)-y.atmos_vars[k].to(torch.float64)), dim = (-2,-1)),dim=(0,1)) for k in vars["atmos_vars"]}, # 13
        }
    final_loss_dict = {'my_mae': loss_dict}

    return final_loss_dict

def rmse(pred: Batch, y: Batch, vars):
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
        }
    final_loss_dict = {'my_rmse': loss_dict}

    return final_loss_dict


def lat_weighted_rmse(pred: Batch, y: Batch, vars: dict, lat: torch.Tensor):
    # w(i) = cos(lat(i)) / (1/H * sum_H i' cos(lat(i')))
    w_lat = torch.cos(torch.deg2rad(lat.to(torch.float32)))
    w_lat = w_lat / w_lat.mean()
    w_lat = w_lat.to(device=lat.device)

    loss_dict = {}
    with torch.no_grad():
        # Surface Variables
        surf_losses = {}
        for k in vars["surf_vars"]:
            pred_tensor = pred.surf_vars[k].to(torch.float64)
            y_tensor = y.surf_vars[k].to(torch.float64)
            w_lat_surf = w_lat.reshape(1, 1, pred_tensor.shape[-2], 1).to(pred_tensor.device)
            weighted_mse = torch.mean(((pred_tensor - y_tensor)**2) * w_lat_surf, dim=(-2, -1))
            rmse = torch.sqrt(weighted_mse)
            surf_losses[k] = torch.mean(rmse)

        # Atmospheric Variables
        atmos_losses = {}
        for k in vars["atmos_vars"]:
            pred_tensor = pred.atmos_vars[k].to(torch.float64)
            y_tensor = y.atmos_vars[k].to(torch.float64)
            w_lat_atmos = w_lat.reshape(1, 1, 1, pred_tensor.shape[-2], 1).to(pred_tensor.device)
            weighted_mse = torch.mean(((pred_tensor - y_tensor)**2) * w_lat_atmos, dim=(-2, -1))
            rmse = torch.sqrt(weighted_mse)
            atmos_losses[k] = torch.mean(rmse, dim=(0, 1))

        loss_dict = {
            'surf_vars': surf_losses,
            'atmos_vars': atmos_losses,
        }

    final_loss_dict = {'lat_weighted_rmse': loss_dict}
    return final_loss_dict


def global_cropped_rmse(pred: Batch, y: Batch, vars):
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

    loss_dict["mean_bias"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict
