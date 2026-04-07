"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
from typing import Generator

import torch

from aurora.batch import Batch
from aurora.model.aurora import Aurora

__all__ = ["rollout"]


def rollout(model: Aurora, batch: Batch, steps: int, leadtime: int = 6, batch_idx = 0) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions.

    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.

    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    # We will need to concatenate data, so ensure that everything is already of the right form.
    # Use an arbitary parameter of the model to derive the data type and device.
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    for step in range(steps):
        this_leadtime = (step + 1) * leadtime
        pred, pred_low, vq_loss, vq_stats, entropy_loss, valid_count = model.forward(batch, leadtime=this_leadtime, batch_idx=batch_idx) ### really don't need to split output?

        yield pred, pred_low, valid_count  # Contains 'yield', so it's a generator function.
        # Return pred to caller first, and move pred to cpu, then return to execute the following codes

        # Add the appropriate history so the model can be run on the prediction.
        
        # Each surface variable (k) has an updated tensor:
		# Historical data shifted by one time step (oldest removed).
	    # Latest prediction (v) appended as the newest time step.
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )
        

