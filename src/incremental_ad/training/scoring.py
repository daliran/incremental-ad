import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from incremental_ad.models.base_model import BaseModel

log = logging.getLogger(__name__)


def collect_scores(
    model: BaseModel,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Scoring",
) -> np.ndarray:
    model.eval()

    # This is a list of lists. Since batch size > 1, each sub list item is the score/reconstruction error of an individual batch/window.
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = batch.to(device)
            scores = model.eval_step(batch)

            # Add the scores to the list of lists
            all_scores.append(scores.cpu().numpy())

    # Flatten the list of lists to create a unique list of scores, removing the grouping from batch size.
    return np.concatenate(all_scores, axis=0)

