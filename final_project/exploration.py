# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np

import torch
from torch.utils.data import TensorDataset, random_split
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM, PeakSignalNoiseRatio as PSNR

from data import load_data
from model import DeepCascade, DeepCascadeTrainer


# %%
data, targets, mask = load_data(size=164, mode='original')
data = torch.tensor(data)
targets = torch.tensor(targets)

dataset = TensorDataset(data, targets)
train_ds, val_ds = random_split(dataset, [0.8, 0.2])


# %%
trainer = DeepCascadeTrainer(
    batch_size=6,
    max_epochs=2,
    download=False,
    mask=mask
)
model = trainer.fit(train_ds, val_ds)


# %%
preds = data[0, 0].unsqueeze(0).unsqueeze(0)
target = targets[0].unsqueeze(0).unsqueeze(0)

ssim = SSIM()
ssim(preds, target)

psnr = PSNR()
psnr(preds, target)


# %%
# %%
from pytorch_lightning.utilities.model_summary import ModelSummary
model = DeepCascade(mask=mask)
ModelSummary(model, max_depth=-1)


# %%
