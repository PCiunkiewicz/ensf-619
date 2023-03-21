# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np

import torch
from torch.utils.data import random_split, DataLoader
from torchmetrics.functional import structural_similarity_index_measure as ssim, peak_signal_noise_ratio as psnr
from lightning.pytorch.utilities.model_summary import ModelSummary

from data import load_data, DeepCascadeDataset
from model import DeepCascade
from utils import nrmse, nrmse2
from transform import DeepCascadeTransform
from graveyard import custom_imshow
from paths import MODEL_PATH


# %%
SIZE = 164

images, masks = load_data(size=SIZE, mode='original')
images = torch.tensor(images, dtype=torch.float32)
images = images.unsqueeze(1)
masks = torch.tensor(masks, dtype=torch.float32)

transform = DeepCascadeTransform(size=SIZE)
dataset = DeepCascadeDataset(images, masks, transform=transform)
train_ds, val_ds = random_split(dataset, [0.8, 0.2])
val_ds.dataset.val = True

checkpoint_path = MODEL_PATH / 'version_2.ckpt'
model = DeepCascade.load_from_checkpoint(checkpoint_path)

# %%
# slice_idx = 0
# kspace, mask, img = val_ds[slice_idx:slice_idx+1]

batch_size = 8
kspace, mask, img = next(iter(DataLoader(val_ds, batch_size=batch_size)))

rec = torch.abs(torch.fft.ifft2(kspace[:,0] + 1j * kspace[:,1]))
pred = model(kspace, mask).detach()

custom_imshow([rec[0], pred[0], img[0]], ['Undersampled', 'Prediction', 'Original'])
print('rec ssim:', ssim(rec.unsqueeze(1), img.unsqueeze(1), data_range=1))
print('rec psnr:', psnr(rec, img, data_range=1))
print('rec nrmse:', nrmse(rec, img))
print('rec nrmse2:', nrmse2(rec, img))
print('pred ssim:', ssim(pred.unsqueeze(1), img.unsqueeze(1), data_range=1))
print('pred psnr:', psnr(pred, img, data_range=1))
print('pred nrmse:', nrmse(pred, img))
print('pred nrmse2:', nrmse2(pred, img))


# %%
ModelSummary(model, max_depth=-1)
