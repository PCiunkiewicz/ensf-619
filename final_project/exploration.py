"""
Exploration of the data and the model using IPython.
"""
# %%
import os

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import random_split, DataLoader
from torchmetrics.functional import structural_similarity_index_measure as ssim, peak_signal_noise_ratio as psnr
from lightning.pytorch.utilities.model_summary import ModelSummary

from data import load_data, DeepCascadeDataset
from model import DeepCascade
from utils import nrmse, nrmse2, custom_imshow
from transform import DeepCascadeTransform
from paths import MODEL_PATH


# %%
SIZE = 164

images, masks = load_data(size=SIZE, mode='original')
images = torch.tensor(images, dtype=torch.float32)
images = images.unsqueeze(1)
masks = torch.tensor(masks, dtype=torch.float32)

transform = DeepCascadeTransform(size=SIZE)
dataset = DeepCascadeDataset(images, masks, transform=transform)
dataset.val = True
# train_ds, val_ds = random_split(dataset, [0.8, 0.2])


# %%
model = {}
for version in (2, 7, 8, 9, 10):
    checkpoint_path = MODEL_PATH / f'version_{version}.ckpt'
    model[version] = DeepCascade.load_from_checkpoint(checkpoint_path).eval()


# %%
rec = []
imgs = []
pred = []
for kspace, mask, img in tqdm(DataLoader(dataset, batch_size=8)):
    rec.append(torch.abs(torch.fft.ifft2(kspace[:,0] + 1j * kspace[:,1])))
    imgs.append(img)
    pred.append(model[10](kspace, mask).detach())

rec = torch.cat(rec)
img = torch.cat(imgs)
pred = torch.cat(pred)


# %%
print('rec ssim:', ssim(rec.unsqueeze(1), img.unsqueeze(1), data_range=1))
print('rec psnr:', psnr(rec, img, data_range=1))
print('rec nrmse:', nrmse(rec, img))
print('rec nrmse2:', nrmse2(rec, img))
print('pred ssim:', ssim(pred.unsqueeze(1), img.unsqueeze(1), data_range=1))
print('pred psnr:', psnr(pred, img, data_range=1))
print('pred nrmse:', nrmse(pred, img))
print('pred nrmse2:', nrmse2(pred, img))


# %%
show_slice = 1150
custom_imshow([rec[show_slice], pred[show_slice], img[show_slice]], ['Undersampled', 'Prediction', 'Original'])


# %%
ModelSummary(model, max_depth=-1)


# %%
fig, ax = plt.subplots(figsize=(12, 12))

for mode in ('original', 'negative', 'newborn'):
    images, masks = load_data(size=SIZE, mode=mode)
    images = torch.tensor(images, dtype=torch.float32)
    images = torch.flatten(torch.mean(images, dim=0))
    _ = ax.hist(images, bins=100, alpha=0.5, label=mode, density=True)

ax.legend()
# %%
