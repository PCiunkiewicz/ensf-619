"""
Exploration of the data and the model using IPython.
"""
# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import ks_2samp

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

images, masks = load_data(size=SIZE, mode='newborn')
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
msks = []
pred = []
for kspace, mask, img in tqdm(DataLoader(dataset, batch_size=8)):
    rec.append(torch.abs(torch.fft.ifft2(kspace[:,0] + 1j * kspace[:,1])))
    imgs.append(img)
    msks.append(mask)
    pred.append(model[10](kspace, mask).detach())

rec = torch.cat(rec)
img = torch.cat(imgs)
msks = torch.cat(msks)
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
show_slice = 150
custom_imshow([img[show_slice], np.fft.fftshift(1 - msks[show_slice]), rec[show_slice]], ['Original', 'Sampling Mask', 'Undersampled Reconstruction'], filename='sampling.png')


# %%
ModelSummary(model, max_depth=-1)


# %%

dist = {}
for mode in ('original', 'negative', 'newborn'):
    images, masks = load_data(size=SIZE, mode=mode)
    images = torch.tensor(images, dtype=torch.float32)
    images = torch.flatten(torch.mean(images, dim=0))
    dist[mode] = images


# %%
fig, ax = plt.subplots(figsize=(7, 5))
for mode, images in dist.items():
    if mode in ('newborn', 'original'):
        _ = ax.hist(images, bins=200, alpha=0.5, label=mode, density=True)
    if mode == 'negative':
        _ = ax.hist((images - 0.032), bins=200, alpha=0.5, label='negative (shifted)', density=True)

_ = ax.set(
    xlim=(0, 0.25),
    ylim=(0, 25),
    xlabel='Mean Pixel Intensity',
    ylabel='Normalized Count',
)

ax.legend()
plt.savefig('histogram.png', dpi=300, bbox_inches='tight', pad_inches=0)


# %%
print('original vs negative:', ks_2samp(dist['original'], dist['negative']))
print('original vs newborn:', ks_2samp(dist['original'], dist['newborn']))
print('negative vs newborn:', ks_2samp(dist['negative'], dist['newborn']))
print('shifted vs newborn:', ks_2samp((dist['negative'] - 0.032), dist['newborn']))


# %%
original, masks = load_data(size=SIZE, mode='original')
negative, masks = load_data(size=SIZE, mode='negative')
newborn, masks = load_data(size=SIZE, mode='newborn')


# %%
custom_imshow([original[150], negative[150], newborn[1150]], ['Original', 'Negative', 'Newborn'], filename='mr-samples.png')

# %%
def custom_imshow_2(imgs, titles=None, figsize=(10, 10), cmap='gray', origin='lower', filename=None):
    """
    Custom imshow function for displaying multiple images.
    """
    _, axes = plt.subplots(2, len(imgs)//2, figsize=figsize)
    for i, img in enumerate(imgs):
        axes.flatten()[i].imshow(img.T, cmap=cmap, origin=origin)
        axes.flatten()[i].set_axis_off()
        if titles is not None:
            axes.flatten()[i].set(title=titles[i])

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


img = torch.Tensor(original[150]).unsqueeze(0)
custom_imshow_2([img, transform(img), transform(img), transform(img)], ['Original', 'Sample Tranform 1', 'Sample Tranform 2', 'Sample Tranform 3'], filename='transform.png')

# %%
