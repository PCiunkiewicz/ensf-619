# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler


from transform import minmax
from data import load_data


# %%
data, targets, mask = load_data(164, mode='negative')

tensor_x = torch.Tensor(list(data)) # transform to torch tensor
tensor_y = torch.Tensor(list(targets))
mask = torch.Tensor(mask)

my_dataset = TensorDataset(tensor_x, tensor_y) # create your datset
my_dataloader = DataLoader(my_dataset) # create your dataloader


# %%
def custom_imshow(imgs, titles=None, figsize=(10, 10), cmap='gray', origin='lower'):
    fig, axes = plt.subplots(1, len(imgs), figsize=figsize)
    for i, img in enumerate(imgs):
        axes[i].imshow(img.T, cmap=cmap, origin=origin)
        if titles is not None:
            axes[i].set(title=titles[i])
    plt.show()


custom_imshow([targets[150], mask, *data[150]], ['Target', 'Mask', 'Subsampled Real', 'Subsampled Imag'])


# %%
def cnn_block(in_channels, nf, depth=3, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, nf, kernel_size, padding='same'),
        nn.LeakyReLU(0.1, inplace=True)
        *[
            nn.Conv2d(nf, nf, kernel_size, padding='same'),
            nn.LeakyReLU(0.1, inplace=True)
        ] * depth,
        nn.Conv2d(nf, in_channels, 1)
    )


class DCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=3, kernel_size=3):
        super().__init__()


class DeepCascade(pl.LightningModule):
    def __init__(self, n_channels=2, depth_str='ikikii', depth=5, kernel_size=3, nf=48):
        super().__init__()
        self.save_hyperparameters()

        self.cnns = [cnn_block(n_channels, nf)]


# %%
from sampling import gaussian2d
from paths import DATA_PATH
import numpy as np

size = 100
shape = (size, size)
mask = gaussian2d(shape, 0.2, radius=18, seed=42)
np.save(DATA_PATH / f'sampling/mask_{size}.npy', mask)


# %%
