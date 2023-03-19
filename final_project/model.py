import os
import logging
import urllib
from urllib.error import HTTPError

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM, PeakSignalNoiseRatio as PSNR
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM, PeakSignalNoiseRatio as PSNR

from paths import MODEL_PATH, PRETRAINED_PATH


LOG = logging.getLogger('main')
LOG.setLevel(logging.INFO)

NUM_WORKERS = os.cpu_count()

# Try to use GPU (either metal or cuda api), fall back to CPU
if torch.backends.mps.is_available():
    torch.backends.mps.determinstic = True
    torch.backends.mps.benchmark = False
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

LOG.info(f'Hardware accelerator: {DEVICE}')
LOG.info(f'Number of cpu cores: {NUM_WORKERS}')


class CNNBLock(nn.Module):
    def __init__(self, in_channels, nf, depth=3, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, nf, kernel_size, padding='same'),
            nn.LeakyReLU(0.1, inplace=True),
            *[
                op for _ in range(depth - 1) for op in (
                    nn.Conv2d(nf, nf, kernel_size, padding='same'),
                    nn.LeakyReLU(0.1, inplace=True)
                )
            ],
            nn.Conv2d(nf, in_channels, 1)
        )

    def forward(self, x):
        conv = self.conv(x)
        return torch.add(x, conv)


class FFTBlock(nn.Module):
    def __init__(self, mode='fft'):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        real = x[:, 0, :, :]
        imag = x[:, 1, :, :]
        x_complex = torch.complex(real, imag)
        if self.mode == 'fft':
            x_complex = torch.fft.fft2(x_complex)
        elif self.mode == 'ifft':
            x_complex = torch.fft.ifft2(x_complex)

        real = torch.unsqueeze(x_complex.real, 1)
        imag = torch.unsqueeze(x_complex.imag, 1)
        return torch.cat([real, imag], dim=1)


class DCBlock(nn.Module):
    def __init__(self, mask, kspace=False):
        super().__init__()
        self.mask = torch.tensor(mask, device=DEVICE)
        self.kspace = kspace

    def forward(self, x, inputs):
        if not self.kspace:
            x = FFTBlock(mode='fft')(x)

        x = torch.mul(x, self.mask)
        return torch.add(x, inputs)


def nrmse(preds, target):
    return torch.sqrt(torch.mean((preds - target)**2)) / torch.sqrt(torch.mean(target**2))


class DeepCascade(pl.LightningModule):
    def __init__(
        self,
        mask,
        n_channels=2,
        depth_str='ikikii',
        depth=5,
        kernel_size=3,
        nf=48
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cnns = nn.ModuleList([CNNBLock(n_channels, nf, depth, kernel_size) for _ in depth_str])

    def forward(self, x):
        inputs = x.detach().clone()
        kspace_flag = True
        for i, domain in enumerate(self.hparams.depth_str):
            if domain == 'i':
                x = FFTBlock(mode='ifft')(x)
                kspace_flag = False

            x = self.cnns[i](x)
            x = DCBlock(self.hparams.mask, kspace_flag)(x, inputs)
            kspace_flag = True
        return FFTBlock(mode='ifft')(x)[:,0,:,:] # Only return real part

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)

        ssim = SSIM().to(DEVICE)
        psnr = PSNR().to(DEVICE)

        self.log('train_loss', loss)
        self.log('train_nrmse', nrmse(y_hat, y))
        self.log('train_ssim', ssim(y_hat.unsqueeze(1), y.unsqueeze(1)))
        self.log('train_psnr', psnr(y_hat, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)

        ssim = SSIM().to(DEVICE)
        psnr = PSNR().to(DEVICE)

        self.log('val_loss', loss)
        self.log('val_nrmse', nrmse(y_hat, y))
        self.log('val_ssim', ssim(y_hat.unsqueeze(1), y.unsqueeze(1)))
        self.log('val_psnr', psnr(y_hat, y))
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)


class DeepCascadeTrainer(pl.Trainer):
    def __init__(self, batch_size, max_epochs, download=False, **kwargs):
        super().__init__(
            default_root_dir=MODEL_PATH / 'DeepCascade',
            accelerator=str(DEVICE) if str(DEVICE) in {'mps', 'cpu'} else 'auto',
            max_epochs=max_epochs,
            callbacks=[
                ModelCheckpoint(save_weights_only=True, mode='min', monitor='val_loss'),
                LearningRateMonitor('epoch')
            ]
        )

        self.batch_size = batch_size
        self.is_trained = False
        # Attempt to load pre-trained model
        self.model = self.load_pretrained(download)
        # Fall back to new DeepCascade instance if self.model is None
        if not self.model:
            self.model = DeepCascade(**kwargs)

    def load_pretrained(self, download=False):
        ckpt_path = PRETRAINED_PATH / 'DeepCascade.ckpt'
        # Download pre-trained model from Github
        if download:
            self.download_pretrained(ckpt_path)

        # Load pre-trained model from disk if file exists
        if os.path.isfile(ckpt_path):
            self.is_trained = True
            LOG.info(f'Loading pre-trained model from path: {ckpt_path}')
            return DeepCascade.load_from_checkpoint(ckpt_path)

    def download_pretrained(self, ckpt_path):
        # Pre-trained model checkpoint from my Github repo
        pretrained_url = 'https://raw.githubusercontent.com/pciunkiewicz/ensf-619/master/final_project/models/pretrained/DeepCascade.ckpt'
        os.makedirs('models', exist_ok=True)

        # Only download the file if not found on disk; no overwrite
        if not os.path.isfile(ckpt_path):
            try:
                LOG.info(f'Downloading pre-trained model from {pretrained_url}')
                urllib.request.urlretrieve(pretrained_url, ckpt_path)
            except HTTPError as exception:
                LOG.error(exception)

    def fit(self, train, val):
        # Check if pre-trained model has been loaded
        if not self.is_trained:
            # Create PyTorch DataLoaders for training and validation sets
            train_loader = DataLoader(
                train,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=NUM_WORKERS,
            )
            val_loader = DataLoader(
                val,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=NUM_WORKERS,
            )

            super().fit(self.model, train_loader, val_loader)
            self.model = DeepCascade.load_from_checkpoint(self.checkpoint_callback.best_model_path)
        else:
            LOG.info('Model is already trained; returning pre-trained model.')
        return self.model
