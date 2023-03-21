import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch import optim
from torchmetrics.functional import structural_similarity_index_measure as ssim, peak_signal_noise_ratio as psnr

from utils import nrmse


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
    def __init__(self, kspace=False):
        super().__init__()
        self.kspace = kspace

    def forward(self, x, inputs, mask):
        if not self.kspace:
            x = FFTBlock(mode='fft')(x)

        x = torch.mul(x, mask.unsqueeze(1))
        return torch.add(x, inputs)


class AbsBlock(nn.Module):
    def forward(self, x):
        return torch.sqrt(x[:,0,:,:]**2 + x[:,1,:,:]**2)


class DeepCascade(pl.LightningModule):
    def __init__(
        self,
        n_channels=2,
        depth_str='ikikii',
        depth=5,
        kernel_size=3,
        nf=48,
        lr=1e-3,
        weight_decay=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cnns = nn.ModuleList([CNNBLock(n_channels, nf, depth, kernel_size) for _ in depth_str])

    def forward(self, x, mask):
        inputs = x.detach().clone()
        kspace_flag = True
        for i, domain in enumerate(self.hparams.depth_str):
            if domain == 'i':
                x = FFTBlock(mode='ifft')(x)
                kspace_flag = False

            x = self.cnns[i](x)
            x = DCBlock(kspace_flag)(x, inputs, mask)
            kspace_flag = True
        x = FFTBlock(mode='ifft')(x)
        return AbsBlock()(x)

    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self.forward(x, mask)
        loss = F.mse_loss(y_hat, y)

        self.log('train_loss', loss)
        self.log('train_nrmse', nrmse(y_hat, y))
        self.log('train_ssim', ssim(y_hat.unsqueeze(1), y.unsqueeze(1)))
        self.log('train_psnr', psnr(y_hat, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self.forward(x, mask)
        loss = F.mse_loss(y_hat, y)

        self.log('val_loss', loss)
        self.log('val_nrmse', nrmse(y_hat, y))
        self.log('val_ssim', ssim(y_hat.unsqueeze(1), y.unsqueeze(1)))
        self.log('val_psnr', psnr(y_hat, y))
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
