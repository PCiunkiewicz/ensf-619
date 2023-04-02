"""
DANN model definition and NN block module.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch import optim
from torch.autograd import Function
from torchmetrics.functional import structural_similarity_index_measure as ssim, peak_signal_noise_ratio as psnr

from paths import MODEL_PATH
from utils import nrmse2


class CNNBLock(nn.Module):
    """
    Convolutional Neural Network Block for Deep Cascade model.
    """
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
            ]
        )
        self.conv2 = nn.Conv2d(nf, in_channels, 1)

    def forward(self, x):
        features = self.conv(x).detach().clone()
        conv = self.conv2(features)
        return torch.add(x, conv), features


class ReconstructionBlock(nn.Module):
    """
    Regressor Block for DANN with partial CNNBlock structure.
    """
    def forward(self, x):
        x = FFTBlock(mode='ifft')(x)
        return AbsBlock()(x)


class DomainClassifierBlock(nn.Module):
    """
    Domain Classifier Block for DANN.
    """
    def __init__(self, nf=48, size=164, pool=4):
        super().__init__()
        self.flat_shape = nf * size * size // pool // pool
        self.pool = pool
        self.conv = nn.Sequential(
            nn.Linear(self.flat_shape, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = nn.MaxPool2d(self.pool)(x)
        x = x.view(-1, self.flat_shape)
        return self.conv(x)


class FFTBlock(nn.Module):
    """
    Fast Fourier Transform Block for Deep Cascade model.
    """
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
    """
    Data Consistency Block for Deep Cascade model.
    """
    def __init__(self, kspace=False):
        super().__init__()
        self.kspace = kspace

    def forward(self, x, inputs, mask):
        if not self.kspace:
            x = FFTBlock(mode='fft')(x)

        x = torch.mul(x, mask.unsqueeze(1))
        return torch.add(x, inputs)


class AbsBlock(nn.Module):
    """
    Absolute magnitude Block for Deep Cascade model.
    """
    def forward(self, x):
        return torch.sqrt(x[:,0,:,:]**2 + x[:,1,:,:]**2)


class FeatureNetwork(pl.LightningModule):
    """
    Feature Network for DANN.
    """
    def __init__(
        self,
        n_channels=2,
        depth_str='ikikii',
        depth=5,
        kernel_size=3,
        nf=48,
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

            x, feature = self.cnns[i](x)
            x = DCBlock(kspace_flag)(x, inputs, mask)
            kspace_flag = True
        return x, feature


class ReverseLayerF(Function):
    """
    Reverse Layer Function for DANN.
    This function is taken from https://github.com/rmsouza01/deep-learning/.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DeepCascadeDANN(pl.LightningModule):
    """
    Deep Cascade DANN model for Compressed Sensing MRI Reconstruction.
    """
    def __init__(
        self,
        n_channels=2,
        depth_str='ikikii',
        depth=5,
        kernel_size=3,
        nf=48,
        img_size=164,
        beta=0.001,
        lr=1e-3,
        weight_decay=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.feature = FeatureNetwork(n_channels, depth_str, depth, kernel_size, nf)
        self.reconstruct = ReconstructionBlock()
        self.domain_classifier = DomainClassifierBlock(nf, img_size)

    def _calculate_alpha(self, batch_idx):
        n_batches = self.trainer.num_training_batches
        epoch = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs
        p = float(batch_idx + epoch * n_batches) / max_epochs / n_batches
        alpha = 2 / (1 + np.exp(-10 * p)) - 1
        return alpha

    def forward(self, x, mask, alpha):
        x, feature = self.feature(x, mask)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        img_output = self.reconstruct(x)
        domain_output = self.domain_classifier(reverse_feature)

        return img_output, domain_output

    def training_step(self, batch, batch_idx):
        src_x, target_x, mask, y = batch
        device = src_x.device

        alpha = self._calculate_alpha(batch_idx)
        y_hat, src_domain = self.forward(src_x, mask, alpha)
        _, target_domain = self.forward(target_x, mask, alpha)

        loss = F.mse_loss(y_hat, y)
        src_err = F.nll_loss(src_domain, torch.zeros(y.size(0)).long().to(device))
        target_err = F.nll_loss(target_domain, torch.ones(y.size(0)).long().to(device))

        self.log('train_loss', loss)
        self.log('train_src_err', src_err)
        self.log('train_target_err', target_err)
        self.log('train_nrmse', nrmse2(y_hat, y))
        self.log('train_ssim', ssim(y_hat.unsqueeze(1), y.unsqueeze(1), data_range=1))
        self.log('train_psnr', psnr(y_hat, y, data_range=1))
        return loss + (src_err + target_err) * self.hparams.beta

    def validation_step(self, batch, batch_idx):
        src_x, target_x, mask, y = batch
        device = src_x.device

        alpha = self._calculate_alpha(batch_idx)
        y_hat, src_domain = self.forward(src_x, mask, alpha)
        _, target_domain = self.forward(target_x, mask, alpha)

        loss = F.mse_loss(y_hat, y)
        src_err = F.nll_loss(src_domain, torch.zeros(y.size(0)).long().to(device))
        target_err = F.nll_loss(target_domain, torch.ones(y.size(0)).long().to(device))

        self.log('val_loss', loss)
        self.log('val_src_err', src_err)
        self.log('val_target_err', target_err)
        self.log('val_nrmse', nrmse2(y_hat, y))
        self.log('val_ssim', ssim(y_hat.unsqueeze(1), y.unsqueeze(1), data_range=1))
        self.log('val_psnr', psnr(y_hat, y, data_range=1))
        return loss + (src_err + target_err) * self.hparams.beta

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @classmethod
    def load_model(cls, name, model_dir='DeepCascade'):
        lightning_logs = MODEL_PATH / model_dir / 'lightning_logs'
        assert os.path.isdir(lightning_logs), f'Lightning logs not found for model {model_dir}'

        checkpoints = lightning_logs / name / 'checkpoints'
        assert os.path.isdir(checkpoints), f'Checkpoints not found for {model_dir} {name}'

        latest = sorted(os.listdir(checkpoints))[-1]
        return cls.load_from_checkpoint(checkpoints / latest)
