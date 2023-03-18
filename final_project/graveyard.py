import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def conv_block(in_channels, out_channels, n_convs=3, kernel_size=3):
    return nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.ReLU(inplace=True)
        ] * n_convs,
        # nn.MaxPool2d(2)
        # nn.Upsample(2),
    )


class Unet(pl.LightningModule):
    def __init__(self, n_channels=2):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = conv_block(self.n_channels, 48)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = conv_block(48, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = conv_block(128, 256)

        self.up1 = nn.Upsample(2)
        self.conv5 = conv_block(256, 128)
        self.up2 = nn.Upsample(2)
        self.conv6 = conv_block(128, 64)
        self.up3 = nn.Upsample(2)
        self.conv7 = conv_block(64, 48)

        self.conv8 = nn.Conv2d(48, self.n_channels, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

        u1 = torch.cat([self.up1(c4), c3], dim=1)
        c5 = self.conv5(u1)
        u2 = torch.cat([self.up2(c5), c2], dim=1)
        c6 = self.conv6(u2)
        u3 = torch.cat([self.up3(c6), c1], dim=1)
        c7 = self.conv7(u3)

        c8 = self.conv8(c7)
        out = torch.add(c8, x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return {'val_loss': loss}
