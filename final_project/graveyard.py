import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt


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


def custom_imshow(imgs, titles=None, figsize=(10, 10), cmap='gray', origin='lower'):
    fig, axes = plt.subplots(1, len(imgs), figsize=figsize)
    for i, img in enumerate(imgs):
        axes[i].imshow(img.T, cmap=cmap, origin=origin)
        axes[i].set_axis_off()
        if titles is not None:
            axes[i].set(title=titles[i])
    plt.show()

# custom_imshow([targets[150], mask, *data[150]], ['Target', 'Mask', 'Subsampled Real', 'Subsampled Imag'])


# class DeepCascadeTrainer(pl.Trainer):
#     def __init__(self, batch_size, max_epochs, precision='16-mixed', download=False, **kwargs):
#         super().__init__(
#             default_root_dir=MODEL_PATH / 'DeepCascade',
#             accelerator=str(DEVICE) if str(DEVICE) in {'mps', 'cpu'} else 'auto',
#             max_epochs=max_epochs,
#             precision=precision,
#             callbacks=[
#                 ModelCheckpoint(save_weights_only=True, mode='min', monitor='val_loss'),
#                 LearningRateMonitor('epoch')
#             ]
#         )

#         self.batch_size = batch_size
#         self.is_trained = False
#         # Attempt to load pre-trained model
#         self.model = self.load_pretrained(download)
#         # Fall back to new DeepCascade instance if self.model is None
#         if not self.model:
#             self.model = DeepCascade(**kwargs)

#     def load_pretrained(self, download=False):
#         ckpt_path = PRETRAINED_PATH / 'DeepCascade.ckpt'
#         # Download pre-trained model from Github
#         if download:
#             self.download_pretrained(ckpt_path)

#         # Load pre-trained model from disk if file exists
#         if os.path.isfile(ckpt_path):
#             self.is_trained = True
#             LOG.info(f'Loading pre-trained model from path: {ckpt_path}')
#             return DeepCascade.load_from_checkpoint(ckpt_path)

#     def download_pretrained(self, ckpt_path):
#         # Pre-trained model checkpoint from my Github repo
#         pretrained_url = 'https://raw.githubusercontent.com/pciunkiewicz/ensf-619/master/final_project/models/pretrained/DeepCascade.ckpt'
#         os.makedirs('models', exist_ok=True)

#         # Only download the file if not found on disk; no overwrite
#         if not os.path.isfile(ckpt_path):
#             try:
#                 LOG.info(f'Downloading pre-trained model from {pretrained_url}')
#                 urllib.request.urlretrieve(pretrained_url, ckpt_path)
#             except HTTPError as exception:
#                 LOG.error(exception)

#     def fit(self, train, val):
#         # Check if pre-trained model has been loaded
#         if not self.is_trained:
#             # Create PyTorch DataLoaders for training and validation sets
#             train_loader = DataLoader(
#                 train,
#                 batch_size=self.batch_size,
#                 shuffle=True,
#                 drop_last=True,
#                 pin_memory=True,
#                 num_workers=NUM_WORKERS,
#             )
#             val_loader = DataLoader(
#                 val,
#                 batch_size=self.batch_size,
#                 shuffle=False,
#                 drop_last=True,
#                 pin_memory=True,
#                 num_workers=NUM_WORKERS,
#             )

#             super().fit(self.model, train_loader, val_loader)
#             self.model = DeepCascade.load_from_checkpoint(self.checkpoint_callback.best_model_path)
#         else:
#             LOG.info('Model is already trained; returning pre-trained model.')
#         return self.model
