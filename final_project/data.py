"""
Data loading and processing module.
"""
import os

import tqdm
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import TensorDataset

from paths import DATA_PATH, ORIGINALS, NEGATIVES, NEWBORNS
from utils import remove_blank_slices, center_crop_2d, min_max_norm
from sampling import gaussian2d


class DeepCascadeDataset(TensorDataset):
    """
    Dataset for DeepCascade.
    """
    def __init__(self, images, masks, transform=None):
        super().__init__(images, masks)
        assert images.size(0) == masks.size(0), 'Size mismatch between tensors'
        self.images = images
        self.masks = masks
        self.transform = transform
        self.val = False

    def __getitem__(self, index):
        img = self.images[index]
        mask = self.masks[np.random.randint(self.masks.size(0))]
        if self.transform and not self.val:
            img = self.transform(img)
        kspace = torch.fft.fft2(img) * mask
        kspace = torch.cat([kspace.real, kspace.imag], dim=-3)

        return kspace, 1 - mask, img[0]


class DANNDataset(TensorDataset):
    """
    Dataset for DeepCascade domain adaptation.
    """
    def __init__(self, src_images, masks, target_images, transform=None):
        super().__init__(src_images, masks)
        assert src_images.size(0) == masks.size(0), 'Size mismatch between tensors'
        self.src_images = src_images
        if target_images.size(0) != src_images.size(0):
            target_images = target_images[torch.randint(target_images.size(0), (src_images.size(0),))]
        self.target_images = target_images
        self.masks = masks
        self.transform = transform
        self.val = False

    def __getitem__(self, index):
        src_img = self.src_images[index]
        target_img = self.target_images[index]
        mask = self.masks[np.random.randint(self.masks.size(0))]
        if self.transform and not self.val:
            src_img = self.transform(src_img)
            target_img = self.transform(target_img)

        src_kspace = torch.fft.fft2(src_img) * mask
        src_kspace = torch.cat([src_kspace.real, src_kspace.imag], dim=-3)

        target_kspace = torch.fft.fft2(target_img) * mask
        target_kspace = torch.cat([target_kspace.real, target_kspace.imag], dim=-3)

        return src_kspace, target_kspace, 1 - mask, src_img[0]


def process_images(size, mode='original'):
    """
    Process images into subsampled kspace data and targets.
    """
    assert mode in {'original', 'negative', 'newborn'}, 'Invalid mode'
    shape = (size, size) # 164 was found to be smallest dim across all images

    if mode in {'original', 'negative'}:
        path = ORIGINALS if mode == 'original' else NEGATIVES
        savedir = DATA_PATH / 'adult'
    elif mode == 'newborn':
        path = NEWBORNS
        savedir = DATA_PATH / 'newborn'

    images = []
    masks = []
    for filename in tqdm.tqdm(os.listdir(path)):
        img = nib.load(path / filename).get_fdata()
        if mode == 'newborn':
            img = np.swapaxes(img, 2, 1)
        img = remove_blank_slices(img)
        img = min_max_norm(img)
        for i in range(img.shape[2]):
            cropped = center_crop_2d(img[:,:,i], shape)
            images.append(cropped)

            mask = gaussian2d(shape, 0.2, radius=18)
            masks.append(np.fft.fftshift(mask))

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=bool)

    np.save(savedir / f'images_{mode}_{size}.npy', images)
    np.save(savedir / f'masks_{mode}_{size}.npy', masks)


def load_data(size, mode='original'):
    """
    Load processed data and targets.
    """
    assert mode in {'original', 'negative', 'newborn'}, 'Invalid mode'
    if mode in {'original', 'negative'}:
        savedir = DATA_PATH / 'adult'
    elif mode == 'newborn':
        savedir = DATA_PATH / 'newborn'

    images = np.load(savedir / f'images_{mode}_{size}.npy')
    masks = np.load(savedir / f'masks_{mode}_{size}.npy')

    return images, masks


if __name__ == '__main__':
    for size in [100, 164]:
        for mode in ['original', 'negative', 'newborn']:
            process_images(size, mode=mode)
