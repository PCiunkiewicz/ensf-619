import os

import tqdm
import numpy as np
import nibabel as nib

from paths import DATA_PATH, ORIGINALS, NEGATIVES, NEWBORNS
from transform import remove_blank_slices, center_crop_2d, fft_transform
from sampling import gaussian2d


def process_images(size, mode='original'):
    """
    Process images into subsampled kspace data and targets.
    """
    assert mode in {'original', 'negative', 'newborn'}, 'Invalid mode'
    shape = (size, size) # 164 was found to be smallest dim across all images
    mask = gaussian2d(shape, 0.2, radius=18, seed=42)
    mask = np.fft.fftshift(mask)
    np.save(DATA_PATH / f'sampling/mask_{size}.npy', mask)

    data = []
    targets = []

    if mode in {'original', 'negative'}:
        path = ORIGINALS if mode == 'original' else NEGATIVES
        savedir = DATA_PATH / 'adult'
        slice_axis = 2
    elif mode == 'newborn':
        path = NEWBORNS
        savedir = DATA_PATH / 'newborn'
        slice_axis = 1

    for filename in tqdm.tqdm(os.listdir(path)):
        img = nib.load(path / filename).get_fdata()
        img = remove_blank_slices(img)
        for i in range(img.shape[slice_axis]):
            if mode in {'original', 'negative'}:
                cropped = center_crop_2d(img[:,:,i], shape)
            elif mode == 'newborn':
                cropped = center_crop_2d(img[:,i,:], shape)
            kspace = fft_transform(cropped, to='kspace')
            kspace = kspace * mask
            subsampled = fft_transform(kspace, to='image')

            data.append(np.array((subsampled.real, subsampled.imag)))
            targets.append(cropped)

    data = np.array(data, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    np.save(savedir / f'data_{mode}_{size}.npy', data)
    np.save(savedir / f'targets_{mode}_{size}.npy', targets)


def load_data(size, mode='original'):
    """
    Load processed data and targets.
    """
    assert mode in {'original', 'negative', 'newborn'}, 'Invalid mode'
    if mode in {'original', 'negative'}:
        savedir = DATA_PATH / 'adult'
    elif mode == 'newborn':
        savedir = DATA_PATH / 'newborn'

    data = np.load(savedir / f'data_{mode}_{size}.npy')
    targets = np.load(savedir / f'targets_{mode}_{size}.npy')
    mask = np.load(DATA_PATH / f'sampling/mask_{size}.npy')

    return data, targets, mask


if __name__ == '__main__':
    for size in [100, 164]:
        for mode in ['original', 'negative', 'newborn']:
            process_images(size, mode=mode)
