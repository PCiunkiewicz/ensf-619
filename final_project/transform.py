import numpy as np


def fft_transform(x, to='kspace'):
    """
    Transform a 2d image from image-space to k-space,
    or transform from k-space to image-space.
    """
    assert to in {'kspace', 'image'}, 'kwarg `to` must be one of "kspace" | "image"'
    if to == 'kspace':
        return np.fft.fft2(x)
    elif to == 'image':
        return np.fft.ifft2(x)


def minmax(x):
    """
    Perform min-max normalization to input data.
    """
    return  (x - x.min()) / (x.max() - x.min())


def remove_blank_slices(x):
    """
    Remove blank slices from the input image.
    """
    mask = x.sum(axis=(0, 1)) != 0
    return x[:,:,mask]


def center_crop_2d(x, size):
    """
    Center crop a 2d image.
    """
    assert len(size) == 2, 'size must be a tuple of length 2'
    h, w = x.shape[:2]
    ch, cw = size
    assert ch <= h and cw <= w, 'size must be smaller than image'
    h1 = (h - ch) // 2
    w1 = (w - cw) // 2
    return x[h1:h1+ch, w1:w1+cw]


def zero_pad_2d(x, size):
    """
    Zero pad a 2d image.
    """
    assert len(size) == 2, 'size must be a tuple of length 2'
    h, w = x.shape[:2]
    ch, cw = size
    assert ch >= h and cw >= w, 'size must be larger than image'
    h1 = (ch - h) // 2
    w1 = (cw - w) // 2
    return np.pad(x, ((h1, ch-h-h1), (w1, cw-w-w1)), 'constant')
