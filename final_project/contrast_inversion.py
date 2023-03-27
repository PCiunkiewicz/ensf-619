"""
Contrast inversion on gray and white matter in the brain images.
"""
import os

import tqdm
import nibabel as nib

from paths import ORIGINALS, NEGATIVES, MASKS


def contrast_invert(filename, save=False):
    """
    Perform contrast inversion on gray and white matter
    in the brain images. Optionally save as nifti.
    """
    img = nib.load(ORIGINALS / filename)
    affine = img.affine
    img = img.get_fdata()

    mask = nib.load(MASKS / filename).get_fdata()
    mask = (mask == 2) | (mask == 3)

    negative = img * mask
    negative = (negative.max() - negative) * mask
    negative += img * ~mask

    if save:
        nii = nib.Nifti1Image(negative, affine)
        nib.save(nii, NEGATIVES / filename)

    return negative


if __name__ == '__main__':
    for filename in tqdm.tqdm(os.listdir(ORIGINALS)):
        contrast_invert(filename, save=True)
