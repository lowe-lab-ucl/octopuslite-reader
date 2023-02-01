import os

import numpy as np
import skimage.transform as tf

from typing import Optional, Union


class StackTransformer:
    def __init__(self, transforms: np.ndarray):
        self.transforms = transforms

    def __call__(self, x, idx):

        if self.transforms is None:
            return x

        tform = tf.AffineTransform(translation=self.transforms[idx, :2, 2])
        return tf.warp(x, tform, order=0, preserve_range=True)


def parse_transforms(
    transforms: Optional[Union[np.array, os.PathLike]]
) -> StackTransformer:
    """Parse a file or folder containing registration transforms.

    Parameters
    ----------
    transforms : PathLike, array
        A path to either a numpy file or a numpy array of transforms.

    Returns
    -------
    transformer : StackTransformer
        A class that will transform an image with the correct transform.
    """
    if isinstance(transforms, (str, os.PathLike)):
        transforms = np.load(transforms)

    return StackTransformer(transforms)
