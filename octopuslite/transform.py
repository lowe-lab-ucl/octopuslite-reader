import os

# import xml.etree.ElementTree as ET
import numpy as np
import skimage.transform as tf

# from .utils import parse_filename


class StackTransformer:
    def __init__(self, transforms: np.ndarray):
        self.transforms = transforms

    def __call__(self, x, idx):

        if self.transforms is None:
            return x

        tform = tf.AffineTransform(translation=self.transforms[idx, :2, 2])
        return tf.warp(x, tform, preserve_range=True)


def parse_transforms(path: os.PathLike) -> StackTransformer:
    """Parse a file or folder containing registration transforms.

    Parameters
    ----------
    path : PathLike
        A path to either a numpy file of transforms, or a folder containing
        transforms as XML files.

    Returns
    -------
    transformer : StackTransformer
        A class that will transform an image with the correct transform.

    """
    if path is None:
        return StackTransformer(None)

    if path.endswith(".npy"):
        transforms = np.load(path)
        return StackTransformer(transforms)

    # # get the transform files and then sort them by time
    # transform_files = [f for f in os.listdir(path) if f.endswith(".xml")]
    # transform_files.sort(key=lambda f: parse_filename(f)['time'])
    # n = len(transform_files)
    # transforms = np.empty((n, 2), dtype=np.float32) * np.nan
    #
    # for file in transform_files:
    #
    #     t = int(parse_filename(file)['time'])
    #
    #     # we assume that the first time point has zero displacement
    #     if t == 0:
    #         data = np.array([0.0, 0.0], dtype=np.float32)
    #         transforms[t, :] = data
    #         continue
    #
    #     tree = ET.parse(os.path.join(path, file))
    #     root = tree.getroot()
    #
    #     for child in root[:1]:
    #         data = np.fromstring(child.attrib["data"], dtype=np.float32, sep=' ')
    #         transforms[t, :] = data
    #
    # # interpolate any missing values
    # for dim in range(transforms.shape[-1]):
    #     v = transforms[:, dim]
    #     nans = np.isnan(v)
    #     nans_idx = lambda x: x.nonzero()[0]
    #     v[nans] = np.interp(nans_idx(nans), nans_idx(~nans), v[~nans])
    #     transforms[:, dim] = -v
    #
    # transformer = StackTransformer(transforms)

    return StackTransformer(None)
