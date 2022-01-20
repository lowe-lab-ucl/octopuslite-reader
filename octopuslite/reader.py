import os
import sys
import dask

import dask.array as da
import numpy as np

from skimage import io
from typing import Union, Optional

from .utils import (
    Channels,
    estimate_background,
    estimate_mask,
    parse_filename,
    remove_background,
    remove_outliers,
)
from .transform import parse_transforms


class DaskOctopusLiteLoader:
    """Load multidimensional image stacks using lazy loading.

    A simple class to load OctopusLite data from a directory. Caches data once
    it is loaded to prevent excessive I/O to the data server. Can directly
    address different channels using the `Channels` enumerator.

    Parameters
    ----------
    path : str
        The path to the dataset.
    crop : tuple, optional
        An optional tuple which can be used to perform a centred crop on the data.
    transforms : Path to transform matrix
        Transform matrix (as np.ndarray) to be applied to the image stack.
    remove_background : bool
        Use a estimated polynomial surface to remove uneven illumination.

    Methods
    -------
    __getitem__ : Channels, str
        Return a dask lazy array of the image data for the channel. If cropping
        has been specified, the images are also cropped to this size.

    Properties
    ----------
    shape :
        Returns the shape of the uncropped data.
    channels :
        Return the channels found in the dataset.
    path :
        Return the path to the dataset.
    files(channel_name) :
        Return the pathname of all channel_name images found in the dataset.
    channel_name_from_index(channel_value):
        Return the channel specified by channel_value.

    Usage
    -----
    >>> octopus =  DaskOctopusLiteLoader('/path/to/your/data/')
    >>> gfp = octopus["GFP"]
    >>> gfp_filenames = octopus.files("GFP")
    """

    def __init__(
        self,
        path: str,
        crop: Optional[tuple] = None,
        transforms: Optional[os.PathLike] = None,
        remove_background: bool = True,
    ):
        self.path = path
        self._files = {}
        self._lazy_arrays = {}
        self._crop = crop
        self._shape = ()
        self._remove_background = remove_background

        print(f'Using cropping: {crop}')

        # parse the files
        self._parse_files()
        self._transformer = parse_transforms(transforms)

    def __contains__(self, channel):
        return channel in self.channels

    @property
    def channels(self):
        return list(self._files.keys())

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, channel_name: Union[str, Channels]):

        if isinstance(channel_name, str):
            channel_name = Channels[channel_name.upper()]

        if channel_name not in self.channels:
            raise ValueError(f"Channel {channel_name} not found in {self.path}")

        return self._lazy_arrays[channel_name]


    def files(self, channel_name: str) -> list:
        return self._files[Channels[channel_name.upper()]]

    def _load_and_process(self, fn: str) -> np.ndarray:
        """Load and crop the image."""
        image = io.imread(fn)

        t = int(parse_filename(fn)['time'])
        image = self._transformer(image, t)

        if self._crop is None:
            return image

        assert isinstance(self._crop, tuple)

        dims = image.ndim
        shape = image.shape
        dtype = image.dtype
        crop = np.array(self._crop).astype(np.int64)

        # check that we don't exceed any dimensions
        assert all([crop[i] <= s for i, s in enumerate(shape)])

        # automagically build the slices for the array
        cslice = lambda d: slice(
            int((shape[d] - crop[d]) // 2),
            int((shape[d] - crop[d]) // 2 + crop[d])
        )
        crops = tuple([cslice(d) for d in range(dims)])

        cleaned = remove_outliers(image[crops])

        if self._remove_background:
            return remove_background(cleaned) #.astype(dtype)

        return cleaned

    def _parse_files(self):
        """Parse out the files from the folder and create lazy arrays."""

        # find the files in the dataset
        files = [
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if f.endswith((".tif", ".tiff"))
        ]

        if not files:
            raise FileNotFoundError(f"No files found in directory: {self.path}")

        # take a sample of the dataset
        sample = io.imread(files[0])
        self._shape = sample.shape if self._crop is None else self._crop

        channels = {k:[] for k in Channels}

        # parse the files
        for f in files:
            channel = parse_filename(f)['channel']
            channels[channel].append(f)

        # sort them by time
        for channel in channels.keys():
            channels[channel].sort(key=lambda f: parse_filename(f)['time'])

        # set the output type
        dtype = np.float32 if self._remove_background else sample.dtype

        # remove any channels that are empty
        self._files = {k: v for k, v in channels.items() if v}

        # now set up the lazy loaders
        for channel, files in self._files.items():
            self._lazy_arrays[channel] = [
                da.from_delayed(
                    dask.delayed(self._load_and_process)(fn),
                    shape=self._shape,
                    dtype=dtype
                ) for fn in files
            ]

            # concatenate them along the time axis
            self._lazy_arrays[channel] = da.stack(self._lazy_arrays[channel], axis=0)
