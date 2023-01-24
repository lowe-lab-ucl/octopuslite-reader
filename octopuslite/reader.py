import dask
import dask.array as da
import numpy as np
from skimage import io

from . import base
from .metadata import Channels
from .utils import _load_and_process


class DaskOctopusLite(base.BaseReader):
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
    parser : enum
        A parser for metadata.
    position : str, optional
        An optional position identifier.

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

    Methods
    -------
    files(channel_name) :
        Return the pathname of all channel_name images found in the dataset.

    Usage
    -----
    >>> octopus =  DaskOctopusLiteLoader('/path/to/your/data/')
    >>> gfp = octopus["GFP"]
    >>> gfp_filenames = octopus.files("GFP")
    """

    @property
    def channels(self):
        return list(self._metadata.keys())

    def __len__(self):
        return 0

    def files(self, channel_name: str) -> list:
        return [
            f.filename for f in self._metadata[Channels[channel_name.upper()]]
        ]

    def post_init(self):
        """Parse out the files from the folder and create lazy arrays."""

        # find the files in the dataset
        files = list(self.path.glob("*.tif*"))

        if not files:
            raise FileNotFoundError(
                f"No files found in directory: {self.path}"
            )

        # take a sample of the dataset
        sample = io.imread(files[0])
        self._shape = sample.shape if self.crop is None else self.crop

        channels = {k: [] for k in Channels}

        # parse all the files
        for f in files:
            metadata = self.parser.from_filename(f)
            channel = metadata.channel
            channels[channel].append(metadata)

        # sort them by time
        for channel in channels.keys():
            channels[channel].sort(key=lambda metadata: metadata.time)

        # set the output type
        dtype = np.float32 if self.remove_background else sample.dtype

        # remove any channels that are empty
        self._metadata = {k: v for k, v in channels.items() if v}

        # now set up the lazy loaders
        for channel, metadata in self._metadata.items():
            self._data[channel] = [
                da.from_delayed(
                    dask.delayed(_load_and_process)(
                        meta,
                        crop=self.crop,
                        transformer=self.transformer,
                        remove_bg=self.remove_background,
                    ),
                    shape=self._shape,
                    dtype=dtype,
                )
                for meta in metadata
            ]

            # concatenate them along the time axis
            self._data[channel] = da.stack(self._data[channel], axis=0)


DaskOctopusLiteLoader = DaskOctopusLite


# def remove_bg(x):
#     x = remove_outliers(x)
#     x = remove_background(x)
#     return x


# class VirtualOctopusLite(base.BaseReader):
#     """Virtual reader for on-disk single tiff stacks."""

#     def post_init(self):
#         stack = io.imread(self.path)
#         self._shape = stack.shape[1:-1]
#         self._channels = [Channels(i) for i in range(stack.shape[-1])]
#         self._files = self.path
#         self._data = {
#             channel: da.stack(
#                 [
#                     da.from_delayed(
#                         dask.delayed(remove_bg)(stack[n, ..., idx]),
#                         shape=self._shape,
#                         dtype=np.float32,
#                     )
#                     for n in range(stack.shape[0])
#                 ]
#             )
#             for idx, channel in enumerate(self.channels)
#         }

#     @property
#     def channels(self):
#         return self._channels
