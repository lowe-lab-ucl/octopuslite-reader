import enum
import os
import re

import numpy as np
from skimage import io



@enum.unique
class Channels(enum.Enum):
    BRIGHTFIELD = 0
    GFP = 1
    RFP = 2
    IRFP = 3
    PHASE = 4
    WEIGHTS = 98
    MASK = 99


def crop_image(image, crop):
    """ Crop an image or volume """
    if crop is None: return image

    assert(isinstance(crop, tuple))

    dims = image.ndim
    shape = image.shape
    crop = np.array(crop).astype(np.int)

    # check that we don't exceed any dimensions
    assert(all([crop[i] <= s for i, s in enumerate(shape)]))

    # automagically build the slices for the array
    cslice = lambda d: slice(int((shape[d]-crop[d])/2),
                             int((shape[d]-crop[d])/2+crop[d]))
    crops = tuple([cslice(d) for d in range(dims)])

    return image[crops]


class SimpleOctopusLiteLoader(object):
    """ SimpleOctopusLiteLoader

    A simple class to load OctopusLite data from a directory.
    Caches data once it is loaded to prevent excesive io to
    the data server.

    Can directly address fluorescence channels using the
    `Channels` enumerator:

        Channels.BRIGHTFIELD
        Channels.GFP
        Channels.RFP
        Channels.IRFP

    Usage:
        octopus = SimpleOctopusLiteLoader('/path/to/your/data')
        gfp = octopus[Channels.GFP]

    Parameters
    ----------

    path : str
    crop : bool

    """
    def __init__(self, path: str, crop: bool = True):
        self.path = path
        self._files = {}
        self._data = {}

        # parse the files
        self._parse_files()

        self._crop = crop
        self._shape = (0, 1352, 1688)

        print(f'Using cropping: {crop}')

    def __contains__(self, channel):
        return channel in self.channels

    @property
    def channels(self):
        return list(self._files.keys())

    @property
    def shape(self):
        return self._shape

    def channel_name_from_index(self, channel_index: int):
        return Channels(int(channel_index))

    def preload(self, channels=None):
        if channels is None:
            channels = self.channels

        for channel in channels:
            self._load_channel(channel)


    def __getitem__(self, channel_name):
        assert(channel_name in self.channels)

        if channel_name not in self._data:
            self._load_channel(channel_name)

        return self._data[channel_name]


    def _parse_files(self):
        """ parse out the files from the folder """
        files = [f for f in os.listdir(self.path) if f.endswith('.tif')]

        def parse_filename(fn):
            pattern = "img_channel([0-9]+)_position([0-9]+)_time([0-9]+)_z([0-9]+)"
            params = re.match(pattern, fn)
            return self.channel_name_from_index(params.group(1)), params.group(3)

        channels = {k:[] for k in Channels}

        # parse the files and sort them
        for f in files:
            channel, time = parse_filename(f)
            channels[channel].append(f)

        for channel in channels.keys():
            channels[channel].sort(key=lambda f: parse_filename(f)[1])

        # remove any channels that are empty
        self._files = {k:v for k, v in channels.items() if v}

    def _load_channel(self, channel_name):
        assert(channel_name in self.channels)

        def load_image(fn):
            im_full = io.imread(os.path.join(self.path, fn))
            if self._crop:
                return crop_image(im_full, (1200, 1600))
            return im_full

         # load the first image
        im = load_image(self._files[channel_name][0])

        # preload the stack
        stack = np.zeros((len(self._files[channel_name]),)+im.shape, dtype=im.dtype)
        self._shape = stack.shape

        print('Loading: {} --> {} ({})...'.format(channel_name, stack.shape, stack.dtype))

        stack[0,...] = im
        for i in range(1, stack.shape[0]):
            stack[i,...] = load_image(self._files[channel_name][i])

        self._data[channel_name] = stack


    def clear_cache(self, channel_name):
        print('Warning! You are clearing the cache for: {}'.format(channel_name))
        self._data[channel_name] = None
