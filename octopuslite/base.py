import abc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dask.array as da
import numpy as np

from .metadata import Channels, ImageMetadata, MetadataParser, WellPositionID
from .transform import parse_transforms


class BaseReader(abc.ABC):
    def __init__(
        self,
        path: os.PathLike,
        *,
        position: Optional[str] = None,
        crop: Optional[tuple] = None,
        transforms: Optional[Union[np.ndarray, os.PathLike]] = None,
        remove_background: bool = True,
        parser: MetadataParser = MetadataParser.OCTOPUS,
    ):
        self.path = Path(path)
        self.crop = crop
        self.parser = parser.value
        self.transformer = parse_transforms(transforms)
        self.remove_background = remove_background
        self.position = (
            WellPositionID(position) if position is not None else None
        )
        self._shape: tuple = ()
        self._data: Dict[str, Any] = {}
        self._initalized: bool = False
        self._metadata: Dict[str, List[ImageMetadata]] = {}

        self.post_init()

    def __contains__(self, channel: str) -> bool:
        return channel in self.channels

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of images."""
        raise NotImplementedError

    @property
    def shape(self):
        return self._shape

    @property
    def metadata(self):
        return self._metadata

    @abc.abstractproperty
    def channels(self):
        raise NotImplementedError

    def __getitem__(self, channel_name: Union[str, Channels]) -> da.Array:
        """Get an image array for the specified channel."""

        if isinstance(channel_name, str):
            channel_name = Channels[channel_name.upper()]

        if channel_name not in self.channels:
            raise ValueError(
                f"Channel {channel_name} not found in {self.path}"
            )

        return self._data[channel_name]

    @abc.abstractmethod
    def post_init(self):
        raise NotImplementedError

    def asarray(self):
        return da.stack([d for d in self._data.values()], axis=0)
