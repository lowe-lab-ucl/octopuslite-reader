import os

from .metadata import Channels, MetadataParser
from pathlib import Path
from skimage.io import imsave
from typing import Optional

import numpy.typing as npt


class Writer:
    def __init__(
        self,
        path: os.PathLike,
        *,
        position: str = "001",
        channel: Channels = Channels.GFP,
        parser: MetadataParser = MetadataParser.OCTOPUS,
        compression: Optional[str] = None,
    ) -> None:
        """Writer for image data."""
        self.path = Path(path)
        self.position = position
        self.parser = parser.value
        self.channel = channel
        self.compression = compression

    def write(self, image: npt.NDArray, t: int, *, z: int = 0) -> None:
        """Write a single frame of the data."""
        params = {
            "filename": "",
            "position": self.position,
            "channel": self.channel,
            "time": int(t),
            "z": z,
        }
        metadata = self.parser.from_dict(params)

        filename = Path(self.path / metadata.to_filename()).with_suffix(".tif")
        imsave(
            filename, image, check_contrast=False, compression=self.compression
        )


def write(data: npt.NDArray, path: os.PathLike, **kwargs) -> None:
    """Write out image data in a format that can be read by DaskOctopus."""

    writer = Writer(path, **kwargs)

    for t in range(data.shape[0]):
        writer.write(data[t, ...], t)
