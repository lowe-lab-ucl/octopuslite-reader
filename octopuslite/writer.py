import os

from .metadata import Channels, MetadataParser
from pathlib import Path
from skimage.io import imsave

import numpy.typing as npt


def write(
    data: npt.NDArray,
    path: os.PathLike,
    *,
    position: str = "001",
    channel: Channels = Channels.GFP,
    parser: MetadataParser = MetadataParser.OCTOPUS,
) -> None:
    """Write out image data in a format that can be read by DaskOctopus."""

    parser = parser.value
    path = Path(path)

    for t in range(data.shape[0]):
        metadata = parser.from_dict(
            {
                "filename": "",
                "position": position,
                "channel": channel,
                "time": int(t),
                "z": 0,
            }
        )

        filename = Path(path / metadata.to_filename()).with_suffix(".tif")
        imsave(filename, data[t, ...], check_contrast=False)
