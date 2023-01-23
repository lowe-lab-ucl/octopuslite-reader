from __future__ import annotations

import dataclasses
import enum
import os
import re
from pathlib import Path

import numpy as np


@enum.unique
class Channels(enum.IntEnum):
    BRIGHTFIELD = 0
    GFP = 1
    RFP = 2
    IRFP = 3
    PHASE = 4
    WEIGHTS = 50
    MASK_IRFP = 96
    MASK_RFP = 97
    MASK_GFP = 98
    MASK = 99


WELL_ID_PATTERN = "(?P<alpha>[A-Z]?)(?P<numeric>[0-9]+)"

TIMESTAMP_PATTERN = "(?P<days>[0-9]+)d(?P<hours>[0-9]+)h(?P<mins>[0-9]+)m"

MICROMANAGER_FILEPATTERN = (
    "img_channel(?P<channel>[0-9]+)_position(?P<position>[0-9]+)"
    "_time(?P<time>[0-9]+)_z(?P<z>[0-9]+)"
)


INCUCYTE_FILEPATTERN = (
    "(?<experiment[A-Za-z0-9]+)_(?P<channel>[a-z]+)"
    "_(?P<position>[A-Z][0-9]+)_(?P<time>[0-9]+d[0-9]+h[0-9]+m)"
)


@dataclasses.dataclass
class WellPositionID:
    """A dataclass to store a well position identifier.

    Well positions can be either numeric, for example "11" or alphanumeric in cases
    of multi-well plates, e.g. "A11".

    """

    raw: str

    def __post_init__(self):
        params = re.match(WELL_ID_PATTERN, self.raw)
        if not params:
            raise ValueError(f"{self.raw} is not a valid position ID.")

    @property
    def alpha(self) -> str:
        params = re.match(WELL_ID_PATTERN, self.raw)
        if not params:
            return ""
        return params.groupdict()["alpha"]

    @property
    def numeric(self) -> int:
        params = re.match(WELL_ID_PATTERN, self.raw).groupdict()
        return int(params["numeric"])

    def __lt__(self, cls: WellPositionID) -> bool:
        if cls.alpha == self.alpha:
            return self.numeric < cls.numeric
        elif self.alpha < cls.alpha:
            return True
        else:
            return False


@dataclasses.dataclass
class Timestamp:
    raw: str

    def is_numeric(self) -> bool:
        params = re.match(TIMESTAMP_PATTERN, self.raw)
        return params is None

    def as_numeric(self) -> int:
        if not self.is_numeric():
            raise ValueError("Timestamp is not numeric.")
        return int(self.raw)

    def as_seconds(self):
        if self.is_numeric():
            raise ValueError(
                "Timestamp is stored as an index. Cannot convert to seconds."
            )

        params = re.match(TIMESTAMP_PATTERN, self.raw)
        params_numeric = {k: int(v) for k, v in params.groupdict().items()}

        seconds = (
            params_numeric["days"] * 24 * 60 * 60
            + params_numeric["hours"] * 60 * 60
            + params_numeric["mins"] * 60
        )
        return seconds

    def __lt__(self, cls: Timestamp) -> bool:
        if self.is_numeric() and cls.is_numeric():
            return self.as_numeric() < cls.as_numeric()

        try:
            flag = self.as_seconds() < cls.as_seconds()
        except ValueError:
            pass

        return flag


@dataclasses.dataclass
class ImageMetadata:
    filename: os.PathLike
    channel: Channels
    position: WellPositionID
    time: Timestamp
    z: int = 0
    experiment: str = "Default"
    transform: np.ndarray = np.empty((2, 2), dtype=np.float32)


class IncucyteMetadata(ImageMetadata):
    """Incucyte config


    <FileName Prefix>_<Site/Well ID>_<ImageNumber><Date/Timestamp><File Format>,
    where the File Name Timestamp option determines the timestamp.
    See “File Name Timestamp” on page 78.

    VID1957_phase_E11_5_06d12h05m

    """

    @staticmethod
    def from_filename(filename: os.PathLike) -> IncucyteMetadata:
        filename = Path(filename)
        filestem = str(filename.stem)
        params = re.match(INCUCYTE_FILEPATTERN, filestem)

        metadata = IncucyteMetadata(
            filename=filename,
            channel=None,
            channel=Channels[params["channel"].upper()],
            position=WellPositionID(params["position"]),
            time=Timestamp(params["time"]),
        )

        return metadata

    def filename(self) -> os.PathLike:
        pass


class MicromanagerMetadata(ImageMetadata):
    """Incucyte config"""

    @staticmethod
    def from_filename(filename: os.PathLike) -> MicromanagerMetadata:
        filename = Path(filename)
        filestem = str(filename.stem)

        params = re.match(MICROMANAGER_FILEPATTERN, filestem).groupdict()
        metadata = MicromanagerMetadata(
            filename=filename,
            channel=Channels(int(params["channel"])),
            position=WellPositionID(params["position"]),
            time=Timestamp(params["time"]),
            z=int(params["z"]),
        )
        return metadata

    def filename(self) -> os.PathLike:
        pass


class MetadataParser(enum.Enum):
    OCTOPUS = MicromanagerMetadata
    MICROMANAGER = MicromanagerMetadata
    INCUCYTE = IncucyteMetadata
