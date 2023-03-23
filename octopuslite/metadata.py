from __future__ import annotations

import dataclasses
import datetime
import enum
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import patterns


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

    @staticmethod
    def from_raw(value: str) -> Channels:
        try:
            channel = Channels(int(value))
        except ValueError:
            channel = Channels[value.upper()]
        return channel


@dataclasses.dataclass
class WellPositionID:
    """A dataclass to store a well position identifier.

    Well positions can be either numeric, for example "11" or alphanumeric in cases
    of multi-well plates, e.g. "A11".

    """

    raw: str

    def __post_init__(self):
        params = re.match(patterns.WELL_ID_PATTERN, self.raw)
        if not params:
            raise ValueError(f"{self.raw} is not a valid position ID.")

    @property
    def alpha(self) -> str:
        params = re.match(patterns.WELL_ID_PATTERN, self.raw)
        if not params:
            return ""
        return params.groupdict()["alpha"]

    @property
    def numeric(self) -> int:
        params = re.match(patterns.WELL_ID_PATTERN, self.raw).groupdict()
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
        params = re.match(patterns.TIMESTAMP_PATTERN, self.raw)
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

        params = re.match(patterns.TIMESTAMP_PATTERN, self.raw)
        params_numeric = {k: int(v) for k, v in params.groupdict().items()}

        delta = datetime.timedelta(**params_numeric)
        return delta.total_seconds()

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
    transform: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> ImageMetadata:
        metadata = cls(
            filename=params["filename"],
            experiment=params.get("experiment", "Default"),
            channel=Channels.from_raw(params["channel"]),
            position=WellPositionID(params["position"]),
            time=Timestamp(str(params["time"])),
            z=int(params.get("z", 0)),
        )
        return metadata


def metadata_from_filename(
    filename: os.PathLike, pattern: str
) -> Dict[str, Any]:
    filename = Path(filename)
    filestem = str(filename.stem)
    params = re.match(pattern, filestem)

    if not params:
        raise ValueError("Could not parse filename with pattern.")

    params = params.groupdict()
    params.update({"filename": filename})

    return params


class IncucyteMetadata(ImageMetadata):
    """Incucyte config

    <FileName Prefix>_<Site/Well ID>_<ImageNumber><Date/Timestamp><File Format>,
    where the File Name Timestamp option determines the timestamp.
    See “File Name Timestamp” on page 78.

    VID1957_phase_E11_5_06d12h05m

    """

    @staticmethod
    def from_filename(filename: os.PathLike) -> IncucyteMetadata:
        params = metadata_from_filename(
            filename, patterns.INCUCYTE_FILEPATTERN
        )
        return IncucyteMetadata.from_dict(params)

    def to_filename(self) -> os.PathLike:
        raise NotImplementedError


class MicromanagerMetadata(ImageMetadata):
    """Micromanager/Octopus config

    img_channel002_position012_time000000995_z000

    """

    @staticmethod
    def from_filename(filename: os.PathLike) -> MicromanagerMetadata:
        params = metadata_from_filename(
            filename, patterns.MICROMANAGER_FILEPATTERN
        )
        return MicromanagerMetadata.from_dict(params)

    def to_filename(self) -> os.PathLike:
        fstr = (
            f"img_channel{self.channel.value:>03d}_"
            f"position{self.position.numeric:>03d}_"
            f"time{self.time.as_numeric():>09d}_"
            f"z{self.z:>03d}"
        )
        return fstr


class MetadataParser(enum.Enum):
    OCTOPUS = MicromanagerMetadata
    MICROMANAGER = MicromanagerMetadata
    INCUCYTE = IncucyteMetadata
