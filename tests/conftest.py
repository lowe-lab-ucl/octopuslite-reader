import pytest

import numpy as np

from octopuslite.metadata import Channels

RNG = np.random.default_rng(seed=1234)

METADATA = {
    "channel": RNG.choice([c.value for c in Channels]),
    "position": RNG.integers(0, 999),
    "time": RNG.integers(0, 99_999),
    "z": RNG.integers(0, 999),
}

FILENAME = (
    f"img_channel{METADATA['channel']:03d}"
    f"_position{METADATA['position']:03d}"
    f"_time{METADATA['time']:09d}"
    f"_z{METADATA['z']:03d}.tif"
)


@pytest.fixture
def octopuslite_filename():
    return FILENAME


@pytest.fixture
def octopuslite_metadata():
    return METADATA
