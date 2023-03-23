from octopuslite import DaskOctopus
from octopuslite.writer import write
from octopuslite.metadata import Channels


from pathlib import Path
import numpy as np

import pytest


@pytest.mark.parametrize("channel", [Channels.GFP, Channels.MASK])
def test_writer(tmp_path, channel):
    data = np.random.randint(0, high=255, size=(100, 64, 64), dtype=np.uint8)

    filepath = Path(tmp_path)

    write(data, filepath, position="001", channel=channel)

    octopus = DaskOctopus(filepath, remove_background=False)

    recovered_data = octopus[channel]

    assert data.shape == recovered_data.shape
    np.testing.assert_equal(data, np.asarray(recovered_data))
