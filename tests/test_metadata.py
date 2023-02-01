import pytest

from octopuslite import metadata


def test_well_position_malformed():
    with pytest.raises(ValueError):
        well = metadata.WellPositionID("BB")  # noqa: F841


def test_well_position_alphanumeric():
    """Test alphanumeric well position."""

    well = metadata.WellPositionID("A11")
    assert well.raw == "A11"
    assert well.alpha == "A"
    assert well.numeric == 11


def test_well_position_numeric():
    """Test numeric only well position."""
    well = metadata.WellPositionID("11")
    assert well.raw == "11"
    assert well.numeric == 11


def test_well_ordering_lexicographic():
    """Test lexicographi sorting of wells."""

    well_a = metadata.WellPositionID("A11")

    well_b = metadata.WellPositionID("A12")
    assert well_a < well_b

    well_b = metadata.WellPositionID("A9")
    assert well_a > well_b

    well_b = metadata.WellPositionID("C12")
    assert well_a < well_b


def test_well_ordering_numeric():
    """Test numeric sorting of wells."""

    well_a = metadata.WellPositionID("11")
    well_b = metadata.WellPositionID("12")
    assert well_a < well_b


def test_well_ordering_sort():
    """Test sorting of wells."""

    well_a = metadata.WellPositionID("A11")
    well_b = metadata.WellPositionID("B12")

    wells = [well_b, well_a]
    sorted_wells = sorted(wells)
    assert sorted_wells == [well_a, well_b]


def test_timestamp_from_numeric():
    timestamp = metadata.Timestamp("11")
    assert timestamp.is_numeric() is True
    assert timestamp.as_numeric() == 11


def test_timestamp_from_timestamp():
    timestamp = metadata.Timestamp("06d12h05m")
    assert timestamp.is_numeric() is False
    assert timestamp.as_seconds() == 561900


def test_timestamp_ordering():
    pass


def test_timestamp_sorting():
    pass
