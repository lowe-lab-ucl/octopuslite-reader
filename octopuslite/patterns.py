WELL_ID_PATTERN = "(?P<alpha>[A-Z]?)(?P<numeric>[0-9]+)"

TIMESTAMP_PATTERN = "(?P<days>[0-9]+)d(?P<hours>[0-9]+)h(?P<minutes>[0-9]+)m"

MICROMANAGER_FILEPATTERN = (
    "img_channel(?P<channel>[0-9]+)_position(?P<position>[0-9]+)"
    "_time(?P<time>[0-9]+)_z(?P<z>[0-9]+)"
)

INCUCYTE_FILEPATTERN = (
    "(?P<experiment>[a-zA-Z0-9]+)_(?P<channel>[a-z]+)"
    "_(?P<position>[A-Z][0-9]+)_(?P<location>[0-9]+)"
    "_(?P<time>[0-9]+d[0-9]+h[0-9]+m)"
)
