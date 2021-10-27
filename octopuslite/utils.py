import enum
import os
import re
import numpy as np

from scipy.ndimage import median_filter


OCTOPUSLITE_FILEPATTERN = (
    "img_channel(?P<channel>[0-9]+)_position(?P<position>[0-9]+)"
    "_time(?P<time>[0-9]+)_z(?P<z>[0-9]+)"
)


@enum.unique
class Channels(enum.Enum):
    BRIGHTFIELD = 0
    GFP = 1
    RFP = 2
    IRFP = 3
    PHASE = 4
    WEIGHTS = 98
    MASK = 99


def remove_outliers(x: np.ndarray) -> np.ndarray:
    """Remove bright outlier pixels from an image.

    Parameters
    ----------
    x : np.ndarray
        An input image containing bright outlier pixels.

    Returns
    -------
    x : np.ndarray
    	An image with the bright outlier pixels removed.
    """
    med_x = median_filter(x, size=2)
    mask = x > med_x
    x = x * (1-mask) + (mask*med_x)
    return x


def estimate_background(x: np.ndarray) -> np.ndarray:
    """Estimate background using a second order polynomial surface.

    Estimate the background of an image using a second-order polynomial surface
    assuming sparse signal in the image.  Essentially a massive least-squares
    fit of the image to the polynomial.

    Parameters
    ----------
    x : np.ndarray
        An input image which is to be used for estimating the background.

    Returns
    -------
    background_estimate : np.ndarray
    	A second order polynomial surface representing the estimated background
        of the image.
    """

    # set up arrays for params and the output surface
    A = np.zeros((x.shape[0]*x.shape[1], 6))
    background_estimate = np.zeros((x.shape[1], x.shape[0]))

    u, v = np.meshgrid(
        np.arange(x.shape[1], dtype=np.float32),
        np.arange(x.shape[0], dtype=np.float32),
    )
    A[:, 0] = 1.
    A[:, 1] = np.reshape(u, (x.shape[0] * x.shape[1],))
    A[:, 2] = np.reshape(v, (x.shape[0] * x.shape[1],))
    A[:, 3] = A[:, 1] * A[:, 1]
    A[:, 4] = A[:, 1] * A[:, 2]
    A[:, 5] = A[:, 2] * A[:, 2]

    # convert to a matrix
    A = np.matrix(A)

    # calculate the parameters
    k = np.linalg.inv(A.T * A) * A.T
    k = np.squeeze(np.array(np.dot(k, np.ravel(x))))

    # calculate the surface
    background_estimate = k[0] + k[1]*u + k[2]*v + k[3]*u*u + k[4]*u*v + k[5]*v*v
    return background_estimate


def parse_filename(filename: os.PathLike) -> dict:
    """Parse an OctopusLite filename and retreive metadata from the file.

    Parameters
    ----------
    filename : PathLike
        The full path to a file to parse.

    Returns
    -------
    metadata : dict
        A dictionary containing the parsed metadata.
    """
    pth, filename = os.path.split(filename)
    params = re.match(OCTOPUSLITE_FILEPATTERN, filename)

    metadata = {
        "filename": filename,
        "channel": Channels(int(params.group('channel'))),
        "time": params.group('time'),
        "position": params.group('position'),
        "z": params.group('z'),
        # "timestamp": os.stat(filename).st_mtime,
    }

    return metadata
