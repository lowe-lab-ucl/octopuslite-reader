from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter
from skimage import io

from .metadata import ImageMetadata
from .transform import StackTransformer


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
    x = x * (1 - mask) + (mask * med_x)
    return x


def remove_background(x: np.ndarray) -> np.ndarray:
    """Remove background using a polynomial surface.

    Parameters
    ----------
    x : np.ndarray
        An input image.

    Returns
    -------
    corrected : np.ndarray
        The corrected input image, with the background removed.
    """
    maskh, maskw = estimate_mask(x)
    x = x.astype(np.float32)
    bg = estimate_background(x[maskh, maskw])
    corrected = x[maskh, maskw] - bg
    corrected = corrected - np.min(corrected)
    x[maskh, maskw] = corrected
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
    A = np.zeros((x.shape[0] * x.shape[1], 6))
    background_estimate = np.zeros((x.shape[1], x.shape[0]))

    u, v = np.meshgrid(
        np.arange(x.shape[1], dtype=np.float32),
        np.arange(x.shape[0], dtype=np.float32),
    )
    A[:, 0] = 1.0
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
    background_estimate = (
        k[0] + k[1] * u + k[2] * v + k[3] * u * u + k[4] * u * v + k[5] * v * v
    )
    return background_estimate


def estimate_mask(x: np.ndarray) -> Tuple[slice]:
    """Estimate the mask of a frame.

    Masking may occur when frame registration has been performed.

    Parameters
    ----------
    x : np.ndarray
        An input image which is to be used for estimating the background.

    Returns
    -------
    mask : tuple (2,)
        Slices representing the mask of the image.
    """
    if hasattr(x, "compute"):
        x = x.compute()
    nonzero = np.nonzero(x)
    sh = slice(np.min(nonzero[0]), np.max(nonzero[0]) + 1, 1)
    sw = slice(np.min(nonzero[1]), np.max(nonzero[1]) + 1, 1)
    return sh, sw


def _crop_slice(shape: tuple, crop: tuple, dim: int) -> slice:
    return slice(
        int((shape[dim] - crop[dim]) // 2),
        int((shape[dim] - crop[dim]) // 2 + crop[dim]),
    )


def crop_image(img: np.ndarray, crop: Tuple[int]) -> np.ndarray:
    """Crops a central window from an input image given a crop area size tuple

    Parameters
    ----------
    img : np.ndarray
        Input image.
    crop : tuple
        An tuple which is used to perform a centred crop on the
        image data.

    Returns
    -------
    img : np.ndarray
        The cropped image.

    """
    shape = img.shape
    dims = img.ndim

    crops = tuple([_crop_slice(shape, crop, d) for d in range(dims)])
    img = img[crops]

    return img


def _load_and_process(
    metadata: ImageMetadata,
    *,
    crop: Optional[Tuple[int]] = None,
    remove_bg: bool = False,
    transformer: Optional[StackTransformer] = None,
) -> np.ndarray:
    """Load and crop the image."""
    image = io.imread(metadata.filename)

    # if self.transformer is not None:
    #     # need to use index of file as some frames may have been removed
    #     channel = parse_filename(fn).channel
    #     files = self.files(channel.name)
    #     files.sort(key=lambda f: parse_filename(f).time)
    #     idx = files.index(fn)
    #     image = self.transformer(image, idx)

    if crop is not None:
        # crop the image
        crop = np.array(crop).astype(np.int64)
        image = crop_image(image, crop)

    # check channel to see if label
    channel = metadata.channel

    # labels cannot be preprocessed so return here
    if channel.name.startswith(("MASK", "WEIGHTS")):
        return image

    if remove_bg:
        cleaned = remove_outliers(image)
        image = remove_background(cleaned)

    return image
