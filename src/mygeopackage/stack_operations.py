from typing import Optional, Protocol, Union, List
import numpy as np
import matplotlib.pyplot as plt

from .data_structures import ImageStack


class ReductionFunction(Protocol):
    """Protocol to define a numpy-like reduction function."""

    def __call__(self, array: np.ndarray, **kwargs) -> np.ndarray: ...


def apply_reduction(
    image_stack: ImageStack, np_reduce_func: ReductionFunction, axis: int = 0, **kwargs
) -> ImageStack:
    """Apply a reduction function to the image stack.

    Args:
        image_stack (ImageStack): image stack.
        np_reduce_func: numpy reduction function.
        axis (int, optional): axis to apply the reduction. Defaults to 0.
        **kwargs: additional arguments to be passed to the reduction function.

    Returns:
        ImageStack: reduced image stack.
    """
    reduced_bands = {}
    for band in image_stack.get_available_bands():
        band_stack = image_stack.get_band_stack(band)
        reduced_band = np_reduce_func(band_stack, axis=axis, **kwargs)
        reduced_band = np.expand_dims(reduced_band, axis=axis)
        reduced_bands[band] = reduced_band
    reduced_image_stack = ImageStack(reduced_bands, image_stack.georref_params.copy())
    return reduced_image_stack


def plot_stack(
    stack: ImageStack,
    band: Union[str, List[str]],
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Plot a band or a list of bands (RGB channels) from the stack.

    Args:
        stack (ImageStack): image stack.
        band (Union[str, List[str]]): band name or list of band names to
        be plotted. If a list is provided, the bands are plotted as RGB channels.
        cmap (str, optional): colormap. Defaults to 'viridis'.
        vmin (Optional[float], optional): minimum value for the colormap. Defaults to None.
        vmax (Optional[float], optional): maximum value for the colormap. Defaults to None.
    """
    if isinstance(band, str):
        plot_data = stack.get_band_stack(band)
        plt.imshow(plot_data[0], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label=band)
        plt.show()
    elif isinstance(band, list):
        data_bands = []
        for b in band:
            data_bands.append(stack.get_band_stack(b))
        plot_data = np.concatenate(data_bands, axis=0)
        plot_data = np.moveaxis(plot_data, 0, -1)
        # Increase contrast
        plot_data = plot_data * 2.5
        plt.imshow(plot_data)
        plt.show()
    else:
        raise ValueError("band must be a BandName or a list of BandName")
