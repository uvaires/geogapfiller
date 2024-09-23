
from .data_structures import ImageStack, BandName

def add_nvdi(image_stack: ImageStack) -> None:
    """Add the Normalized Difference Vegetation Index (NDVI) to the image stack."""
    red = image_stack.get_band_stack(BandName.RED.value)
    nir = image_stack.get_band_stack(BandName.NIR.value)
    ndvi = (nir - red) / (nir + red)
    image_stack.stack["NDVI"] = ndvi

def add_evi(image_stack: ImageStack) -> None:
    """Add the Enhanced Vegetation Index (EVI) to the image stack."""
    red = image_stack.get_band_stack(BandName.RED.value)
    nir = image_stack.get_band_stack(BandName.NIR.value)
    blue = image_stack.get_band_stack(BandName.BLUE.value)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    image_stack.stack["EVI"] = evi