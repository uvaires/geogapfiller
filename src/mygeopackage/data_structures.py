from enum import Enum
from pathlib import Path
import rasterio
from pyproj import CRS
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


class BandName(Enum):
    """Enum to store the Sentinel-2 band names."""

    RED = "B04"
    GREEN = "B03"
    BLUE = "B02"
    NIR = "B08"


@dataclass
class GeoParams:
    """Class to store georeferencing parameters."""

    transform: rasterio.Affine
    crs: CRS
    nodata: float

    def copy(self) -> "GeoParams":
        return GeoParams(rasterio.Affine(*self.transform), CRS(self.crs), self.nodata)


@dataclass
class ImageStack:
    """Class to store a stack of rasters. Each band is stored separately."""

    stack: Dict[str, np.ndarray]
    georref_params: GeoParams

    def get_available_bands(self) -> List[str]:
        return list(self.stack.keys())

    def get_band_stack(self, band: str) -> np.ndarray:
        return self.stack[band]
    
    def export_band(self, band: str, file_path: str) -> None:
        """Export a band to a raster file."""
        file_path = Path(file_path)
        # Create the directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        band_stack = self.get_band_stack(band)
        with rasterio.open(
            file_path,
            "w",
            driver="GTiff",
            height=band_stack.shape[1],
            width=band_stack.shape[2],
            count=band_stack.shape[0],
            dtype=band_stack.dtype,
            crs=self.georref_params.crs,
            transform=self.georref_params.transform,
            nodata=self.georref_params.nodata,
        ) as dst:
            dst.write(band_stack)
