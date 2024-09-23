from typing import Union
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import rasterio

from .data_structures import GeoParams, ImageStack


def stack_months(
    scan_results: pd.DataFrame, target_months: Optional[List[int]] = None
) -> ImageStack:
    """Stack rasters. Each band is stacked separately.

    Args:
        scan_results (pd.DataFrame): DataFrame with the metadata/paths of the images.
        target_months (Optional[List[int]], optional): list of months to be stacked. If None,
        all months are stacked. Defaults to None.

    Returns:
        ImageStack: stacked rasters.
    """
    band_stack = {}
    file_paths_used = []
    for row in scan_results.itertuples():
        file_path = row.file_path
        month = row.month
        band = row.band
        # Skip months not in the target_months list
        if target_months is not None and month not in target_months:
            continue
        # Read raster
        with rasterio.open(file_path) as src:
            raster = src.read()
        file_paths_used.append(file_path)

        if band not in band_stack.keys():
            band_stack[band] = []
        band_stack[band].append(raster)

    for key, value in band_stack.items():
        band_stack[key] = np.concatenate(value, axis=0)

    georref_params = _get_georrefferencing_parameters(file_paths_used)
    stack = ImageStack(band_stack, georref_params)
    return stack


def _get_georrefferencing_parameters(file_path: Union[Path, List[Path]]) -> GeoParams:
    """Get georeferencing parameters from one or more raster files.
    All rasters must have the same georeferencing parameters.

    Args:
        file_path (Union[Path, List[Path]]): a single file path or a list of file paths.

    Returns:
        GeoParams: georeferencing parameters.
    """
    geo_params = None
    for path in file_path:
        with rasterio.open(path) as src:
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
        if geo_params is None:
            geo_params = GeoParams(transform, crs, nodata)
        else:
            if geo_params.transform != transform:
                raise ValueError(
                    f"Transform parameters are different. First raster: {geo_params.transform}, current raster: {transform}"
                )
            if geo_params.crs != crs:
                raise ValueError(
                    f"CRS parameters are different. First raster: {geo_params.crs}, current raster: {crs}"
                )
            if np.isnan(geo_params.nodata) and np.isnan(geo_params.nodata):
                continue
            if geo_params.nodata != nodata:
                raise ValueError(
                    f"Nodata parameters are different. First raster: {geo_params.nodata}, current raster: {nodata}"
                )
    return geo_params


class ImageScanner:
    def __init__(
        self,
        search_folder: str,
    ) -> None:
        """Scans a folder for images and creates a DataFrame with the metadata/paths
        Any folder structure can be used as long as the file names follow the pattern:
        month_x_band_y.tif, where x and y are the month and band name, respectively.
        Example: month_1_band_B02.tif

        Args:
            search_folder (str): Folder to be scanned
        """
        self.search_folder = Path(search_folder)
        self._scan_results_full = None

    def scan(
        self, ini_month: Optional[int] = None, final_month: Optional[int] = None
    ) -> pd.DataFrame:
        """Scans the folder for images and returns a DataFrame with the metadata/paths.

        Args:
            ini_month (Optional[int], optional): keep only images from this month
            onwards. If None, no filter is applied. Defaults to None.
            final_month (Optional[int], optional): keep only images from this month and
            before. If None, no filter is applied. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with the metadata/paths of the images.
        """
        # If scan results are already available, just use them. Otherwise, create them.
        # It is useful to store the full scan results to allow subsequent filtering without
        # the need to scan the images again.
        if self._scan_results_full == None:
            self._scan_results_full = self._create_scan_df()

        scan_results = self._filter_scam_results(self._scan_results_full, ini_month, final_month)
        # Sort by month and band for easier visualization
        scan_results = scan_results.sort_values(by=["month", "band"])
        # Reset index to make index continuous after filtering and sorting
        scan_results = scan_results.reset_index(drop=True)
        return scan_results

    def _filter_scam_results(
        self,
        scan_results: pd.DataFrame,
        ini_month: Optional[int] = None,
        final_month: Optional[int] = None,
    ) -> pd.DataFrame:
        if ini_month is not None:
            scan_results = scan_results[scan_results["month"] >= ini_month]
        if final_month is not None:
            scan_results = scan_results[scan_results["month"] <= final_month]
        return scan_results

    def _create_scan_df(self) -> pd.DataFrame:
        file_paths = self._list_tif_files()
        metadata = self._extract_metadata(file_paths)
        metadata["file_path"] = file_paths
        return pd.DataFrame(metadata)

    def _list_tif_files(self) -> List[Path]:
        return list(self.search_folder.glob("**/*.tif"))

    def _extract_metadata(self, file_paths: List[Path]) -> dict:
        months = [int(p.stem.split("_")[1]) for p in file_paths]
        bands = [p.stem.split("_")[3] for p in file_paths]
        return {"month": months, "band": bands}
