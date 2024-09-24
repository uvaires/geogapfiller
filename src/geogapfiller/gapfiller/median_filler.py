import os
import glob
from datetime import datetime, timedelta
import numpy as np
from joblib import Parallel, delayed
import rasterio


def med_filler(inputdir: str, outputdir: str) -> None:
    """
    Fill the gaps in the geospatial rasters using a median approach
    :param inputdir: location of the raster images
    :param outputdir: location to save the filled raster images
    :return: None
    """
    # Read the images
    img_path = glob.glob(os.path.join(inputdir, '**', '*.tif'), recursive=True)
    # Extract the base dates and product
    dates, _, _, year = _img_metadata(img_path)
    # Stack the EVI images
    stack_imgs = _stack_raster(img_path)
    # Fill the gaps in the EVI images
    filled_raster, _ = _median_filling(img_path, stack_imgs, n_jobs=-1)

    # Export the filled EVI images
    _export_raster(outputdir, img_path, year, dates)


def _median_filling(img_raster, arr, n_jobs=-1):

    base_dates = []
    all_dates = []

    for dates in img_raster:
        dates_raster = os.path.basename(dates).split('_')[0]
        formatted_date = f"{dates_raster[:4]}{dates_raster[4:6]}{dates_raster[6:]}"
        base_date = datetime.strptime(formatted_date, "%Y%m%d")
        base_dates.append(base_date)
        all_dates.append(base_date)

    start_date = min(base_dates)
    end_date = max(base_dates)
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    filled_arr = arr.copy()

    def fill_missing_for_index(i, j, raster_values):
        raster_values_filled = raster_values.copy()

        for index_ii, raster_value in enumerate(raster_values_filled):
            if np.isnan(raster_value):
                start_window = max(0, index_ii - 15)
                end_window = min(len(raster_values_filled), index_ii + 15)

                valid_values = raster_values_filled[start_window:end_window]
                if len(valid_values) > 0:
                    median_value = np.nanmedian(valid_values)
                    raster_values_filled[index_ii] = median_value

        return i, j, raster_values_filled

    # Prepare indices for parallel processing
    indices = [(i, j, arr[:, i, j]) for i in range(arr.shape[1]) for j in range(arr.shape[2])]

    # Process each pixel in parallel using Joblib
    results = Parallel(n_jobs=n_jobs)(delayed(fill_missing_for_index)(*index) for index in indices)

    # Update the filled_arr with the results
    for result in results:
        i, j, raster_values_filled = result
        filled_arr[:, i, j] = raster_values_filled

    return filled_arr, all_dates


## Private functions ###

def _img_metadata(img_raster):

    base_dates = []

    for metadates in img_raster:
        basename = os.path.basename(metadates)
        dates = basename.split('_')[0]
        base_dates.append(dates)

    return base_dates



def _stack_raster(img_raster):
    raster_layers = []
    for img in img_raster:
        with rasterio.open(img) as src:
            raster_layer = src.read(1)
            raster_layers.append(raster_layer)

    # stack the EVI layers
    stacked_raster = np.stack(raster_layers, axis=0)
    stacked_raster = np.squeeze(stacked_raster)

    return stacked_raster



def _export_raster(outputdir, raster_img, raster_filled, base_dates):
     # Open the first image to get the profile
    with rasterio.open(raster_img[0]) as src:
        band_profile = src.profile

    # Output the filled images
    outpupath = os.path.join(outputdir, 'data_processed', 'median_filler')

    for layer_data, current_date in zip(raster_filled, base_dates):
        # Construct the output file path for the current layer
        dir_output_data = os.path.join(outpupath)
        os.makedirs(dir_output_data, exist_ok=True)
        output_filename = f"{current_date}_raster.tif"
        output_filepath = os.path.join(dir_output_data, output_filename)

        # Write the filled EVI image to a GeoTIFF file
        with rasterio.open(output_filepath, 'w', **band_profile) as dst:
            dst.write(layer_data, 1)


