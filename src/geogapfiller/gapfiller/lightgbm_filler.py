import os
import glob
from datetime import timedelta
import numpy as np
from joblib import Parallel, delayed
import rasterio
from lightgbm import LGBMRegressor


def lightgbm_filled(inputdir: str, outputdir: str, base_dates:list, window=15,  n_jobs=-1) -> None:
    """
    Fill the gaps in the geospatial rasters using a LightGBM model
    :param inputdir: location of the raster images
    :param outputdir: location to save the filled raster images
    :param base_dates: list of base dates
    :param window: window size
    :param n_jobs: number of jobs to run in parallel
    :return: None

    """
    # Read the images
    img_path = glob.glob(os.path.join(inputdir, '**', '*.tif'), recursive=True)
    # Extract the base dates and product
    img_names = _img_metadata(img_path)
    # Stack the images
    stack_imgs = _stack_raster(img_path)
    # Fill the gaps in the images
    filled_raster = _lightgbm_filling(stack_imgs, base_dates, window=window, n_jobs=n_jobs)

    # Export the filled images
    _export_raster(outputdir, img_path, filled_raster, img_names)


# Using a lightgbm model to fill the gaps
def _lightgbm_filling(stack_imgs, base_dates, window=15, n_jobs=-1):

    filled_arr = stack_imgs.copy()

    def fill_missing_for_index(i, j, raster_values, n_estimators=50, random_state=0):
        values_filled = raster_values.copy()

        for index_ii, raster_value in enumerate(values_filled):
            if np.isnan(raster_value):
                # Get the current date for the NaN value
                current_date = base_dates[index_ii]

                # Define the start and end of the window based on 15 days around the current date
                start_date = current_date - timedelta(days=window)
                end_date = current_date + timedelta(days=window)

                # Find the valid indices based on dates within the 15-day window
                valid_indices = [idx for idx, date in enumerate(base_dates)
                                 if start_date <= date <= end_date and not np.isnan(values_filled[idx])]

                if len(valid_indices) > 1:
                    # Prepare time (t_valid) and corresponding values (y_valid) for LightGBM model
                    t_valid = np.array([base_dates[idx].toordinal() for idx in valid_indices]).reshape(-1, 1)  # Ordinal date values
                    y_valid = values_filled[valid_indices]  # Corresponding values

                    # Initialize and train the LightGBM model
                    lgbm_model = LGBMRegressor(n_estimators=n_estimators, random_state=random_state)
                    lgbm_model.fit(t_valid, y_valid.ravel())

                    # Predict the missing value for the current date (in ordinal form)
                    values_filled[index_ii] = lgbm_model.predict(np.array([[current_date.toordinal()]]))[0]

        return i, j, values_filled

    # Prepare indices for parallel processing
    indices = [(i, j, stack_imgs[:, i, j]) for i in range(stack_imgs.shape[1]) for j in range(stack_imgs.shape[2])]

    # Process each pixel in parallel using Joblib
    results = Parallel(n_jobs=n_jobs)(delayed(fill_missing_for_index)(*index) for index in indices)

    # Update the filled_arr with the results
    for result in results:
        i, j, values_filled = result
        filled_arr[:, i, j] = values_filled

    return filled_arr



## Private functions ###

def _img_metadata(img_raster):

    imgs_names = []

    for metadates in img_raster:
        basename = os.path.basename(metadates)
        imgs_names.append(basename)

    return imgs_names



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


def _export_raster(outputdir, raster_img, raster_filled, img_names):
     # Open the first image to get the profile
    with rasterio.open(raster_img[0]) as src:
        band_profile = src.profile

    # Output the filled images
    outpupath = os.path.join(outputdir, 'data_processed', 'lightgbm_filler')

    for layer_data, current_name in zip(raster_filled, img_names):
        # Construct the output file path for the current layer
        dir_output_data = os.path.join(outpupath)
        os.makedirs(dir_output_data, exist_ok=True)
        output_filename = f"{current_name}_filled.tif"
        output_filepath = os.path.join(dir_output_data, output_filename)

        # Write the filled EVI image to a GeoTIFF file
        with rasterio.open(output_filepath, 'w', **band_profile) as dst:
            dst.write(layer_data, 1)