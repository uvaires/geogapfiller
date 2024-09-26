import os
import glob
from datetime import datetime, timedelta
import numpy as np
from joblib import Parallel, delayed
import rasterio
from scipy.optimize import curve_fit

def harmonic_filler(inputdir: str, outputdir: str, base_dates: list, window=15, n_jobs=-1):
    """
    Fill the gaps in the geospatial rasters using a harmonic model without predicting daily images.
    :param inputdir: location of the raster images
    :param outputdir: location to save the filled raster images
    :param base_dates: list of base dates corresponding to the rasters
    :param window: window size for harmonic filling
    :param n_jobs: number of jobs to run in parallel
    :return: None
    """
    # Read the image path
    img_path = glob.glob(os.path.join(inputdir, '**', '*.tif'), recursive=True)

    # Stack the images
    stack_imgs = _stack_raster(img_path)

    # Fill the missing values in the image layers using harmonic filling
    img_filled = fill_gaps_in_images(stack_imgs, base_dates, window=window, n_jobs=n_jobs)

    # Convert dates to string
    img_names = _img_metadata(img_path)

    img_layer, img_profile = _read_image(img_path[0])

    # Export the filled images
    _export_img_filled(outputdir, img_filled, img_profile, img_names)


def fill_gaps_in_images(stack_rasters, base_dates, window=15, n_jobs=-1):
    """
    Fill missing values in the stacked raster data using harmonic filling.
    :param stack_rasters: stacked raster data (time, rows, columns)
    :param base_dates: list of dates corresponding to each raster in stack_rasters
    :param window: window size for harmonic filling
    :param n_jobs: number of jobs to run in parallel
    :return: filled array
    """
    filled_arr = stack_rasters.copy()

    def harmonic_model_fit(X, y):
        # Harmonic model terms: a sine and cosine wave with a yearly cycle
        frequency = 2 * np.pi / 365  # yearly frequency (assuming daily data)

        # Construct the design matrix for harmonic regression (includes intercept, sin, cos)
        X_harmonic = np.column_stack([np.ones_like(X), np.sin(frequency * X), np.cos(frequency * X)])

        # Solve using least squares
        coeffs, _, _, _ = np.linalg.lstsq(X_harmonic, y, rcond=None)
        return coeffs

    def harmonic_predict(coeffs, X_pred):
        # Harmonic prediction using the fitted coefficients
        frequency = 2 * np.pi / 365  # yearly frequency
        X_harmonic_pred = np.column_stack(
            [np.ones_like(X_pred), np.sin(frequency * X_pred), np.cos(frequency * X_pred)])
        return np.dot(X_harmonic_pred, coeffs)

    def fill_missing_for_index(i, j, raster_values, base_dates):
        """
        Fill missing values for a specific pixel time series.
        :param i: row index
        :param j: column index
        :param raster_values: time series of raster values at (i, j)
        :param base_dates: list of dates corresponding to the time series
        :return: i, j, filled time series
        """
        raster_values_filled = raster_values.copy()

        for index_ii, raster_value in enumerate(raster_values_filled):
            if np.isnan(raster_value):
                # Get the current date for the NaN value
                current_date = base_dates[index_ii]

                # Define the start and end of the window
                start_date = current_date - timedelta(days=window)
                end_date = current_date + timedelta(days=window)

                # Find valid indices within the window
                valid_indices = [idx for idx, date in enumerate(base_dates)
                                 if start_date <= date <= end_date and not np.isnan(raster_values_filled[idx])]

                # Only fit if there are valid data points
                if valid_indices:
                    X_valid = np.array([base_dates[idx].toordinal() for idx in valid_indices])
                    y_valid = raster_values_filled[valid_indices]

                    # Fit the harmonic model
                    coeffs = harmonic_model_fit(X_valid, y_valid)

                    # Predict the value for the current date
                    X_pred = np.array([[current_date.toordinal()]])
                    pred_value = harmonic_predict(coeffs, X_pred)[0]


                    raster_values_filled[index_ii] = pred_value

        return i, j, raster_values_filled

    # Prepare indices for parallel processing
    indices = [(i, j, stack_rasters[:, i, j], base_dates)
               for i in range(stack_rasters.shape[1])
               for j in range(stack_rasters.shape[2])]

    # Process each pixel in parallel using Joblib
    results = Parallel(n_jobs=n_jobs)(delayed(fill_missing_for_index)(*index) for index in indices)

    # Update the filled_arr with the results
    for result in results:
        i, j, raster_values_filled = result
        filled_arr[:, i, j] = raster_values_filled

    return filled_arr


# Export the filled images to disk
def _export_img_filled(outputdir, img_filled, img_profile, data_range):
    """
    Export the gap-filled images to the output directory.
    :param outputdir: output directory
    :param img_filled: gap-filled raster array (time, rows, columns)
    :param img_profile: raster profile for exporting
    :param data_range: list of base dates corresponding to each raster
    :return: None
    """
    for layer_idx, (layer_data, current_date) in enumerate(zip(img_filled, data_range)):
        print(f"Filling gaps for date: {current_date}")
        dir_output_date = os.path.join(outputdir,  'data_processed', 'harmonic_filler')
        os.makedirs(dir_output_date, exist_ok=True)
        output_filepath = os.path.join(dir_output_date, f"{current_date}_filled.tif")
        img_profile['nodata'] = np.nan
        img_profile['compress'] = 'lzw'

        with rasterio.open(output_filepath, 'w', **img_profile) as dst:
            dst.write(layer_data, 1)

# Read image profile
def _read_image(image_path):
    with rasterio.open(image_path) as src:
        img_layer = src.read(1)
        img_profile = src.profile
    return img_layer, img_profile


# Stack the raster images
def _stack_raster(img_raster):
    raster_layers = []
    for img in img_raster:
        with rasterio.open(img) as src:
            raster_layer = src.read(1)
            raster_layers.append(raster_layer)

    # Stack the layers into a single 3D array (time, rows, columns)
    stacked_raster = np.stack(raster_layers, axis=0)
    return stacked_raster


# Convert dates to string
def _img_metadata(img_raster):

    imgs_names = []

    for metadates in img_raster:
        basename = os.path.basename(metadates)
        imgs_names.append(basename)

    return imgs_names


#
# def harmonic_filler(inputdir:str, outputdir: str, base_dates, n_jobs = -1) -> None:
#     """
#     Fill the gaps in the geospatial rasters using a harmonic regression model
#     :param inputdir: location of the raster images
#     :param outputdir: location to save the filled raster images
#     :param base_dates: list of base dates
#     :param window: window size
#     :param n_jobs: number of jobs to run in parallel
#     :return: None
#     """
#     # Read the images
#     img_path = glob.glob(os.path.join(inputdir, '**', '*.tif'), recursive=True)
#     # Extract the base dates and product
#     img_names = _img_metadata(img_path)
#     # Stack the images
#     stack_imgs = _stack_raster(img_path)
#     # Fill the gaps in the images
#     filled_raster = _harmonic_filling(stack_imgs, base_dates, n_jobs=n_jobs)
#
#     # Export the filled images
#     _export_raster(outputdir, img_path, filled_raster, img_names)
#
#
# # Using a harmonic model to fill the gaps
# def _harmonic_filling(stack_imgs, base_dates, window=15, n_jobs=-1):
#
#     filled_arr = stack_imgs.copy()
#
#     def _fill_missing_for_index(i, j, raster_values, base_dates):
#         values_filled = raster_values.copy()
#
#         for index_ii, raster_value in enumerate(values_filled):
#             if np.isnan(raster_value):
#                 # Get the current date for the NaN value
#                 current_date = base_dates[index_ii]
#
#                 # Define the start and end of the window based on 15 days around the current date
#                 start_date = current_date - timedelta(days=window)
#                 end_date = current_date + timedelta(days=window)
#
#                 # Find valid indices based on dates within the window
#                 valid_indices = [idx for idx, date in enumerate(base_dates)
#                                  if start_date <= date <= end_date and not np.isnan(values_filled[idx])]
#
#                 # Prepare time (t_valid) and corresponding values (y_valid) for the harmonic model
#                 if len(valid_indices) > 1:
#                     t_valid = np.array([base_dates[idx].toordinal() for idx in valid_indices])  # Time in ordinal form
#                     y_valid = values_filled[valid_indices]  # Corresponding values
#
#                     # Fit the harmonic model
#                     try:
#                         pred_values = _fit_harmonic_model(t_valid, y_valid)
#                         # Predict the value for the current date (index_ii)
#                         pred_value = _harmonic_model(current_date.toordinal(), *pred_values)
#
#                         # Assign the predicted value to the missing point
#                         values_filled[index_ii] = pred_value
#                     except Exception as e:
#                         pass  # Handle the fit failure gracefully
#
#         return i, j, values_filled
#
#     # Prepare indices for parallel processing
#     indices = [(i, j, stack_imgs[:, i, j], base_dates) for i in range(stack_imgs.shape[1]) for j in range(stack_imgs.shape[2])]
#
#     # Process each pixel in parallel using Joblib
#     results = Parallel(n_jobs=n_jobs)(delayed(_fill_missing_for_index)(*index) for index in indices)
#
#     # Update the filled_arr with the results
#     for result in results:
#         i, j, values_filled = result
#         filled_arr[:, i, j] = values_filled
#
#     return filled_arr
#
#
# def _harmonic_model(t, amplitude, frequency, phase, offset):
#     """
#     Harmonic model: amplitude * sin(2 * pi * frequency * t + phase) + offset
#     """
#     return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset
#
#
# def _fit_harmonic_model(t, y):
#     """
#     Fit the harmonic model to the provided data (t, y).
#     """
#     # Initial guess for the amplitude, frequency, phase, and offset
#     p0 = [np.nanmax(y) - np.nanmin(y), 1.0 / (2 * np.pi), 0, np.nanmin(y)]
#
#     # Fit the harmonic model using non-linear least squares
#     params, _ = curve_fit(_harmonic_model, t, y, p0=p0)
#
#     return params
#
#
# ## Private functions ###
# #
# def _img_metadata(img_raster):
#
#     imgs_names = []
#
#     for metadates in img_raster:
#         basename = os.path.basename(metadates)
#         imgs_names.append(basename)
#
#     return imgs_names
#
#
#
# def _stack_raster(img_raster):
#     raster_layers = []
#     for img in img_raster:
#         with rasterio.open(img) as src:
#             raster_layer = src.read(1)
#             raster_layers.append(raster_layer)
#
#     # stack the EVI layers
#     stacked_raster = np.stack(raster_layers, axis=0)
#     stacked_raster = np.squeeze(stacked_raster)
#
#     return stacked_raster


# def _export_raster(outputdir, raster_img, raster_filled, img_names):
#      # Open the first image to get the profile
#     with rasterio.open(raster_img[0]) as src:
#         band_profile = src.profile
#
#     # Output the filled images
#     outpupath = os.path.join(outputdir, 'data_processed', 'harmonic_filler')
#
#     for layer_data, current_name in zip(raster_filled, img_names):
#         # Construct the output file path for the current layer
#         dir_output_data = os.path.join(outpupath)
#         os.makedirs(dir_output_data, exist_ok=True)
#         output_filename = f"{current_name}_filled.tif"
#         output_filepath = os.path.join(dir_output_data, output_filename)
#
#         # Write the filled EVI image to a GeoTIFF file
#         with rasterio.open(output_filepath, 'w', **band_profile) as dst:
#             dst.write(layer_data, 1)

