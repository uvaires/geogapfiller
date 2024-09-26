import os
import glob
import rasterio
from datetime import timedelta
from collections import Counter
from joblib import Parallel, delayed
import numpy as np


def predicted_img(inputdir: str, outputdir: str, base_dates: list, window=15, interval=1, n_jobs=-1):
    """
    Fill the gaps in the geospatial rasters using a harmonic model
    :param inputdir: location of the raster images
    :param outputdir: location to save the filled raster images
    :param base_dates: list of base dates
    :param interval: interval between the base dates
    :param window: window size for harmonic filling
    :param n_jobs: number of jobs to run in parallel
    :return: None
    """

    # Read the image path
    img_path = glob.glob(os.path.join(inputdir, '**', '*.tif'), recursive=True)

    # Stack the images
    stack_imgs = _stack_raster(img_path)

    # Fill the missing values in the image layers using harmonic filling
    img_filled, data_range = predict_daily_imgs(stack_imgs, base_dates, interval=interval, window=window, n_jobs=n_jobs)

    # Convert dates to string
    data_range = _convert_dates(data_range)

    img_layer, img_profile = _read_image(img_path[0])

    # Export the predicted images
    _export_img_daily(outputdir, img_filled, img_profile, data_range)


def predict_daily_imgs(stack_rasters, base_dates, interval=1, window=15, n_jobs=-1):
    dates_counter = Counter(base_dates)
    start_date = min(base_dates)
    end_date = max(base_dates)

    # Calculate the date range based on the specified interval
    date_range = [start_date + timedelta(days=i) for i in range(0, (end_date - start_date).days + 1, interval)]

    # Create an array for the predicted images with the same dimensions
    original_arr = np.full((len(date_range), stack_rasters.shape[1], stack_rasters.shape[2]), np.nan)

    # Fill in the original array with the available raster data
    for i, date in enumerate(base_dates):
        if start_date <= date <= end_date:
            try:
                date_index = date_range.index(date)

                if dates_counter[date] > 1:
                    existing_values = original_arr[date_index, :, :]
                    new_values = stack_rasters[i, :, :]
                    combined_values = np.nanmean([existing_values, new_values], axis=0)
                else:
                    combined_values = stack_rasters[i, :, :]

                original_arr[date_index, :, :] = combined_values
            except ValueError:
                pass  # Date is outside the specified interval

    # Use harmonic filling to fill the gaps
    filled_arr = _harmonic_filling(original_arr, date_range, window=window, n_jobs=n_jobs)

    return filled_arr, date_range


def _harmonic_filling(stacked_images, base_dates, window=15, n_jobs=-1):
    filled_arr = stacked_images.copy()

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

                    # Clip values to stay within range [-1, 1] (if applicable)
                    pred_value = np.clip(pred_value, -1, 1)

                    raster_values_filled[index_ii] = pred_value

        return i, j, raster_values_filled

    # Prepare indices for parallel processing
    indices = [(i, j, stacked_images[:, i, j], base_dates)
               for i in range(stacked_images.shape[1])
               for j in range(stacked_images.shape[2])]

    # Process each pixel in parallel using Joblib
    results = Parallel(n_jobs=n_jobs)(delayed(fill_missing_for_index)(*index) for index in indices)

    # Update the filled_arr with the results
    for result in results:
        i, j, raster_values_filled = result
        filled_arr[:, i, j] = raster_values_filled

    return filled_arr


## Private functions ##

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
def _convert_dates(dates):
    base_dates = [date.strftime("%Y%m%d") for date in dates]
    return base_dates


# Function to save the predicted images to drive
def _export_img_daily(outputdir, img_predicted, img_profile, data_range):
    for layer_idx, (layer_data, current_dates) in enumerate(zip(img_predicted, data_range)):
        print(current_dates)
        dir_output_date = os.path.join(outputdir, 'predicted_img', 'harmonic_pred')
        os.makedirs(dir_output_date, exist_ok=True)
        output_filepath = os.path.join(dir_output_date, f"{current_dates}_pred.tif")
        img_profile['nodata'] = np.nan
        img_profile['compress'] = 'lzw'

        with rasterio.open(output_filepath, 'w', **img_profile) as dst:
            dst.write(layer_data, 1)
