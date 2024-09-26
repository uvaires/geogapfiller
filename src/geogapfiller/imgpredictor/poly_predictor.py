import os
import glob
import rasterio
from datetime import timedelta
from collections import Counter
from joblib import Parallel, delayed
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def predicted_img(inputdir: str, outputdir:str, base_dates:list, window=15, interval=1, poly_degree=2, n_jobs=-1):
    """
    Fill the gaps in the geospatial rasters using a polynomial regression model
    :param inputdir: location of the raster images
    :param outputdir: location to save the filled raster images
    :param base_dates: list of base dates
    :param interval: interval between the base dates
    :param poly_degree: polynomial degree
    :param n_jobs: number of jobs to run in parallel
    :return: None
    """

    # Read the image path
    img_path = glob.glob(os.path.join(inputdir, '**', '*.tif'), recursive=True)

    # Stack the images
    stack_imgs = _stack_raster(img_path)

    # Fill the missing values in the image layers
    img_filled, data_range = predict_daily_imgs(stack_imgs, base_dates, interval=interval, window=window, poly_degree=poly_degree,
                                                n_jobs=n_jobs)
    # Convert dates to string
    data_range = _convert_dates(data_range)

    img_layer, img_profile = _read_image(img_path[0])

    # Export the predicted images
    _export_img_daily(outputdir, img_filled, img_profile,  data_range)


def predict_daily_imgs(stack_rasters, base_dates, interval=1, window=15, poly_degree=2, n_jobs=-1):
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

    # Use the polynomial filling to fill the gaps
    filled_arr = _poly_filling(original_arr, date_range, window=window, poly_degree=poly_degree, n_jobs=n_jobs)

    return filled_arr, date_range


def _poly_filling(stacked_images, base_dates, window=15, poly_degree=2, n_jobs=-1):
    filled_arr = stacked_images.copy()

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

                X_valid = np.array([base_dates[idx].toordinal() for idx in valid_indices]).reshape(-1, 1)
                y_valid = raster_values_filled[valid_indices]

                # Only fit if there are more than one valid data point
                if len(X_valid) > 1:
                    poly = PolynomialFeatures(poly_degree)
                    X_poly = poly.fit_transform(X_valid)

                    model = LinearRegression()
                    model.fit(X_poly, y_valid)

                    # Predict the value for the current date
                    pred_value = model.predict(poly.transform(np.array([[current_date.toordinal()]])))

                    # Clip values to stay within range [-1, 1]
                    pred_value = np.clip(pred_value, -1, 1)

                    raster_values_filled[index_ii] = pred_value[0]

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


## Privite functions ##
# Read image profile
def _read_image(image_path):
    with rasterio.open(image_path) as src:
        img_layer = src.read(1)
        img_profile = src.profile
    return img_layer, img_profile


# Read the image information and return the image layers, band profile, and product
def _img_metadata(img_raster):

    imgs_names = []

    for metadates in img_raster:
        basename = os.path.basename(metadates)
        imgs_names.append(basename)

    return imgs_names

# Stack the raster images
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


# Function to save the predicted images to drive
def _export_img_daily(outputdir, img_predicted, img_profile,  data_range):
    for layer_idx, (layer_data, current_dates) in enumerate(zip(img_predicted, data_range)):
        print(current_dates)
        dir_output_date = os.path.join(outputdir,'predicted_img', 'polynomial_pred')
        os.makedirs(dir_output_date, exist_ok=True)
        output_filepath = os.path.join(dir_output_date, f"{current_dates}_pred.tif")
        img_profile['nodata'] = np.nan
        img_profile['compress'] = 'lzw'

        with rasterio.open(output_filepath, 'w', **img_profile) as dst:
            dst.write(layer_data, 1)


def _convert_dates(dates):
    base_dates = [date.strftime("%Y%m%d") for date in dates]
    return base_dates
