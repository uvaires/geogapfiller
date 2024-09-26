import os
import glob
from datetime import timedelta
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import rasterio


def poly_filler(inputdir: str, outputdir: str, base_dates: list, window=15, poly_degree=2, n_jobs=-1) -> None:
    """
    Fill the gaps in the geospatial rasters using a polynomial regression model
    :param inputdir: location of the raster images
    :param outputdir: location to save the filled raster images
    :param base_dates: list of base dates
    :param window: window size
    :param poly_degree: polynomial degree
    :param n_jobs: number of jobs to run in parallel
    :return: None
    """
    # Read the images
    img_path = glob.glob(os.path.join(inputdir,  '**', '*.tif'), recursive=True)
    # Extract the base dates and product
    img_names = _img_metadata(img_path)
    # Stack the images
    stack_imgs = _stack_raster(img_path)
    # Fill the gaps in the images
    filled_raster = _poly_filling(stack_imgs, base_dates, window=window, poly_degree=poly_degree, n_jobs=n_jobs)

    # Export the filled images
    _export_raster(outputdir, img_path, filled_raster, img_names)


# Function to fill gaps using a polynomial of 2nd degree
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

## Private functions ###
def _img_metadata(img_raster):

    imgs_names = []

    for metadates in img_raster:
        basename = os.path.basename(metadates)
        imgs_names.append(basename)

    return imgs_names


def _stack_raster(raster_img):

    raster_layers = []
    for img in raster_img:
        with rasterio.open(img) as src:
            raster_layer = src.read(1)
            raster_layers.append(raster_layer)

    # stack the EVI layers
    stacked_raster = np.stack(raster_layers, axis=0)
    stacked_raster = np.squeeze(stacked_raster)

    return stacked_raster


def _export_raster(outputdir, img_path, raster_filled, base_names):

    # Open the first image to get the profile
    with rasterio.open(img_path[0]) as src:
        band_profile = src.profile

    # Output the filled EVI images
    outputpath = os.path.join(outputdir, 'data_processed', "polynomial_filler")

    for layer_data, current_names in zip(raster_filled, base_names):
        # Construct the output file path for the current layer
        dir_output_data = os.path.join(outputpath)
        os.makedirs(dir_output_data, exist_ok=True)
        output_filename = f"{current_names}_filled.tif"
        output_filepath = os.path.join(dir_output_data, output_filename)

        # Write the filled EVI image to a GeoTIFF file
        with rasterio.open(output_filepath, 'w', **band_profile) as dst:
            dst.write(layer_data, 1)
