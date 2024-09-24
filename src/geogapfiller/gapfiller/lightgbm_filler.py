import os
import glob
from datetime import datetime, timedelta
import numpy as np
from joblib import Parallel, delayed
import rasterio
from lightgbm import LGBMRegressor


def lightgbm_evi_filled(base_dir: str, filling_tech: str, station_name) -> None:
    """
    Fill the gaps in the EVI images using a lightgbm model
    :param base_dir: location of the EVI images
    :param tile_id: tile ID
    :param filling_tech: name of the filling technique
    :return: None
    """
    # Read the images
    evi_img = glob.glob(os.path.join(base_dir, 'data_processed', station_name, '**', 'spectral_index', '**', '*.tif'), recursive=True)
    # Extract the base dates and product
    dates, product, tile_id, year = _img_metadata(evi_img)
    # Stack the EVI images
    stack_imgs = _stack_evi(evi_img)
    # Fill the gaps in the EVI images
    filled_evi, _ = _lightgbm_filling(evi_img, stack_imgs, n_jobs=-1)

    # Export the filled EVI images
    _export_evi(base_dir, evi_img, year, filled_evi, dates, product, filling_tech, station_name)


# Using a lightgbm model to fill the gaps
def _lightgbm_filling(evi_img, arr, n_jobs=-1):
    """ Fill the gaps in the EVI images using a lightgbm model
    :param evi_img: EVI images
    :param arr: Stacked EVI images
    :param n_jobs: Number of parallel jobs
    :return: EVI images with filled gaps

    """
    base_dates = []
    all_dates = []

    for dates in evi_img:
        dates_evi = os.path.basename(dates).split('_')[0]
        formatted_date = f"{dates_evi[:4]}{dates_evi[4:6]}{dates_evi[6:]}"
        base_date = datetime.strptime(formatted_date, "%Y%m%d")
        base_dates.append(base_date)
        all_dates.append(base_date)

    start_date = min(base_dates)
    end_date = max(base_dates)
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    filled_arr = arr.copy()

    def fill_missing_for_index(i, j, evi_values, n_estimators=50, random_state=0):
        evi_values_filled = evi_values.copy()

        for index_ii, evi_value in enumerate(evi_values_filled):
            if np.isnan(evi_value):
                start_window = max(0, index_ii - 15)
                end_window = min(len(evi_values_filled), index_ii + 15)

                valid_indices = ~np.isnan(evi_values_filled[start_window:end_window])
                t_valid = np.arange(len(evi_values_filled[start_window:end_window]))[valid_indices]
                y_valid = evi_values_filled[start_window:end_window][valid_indices]

                if len(t_valid) > 1:
                    X_valid = t_valid.reshape(-1, 1)
                    y_valid = y_valid.reshape(-1, 1)

                    lgbm_model = LGBMRegressor(n_estimators=n_estimators, random_state=random_state)
                    lgbm_model.fit(X_valid, y_valid.ravel())

                    # Predict the missing value
                    evi_values_filled[index_ii] = lgbm_model.predict(np.array([[index_ii]]))[0]

        return i, j, evi_values_filled

    # Prepare indices for parallel processing
    indices = [(i, j, arr[:, i, j]) for i in range(arr.shape[1]) for j in range(arr.shape[2])]

    # Process each pixel in parallel using Joblib
    results = Parallel(n_jobs=n_jobs)(delayed(fill_missing_for_index)(*index) for index in indices)

    # Update the filled_arr with the results
    for result in results:
        i, j, evi_values_filled = result
        filled_arr[:, i, j] = evi_values_filled


    return filled_arr, all_dates


## Private functions ###

def _img_metadata(evi_img):
    """
    Extract the base dates and product from the image metadata
    :param evi_img: List of image file paths
    :return: Tuple containing lists of dates, products, tile_id, and years
    """
    base_dates = []
    hls_product = []
    tile_id = os.path.basename(evi_img[0]).split('_')[2]
    years = []

    for metadates in evi_img:
        dates = os.path.basename(metadates).split('_')[0]
        convert_date = datetime.strptime(dates, '%Y%m%d')
        year = convert_date.year
        year_str = str(year)
        product = os.path.basename(metadates).split('_')[1]
        base_dates.append(dates)
        hls_product.append(product)
        years.append(year_str)

    return base_dates, hls_product, tile_id, years


def _stack_evi(evi_img):
    """
    Stack the EVI images into a 3D array
    :param evi_img:
    :return: stacked EVI images
    """
    evi_layers = []
    for img in evi_img:
        with rasterio.open(img) as src:
            evi_layer = src.read(1)
            evi_layers.append(evi_layer)

    # stack the EVI layers
    stacked_evi = np.stack(evi_layers, axis=0)
    stacked_evi = np.squeeze(stacked_evi)

    return stacked_evi


def _export_evi(base_dir, evi_img, year, evi_filled, base_dates, product, filling_technique, station_name):
    """
    Export the filled EVI images
    :param base_dir: Location of the EVI images
    :param evi_filled: Filled EVI images
    :param year: Years
    :param base_dates: Base dates
    :param product: Product
    :param filling_technique: Name of the filling technique
    :param station_name: Name of the station
    :return: None
    """
    # Open the first image to get the profile
    with rasterio.open(evi_img[0]) as src:
        band_profile = src.profile

    # Output the filled EVI images
    basedir_evi = os.path.join(base_dir, 'data_processed', station_name)

    for layer_data, current_date, product, year in zip(evi_filled, base_dates, product, year):
        # Construct the output file path for the current layer
        dir_output_date = os.path.join(basedir_evi, year, 'filling_techniques', filling_technique)
        os.makedirs(dir_output_date, exist_ok=True)
        output_filename = f"{current_date}_{product}.tif"
        output_filepath = os.path.join(dir_output_date, output_filename)

        # Write the filled EVI image to a GeoTIFF file
        with rasterio.open(output_filepath, 'w', **band_profile) as dst:
            dst.write(layer_data, 1)