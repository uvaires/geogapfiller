import os
import glob
import rasterio
from datetime import datetime, timedelta
from collections import Counter
from joblib import Parallel, delayed
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def predicted_img(base_dir: str, station_name: str, predicted_interval: int, year: str):
    '''
    Predict the daily images for the specified interval
    :param base_dir: base directory
    :param station_name:    station name
    :param predicted_interval: interval for prediction the images
    :return: none

    '''

    # Read the image path
    evi_img_original = glob.glob(os.path.join(base_dir, '**', station_name, year, '**', 'spectral_index', '**', '*.tif'), recursive=True)
    img_layers, img_profile, product, tile_id = _load_img_layers(evi_img_original)

    # Stack the image layers
    stacked_bands = np.stack(img_layers, axis=0)

    # Fill the missing values in the image for every 3 days
    img_filled, date_range = predict_daily_imgs(evi_img_original, stacked_bands, predicted_interval, n_job=-1)

    # Predicted dates
    base_dates_pred = _convert_dates(date_range)
    # Extract years as strings
    year_str_list = extract_years(base_dates_pred)

    # Export the predicted images
    _export_img_daily(img_filled, base_dates_pred, base_dir, img_profile,  year_str_list, station_name)

def predict_daily_imgs(img_paths, arr, interval=3, n_job=-1):
    base_dates, all_dates,_ = _extract_dates(img_paths)

    dates_counter = Counter(all_dates)
    start_date = min(base_dates)
    end_date = max(base_dates)

    # Calculate the date range based on the specified interval
    date_range = [start_date + timedelta(days=i) for i in range(0, (end_date - start_date).days + 1, interval)]

    original_arr = np.full((len(date_range), arr.shape[1], arr.shape[2]), np.nan)

    for i, date in enumerate(all_dates):
        if start_date <= date <= end_date:
            try:
                date_index = date_range.index(date)

                if dates_counter[date] > 1:
                    existing_values = original_arr[date_index, :, :]
                    new_values = arr[i, :, :]
                    combined_values = np.nanmean([existing_values, new_values], axis=0)
                else:
                    combined_values = arr[i, :, :]

                original_arr[date_index, :, :] = combined_values
            except ValueError:
                pass  # Date is outside the specified interval

    tasks = [(i, j, original_arr[:, i, j]) for i in range(original_arr.shape[1]) for j in range(original_arr.shape[2])]
    results = Parallel(n_jobs=n_job)(delayed(_predict_missing_for_index)(*task) for task in tasks)

    for result in results:
        i, j, image_values_filled = result
        if np.any(~np.isnan(original_arr[:, i, j])):
            clipped_values = np.clip(image_values_filled, 0, 1)
            original_arr[:, i, j] = clipped_values

    return original_arr, date_range



## Privite functions ##
# Read image profile
def _read_image(image_path):
    with rasterio.open(image_path) as src:
        img_layer = src.read(1)
        img_profile = src.profile
    return img_layer, img_profile

# Function to extract years as strings from the list of date strings
def extract_years(base_dates):
    years = [date[:4] for date in base_dates]  # Extract the first 4 characters (YYYY) from each date string
    return years

# Convert date string to datetime and return both date string and datetime
def _extract_dates(img_paths):
    base_dates = []
    all_dates = []
    years = []

    for dates in img_paths:
        # Extract date from the file name
        dates = os.path.basename(dates).split('_')[0]
        convert_date = datetime.strptime(dates, '%Y%m%d')
        year = convert_date.year
        year_str = str(year)
        formatted_date = f"{dates[:4]}{dates[4:6]}{dates[6:]}"
        base_date = datetime.strptime(formatted_date, "%Y%m%d")
        base_dates.append(base_date)
        all_dates.append(base_date)
        years.append(year_str)

    return base_dates, all_dates, years

# Adjust a polynomial regression model to fill missing values within a window of 15 days
def _process_missing_values(index_ii, image_values_filled, degree=2, n_jobs=-1):
    if np.isnan(image_values_filled[index_ii]):
        window_size = 15
        start_window = max(0, index_ii - window_size)
        end_window = min(len(image_values_filled), index_ii + window_size)

        valid_indices = ~np.isnan(image_values_filled[start_window:end_window])
        if not np.any(valid_indices):
            # Expand the window until valid data is found or end of array is reached
            while end_window < len(image_values_filled):
                end_window += 1
                valid_indices = ~np.isnan(image_values_filled[start_window:end_window])
                if np.any(valid_indices):
                    break

        X_valid = np.arange(start_window, end_window)[valid_indices].reshape(-1, 1)
        y_valid = image_values_filled[start_window:end_window][valid_indices].reshape(-1, 1)

        if len(X_valid) > 1:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X_valid)

            model = LinearRegression(n_jobs=n_jobs)
            model.fit(X_poly, y_valid)

            pred_value = model.predict(poly.transform(np.array([[index_ii]])))
            image_values_filled[index_ii] = np.clip(pred_value[0][0], 0, 1)



# Interate over the image values to fill missing values
def _fill_missing_for_index(i, j, image_values):
    image_values_filled = image_values.copy()

    for index_ii in range(len(image_values_filled)):
        _process_missing_values(index_ii, image_values_filled)

    return i, j, image_values_filled

# Function to fill gaps in the image values using polynomial regression and return the filled array
def _fill_gaps_polynomial(img_paths, arr, n_job=-1):
    base_dates, all_dates,_ = _extract_dates(img_paths)

    start_date = min(base_dates)
    end_date = max(base_dates)
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    filled_arr = arr.copy()

    indices = [(i, j, arr[:, i, j]) for i in range(arr.shape[1]) for j in range(arr.shape[2])]
    results = Parallel(n_jobs=n_job)(delayed(_fill_missing_for_index)(*index) for index in indices)

    for result in results:
        i, j, evi_values_filled = result
        filled_arr[:, i, j] = evi_values_filled

    return filled_arr, all_dates

# Function to predict missing values in the image values and return the filled array
def _predict_missing_for_index(i, j, image_values):
    image_values_filled = image_values.copy()

    for index_ii in range(len(image_values_filled)):
        _process_missing_values(index_ii, image_values_filled)

    return i, j, image_values_filled


# Read the image information and return the image layers, band profile, and product
def _load_img_layers(img_paths):
    img_layers = []
    band_profile = None
    product = []

    for img_path in img_paths:
        print(img_path)
        product_hls = os.path.basename(img_path).split('_')[1]
        tile_id = os.path.basename(img_path).split('_')[2]
        product.append(product_hls)
        img_layer, band_profile = _read_image(img_path)
        img_layers.append(img_layer)

    return img_layers, band_profile, product, tile_id

# Convert the dates to string format
def _convert_dates(dates):
    base_dates = [date.strftime("%Y%m%d") for date in dates]
    return base_dates


# Function to save the predicted images to drive
def _export_img_daily(img_predicted, base_dates, output_dir, img_profile, year, station_name):
    for layer_idx, (layer_data, current_date, years) in enumerate(zip(img_predicted, base_dates, year)):
        print(current_date)
        dir_output_date = os.path.join(output_dir,'data_processed', station_name, years, 'img_predicted')
        os.makedirs(dir_output_date, exist_ok=True)
        output_filepath = os.path.join(dir_output_date, f"{current_date}_evi.tif")
        img_profile['nodata'] = np.nan
        img_profile['compress'] = 'lzw'

        with rasterio.open(output_filepath, 'w', **img_profile) as dst:
            dst.write(layer_data, 1)

