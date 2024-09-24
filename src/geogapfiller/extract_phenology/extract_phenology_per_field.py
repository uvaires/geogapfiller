import geopandas as gpd
import pandas as pd
import glob
import os
import rasterio
import numpy as np
from rasterio.mask import mask
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math
import pickle
from sklearn.linear_model import ElasticNet
import joblib

# Define paths and variables
polygon_path = r'C:\crop_phenology\phenocam_shapefile\field_test.shp'
soybeans_planting_model_path = r'C:\crop_phenology\predictions\elastic_net_model_planting_soybeans.pkl'
soybeans_emergence_model_path = r'C:\crop_phenology\predictions\elastic_net_model_emergence_soybeans.pkl'
soybeans_harvest_model_path = r'C:\crop_phenology\predictions\elastic_net_model_harvest_soybeans.pkl'
corn_planting_model_path = r'C:\crop_phenology\predictions\elastic_net_model_planting_corn.pkl'
corn_emergence_model_path = r'C:\crop_phenology\predictions\elastic_net_model_emergence_corn.pkl'
corn_harvest_model_path = r'C:\crop_phenology\predictions\elastic_net_model_harvest_corn.pkl'

base_dir = r'C:\crop_phenology'
station_name = 'NEON.D06.KONA.DP1.00033'
fields = list(range(1, 23))
year = '2023'
vegetation_index = 'mean_evi'
interval = 1


# Asymmetric double sigmoid model function
def _asymmetric_dbl_sigmoid_model(t, Vb, Va, p, Di, q, Dd):
    """A double logistic model, as in Zhong et al 2016"""
    return Vb + 0.5 * Va * (np.tanh(p * (t - Di)) - np.tanh(q * (t - Dd)))


def convert_doy_to_date(year, doy):
    start_date = f'{year}-01-01'
    start_date = pd.to_datetime(start_date)
    target_date = start_date + pd.to_timedelta(doy - 1, unit='d')
    date_str = target_date.strftime('%Y%m%d')
    return date_str


# Load the saved models using joblib
soybeans_planting_model = joblib.load(soybeans_planting_model_path)
soybeans_emergence_model = joblib.load(soybeans_emergence_model_path)
soybeans_harvest_model = joblib.load(soybeans_harvest_model_path)
corn_planting_model = joblib.load(corn_planting_model_path)
corn_emergence_model = joblib.load(corn_emergence_model_path)
corn_harvest_model = joblib.load(corn_harvest_model_path)

# Verify that the loaded models are instances of ElasticNet
print(isinstance(soybeans_planting_model, ElasticNet))  # Should print True
print(isinstance(soybeans_emergence_model, ElasticNet))  # Should print True
print(isinstance(soybeans_harvest_model, ElasticNet))  # Should print True
print(isinstance(corn_planting_model, ElasticNet))  # Should print True
print(isinstance(corn_emergence_model, ElasticNet))  # Should print True
print(isinstance(corn_harvest_model, ElasticNet))  # Should print True


# Reading the polygon data
polygon_data = gpd.read_file(polygon_path)

# Find all HLS images in the directory
img_dir = glob.glob(os.path.join(base_dir, '**', station_name, year, 'smoothed_evi', '*.tif'), recursive=True)

# Initialize a list to store the results
results = []

# Iterate over each field
for field in fields:
    # Filter polygon data based on the current field
    selected_polygon = polygon_data[polygon_data['field'] == field]

    if selected_polygon.empty:
        print(f"No polygon found for field {field}")
        continue

    # Get the crop type for the current field
    crop_type = selected_polygon['crop'].values[0]

    # Reproject the selected polygon to the UTM projection using the CRS of the first image
    with rasterio.open(img_dir[0]) as src:
        target_crs = src.crs
        selected_polygon_utm = selected_polygon.to_crs(target_crs)

    # Initialize lists for mean EVI values and DOY
    mean_evi_values = []
    date_doy = []

    # Iterate over each image and calculate mean EVI
    for img in img_dir:
        # Get the date from the image
        date = os.path.basename(img).split('_')[0]
        # Convert the date 'YYYYMMDD' to Day of the Year (DOY)
        doy = pd.to_datetime(date, format='%Y%m%d').timetuple().tm_yday
        date_doy.append(doy)

        with rasterio.open(img) as src:
            # Mask the image with the reprojected polygon and calculate mean EVI
            masked_image, _ = mask(src, selected_polygon_utm.geometry, crop=True)
            # Calculate mean EVI, ignoring NaN values
            mean_value = np.nanmean(masked_image)
            mean_evi_values.append(mean_value if not np.isnan(mean_value) else np.nan)

    # Create a DataFrame for the EVI time series
    df = pd.DataFrame({'date': pd.to_datetime([convert_doy_to_date(year, doy) for doy in date_doy]),
                       vegetation_index: mean_evi_values})
    df = df.set_index('date')

    # Resample the time series to an equal interval with by max value and interpolate
    vi_time_serie = (df.loc[:, vegetation_index].resample(f'{interval}d').max()
                     .interpolate(method="time", limit_direction='both'))

    xdata = np.array(range(vi_time_serie.shape[0]))
    ydata = np.array(vi_time_serie)

    # Initial guess for [Vb, Va, p, Di, q, Dd]
    p0 = [0.2, 0.6, 0.05, 50 / interval, 0.05, 130 / interval]

    # Lower and upper bounds for [Vb, Va, p, Di, q, Dd]
    bounds = ([0.0, 0.2, -np.inf, 0, 0, 0],
              [0.5, 0.8, np.inf, xdata.shape[0], 0.4, xdata.shape[0]])

    # Fit the model
    popt, pcov = opt.curve_fit(_asymmetric_dbl_sigmoid_model, xdata=xdata, ydata=ydata,
                               p0=p0, bounds=bounds, method='trf', maxfev=10000)
    if np.any(np.isinf(pcov)):
        raise RuntimeError("Covariance of the parameters could not be estimated")

    Vb, Va, p, Di, q, Dd = popt

    # Apply the parameters
    vi_fitted = _asymmetric_dbl_sigmoid_model(xdata, *popt)

    # Calculate phenological dates
    D1 = Di + np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D2 = Di - np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D3 = Dd + np.divide(1, q) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D4 = Dd - np.divide(1, q) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)

    # Convert dates to the desired format (YYYYMMDD)
    phenological_dates = {
        'field': field,
        'crop': crop_type,
        'D1': vi_time_serie.index[int(round(D1))].strftime('%Y%m%d'),
        'Di': vi_time_serie.index[int(round(Di))].strftime('%Y%m%d'),
        'D2': vi_time_serie.index[int(round(D2))].strftime('%Y%m%d'),
        'D3': vi_time_serie.index[int(round(D3))].strftime('%Y%m%d'),
        'Dd': vi_time_serie.index[int(round(Dd))].strftime('%Y%m%d'),
        'D4': vi_time_serie.index[int(round(D4))].strftime('%Y%m%d')
    }

    # Append the phenological dates to the results list
    results.append(phenological_dates)

    # Plot the time series
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[vegetation_index], marker='.', lw=0, label='Raw VI')
    ax.plot(vi_time_serie.index, vi_fitted, label=f'Fitted VI')
    ax.set_ylim(0, 1)
    ax.set_ylabel(vegetation_index)

    colors = ['m', 'g', 'yellow', 'c', 'orange', 'b']
    labels = ['D1', 'Di', 'D2', 'D3', 'Dd', 'D4']
    for i, d in enumerate([D1, Di, D2, D3, Dd, D4]):
        ax.axvline(vi_time_serie.index[int(round(d))], 0, vi_fitted[int(round(d))], color=colors[i],
                   label=f'{labels[i]}: {str(vi_time_serie.index[int(round(d))].date())}', ls='--')

    # Save the plot
    plot_path = os.path.join(base_dir, 'phenology_field_dates', station_name)
    os.makedirs(plot_path, exist_ok=True)

    # Construct the complete file path
    file_path = os.path.join(plot_path, f'{station_name}_{year}_field_{field}.png')

    ax.legend()
    # Save the plot to the specified file path
    plt.savefig(file_path)
    plt.show()

# Convert the results list to a DataFrame
phenological_dates_df = pd.DataFrame(results)

# Convert columns to datetime format
date_columns = ['D1', 'Di', 'D2', 'D3', 'Dd', 'D4']
phenological_dates_df[date_columns] = phenological_dates_df[date_columns].apply(pd.to_datetime, format='%Y%m%d')

# Extract Day of Year (DOY)
for col in date_columns:
    phenological_dates_df[col + '_doy'] = phenological_dates_df[col].dt.dayofyear


# Define a function to apply the models and predict dates
def predict_dates(row, crop):
    features = np.array([[row['D1_doy'], row['Di_doy'], row['D2_doy'], row['D3_doy'], row['Dd_doy'], row['D4_doy']]])
    if crop == 'Soybeans':
        planting_date = soybeans_planting_model.predict(features)[0]
        emergence_date = soybeans_emergence_model.predict(features)[0]
        harvest_date = soybeans_harvest_model.predict(features)[0]
    elif crop == 'Corn':
        planting_date = corn_planting_model.predict(features)[0]
        emergence_date = corn_emergence_model.predict(features)[0]
        harvest_date = corn_harvest_model.predict(features)[0]
    else:
        planting_date, emergence_date, harvest_date = np.nan, np.nan, np.nan

    return planting_date, emergence_date, harvest_date

# Function to round values
def round_values(planting, emergence, harvest):
    return round(planting), round(emergence), round(harvest)

# Apply the models and add the predicted dates to the DataFrame
phenological_dates_df[['planting', 'emergence', 'harvest']] = phenological_dates_df.apply(
    lambda row: round_values(*predict_dates(row, row['crop'])), axis=1, result_type='expand')

# Apply the function and add the dates as integer columns to the DataFrame
phenological_dates_df[['planting_dates', 'emergence_dates', 'harvest_dates']] = phenological_dates_df.apply(
    lambda row: pd.Series([
        convert_doy_to_date(year, row['planting']),
        convert_doy_to_date(year, row['emergence']),
        convert_doy_to_date(year, row['harvest'])
    ]),
    axis=1
)


# Merge the polygon data with the phenological dates
merged_data = polygon_data.merge(phenological_dates_df, on='field')

# Convert the date columns to string format
date_columns = ['D1', 'Di', 'D2', 'D3', 'Dd', 'D4']
merged_data[date_columns] = merged_data[date_columns].astype(str)

date_columns = ['planting_dates', 'emergence_dates', 'harvest_dates']
merged_data[date_columns] = merged_data[date_columns].apply(pd.to_datetime, errors='coerce')

# Step 3: Format date columns with '-' separator
merged_data[date_columns] = merged_data[date_columns].apply(lambda x: x.dt.strftime('%Y-%m-%d'))


# Export to shapefile
output_shapefile = r'C:\crop_phenology\phenocam_shapefile\phenological_dates_field\phenological_dates_per_field.shp'
merged_data.to_file(output_shapefile)

