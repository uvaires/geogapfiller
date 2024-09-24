import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import scipy.optimize as opt
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import pearsonr
import math
from sklearn.metrics import mean_squared_error, r2_score


def _calculate_metrics(observed, predicted):
    # Convert lists to NumPy arrays if they are not already
    observed = np.array(observed)
    predicted = np.array(predicted)

    # Ignore NaN values in observed and predicted arrays
    observed_non_nan = observed[~np.isnan(observed) & ~np.isnan(predicted)]
    predicted_non_nan = predicted[~np.isnan(observed) & ~np.isnan(predicted)]

    # Calculate mean squared error
    mse = np.nanmean((observed_non_nan - predicted_non_nan) ** 2)

    # Calculate root mean squared error
    rmse = np.sqrt(mse)

    # Calculate mean absolute error
    mae = np.nanmean(np.abs(observed_non_nan - predicted_non_nan))

    # Calculate BIAS
    bias = np.nanmean(predicted_non_nan - observed_non_nan)

    # Calculate Pearson correlation coefficient (ignoring NaN values)
    r = np.corrcoef(observed_non_nan, predicted_non_nan)[0, 1]
    r2 = r ** 2

    return rmse, mae, r2, bias


# Define the model equations
def model(params, hls_times, phenocam_times, phenocam_values):
    alpha, beta, lambda_, b = params
    T = lambda_ * (hls_times + beta)
    interpolated_phenocam_values = np.interp(T, phenocam_times, phenocam_values)
    predicted_hls = alpha * interpolated_phenocam_values + b
    return predicted_hls


# Objective function to minimize (Mean Squared Deviation)
def objective_function(params, hls_times, hls_values, phenocam_times, phenocam_values):
    predicted_hls = model(params, hls_times, phenocam_times, phenocam_values)
    msd = np.mean((hls_values - predicted_hls) ** 2)
    return msd

def calibrate_hls_timeseries(dataframe_path: str, phenocam: str, year: str, base_dir: str):
    # Load the HLS EVI2 and PhenoCam GCC data

    df = pd.read_excel(dataframe_path)

    # Example: hls_data and phenocam_data are pandas DataFrames with 'time' and 'value' columns
    hls_times = df['doy_hls']  # HLS time series (days)
    hls_values = df['EVI']  # HLS EVI2 values (mean of ROI)
    phenocam_times = df['doy_phenocam']  # PhenoCam time series (days)
    phenocam_values = df['GCC']  # PhenoCam GCC values (mean of ROI)

    # Initial guess for the parameters
    initial_params = [1.0, 0.0, 1.0, 0.0]

    # Minimize the objective function
    result = minimize(objective_function, initial_params, args=(hls_times, hls_values, phenocam_times, phenocam_values))
    best_params = result.x

    # Extract the best parameters
    alpha, beta, lambda_, b = best_params

    # Calculate the predicted HLS using the best parameters
    predicted_hls = model(best_params, hls_times, phenocam_times, phenocam_values)

    # Calculate correlation coefficient
    R, p_value = pearsonr(hls_values, predicted_hls)

    # Calculate RMSE, MAE, and R²
    rmse, mae, r2, bias = _calculate_metrics(hls_values, predicted_hls)

    # Define the calibration function with time transformation
    def calibrate_hls(phenocam_value, time, alpha, beta, lambda_, b):
        T = lambda_ * (time + beta)
        interpolated_phenocam_value = np.interp(T, phenocam_times, phenocam_values)

        return alpha * interpolated_phenocam_value + b

    # Load the single-column DataFrame
    hls_df = pd.DataFrame({
        'time': hls_times,  # Time series (days)
        'value': hls_values  # HLS EVI2 values
    })

    # Apply the calibration function to the DataFrame

    hls_df['calibrated_evi'] = hls_df.apply(lambda row: calibrate_hls(row['value'], row['time'], alpha, beta, lambda_, b),
                                        axis=1)

    # export the calibrated HLS EVI2 values to an Excel file
    folder_path = os.path.join(base_dir, 'calibrated_hls', 'calibrated_values')
    os.makedirs(folder_path, exist_ok=True)

    output_file = os.path.join(folder_path, f'{phenocam}_{year}_hls_calibrated.xlsx')
    hls_df.to_excel(output_file, index=False)

    # Plotting observed vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted_hls, hls_values, color='blue', label='Predicted vs Actual')
    plt.plot([min(predicted_hls), max(hls_values)], [min(predicted_hls), max(hls_values)], color='red', linestyle='--',
         label='Ideal fit')

    # Add legend with metrics
    plt.legend(title=f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}\nBias: {bias:.2f}', loc='upper left')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Leave-One-Out Cross-Validation Results')
    # Save the plot
    plot_path = os.path.join(base_dir, 'calibrated_hls', 'plots')
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f'{phenocam}_{year}_calibration_plot.png'), dpi=300)


    # Save the plot
    plot_path = os.path.join(base_dir, 'calibrated_hls', 'plots')
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f'{phenocam}_{year}_calibration_plot.png'), dpi=300)
    plt.show()


def _asymmetric_dbl_sigmoid_model(t: np.array, Vb: float, Va: float, p: float,
                                 Di: float, q: float, Dd: float):
    """A double logistic model, as in Zhong et al 2016"""
    return Vb + 0.5 * Va * (np.tanh(p * (t - Di)) - np.tanh(q * (t - Dd)))


# Assuming the convert_doy_to_date function is defined as follows:
def convert_doy_to_date(year, doy):
    start_date = f'{year}-01-01'
    start_date = pd.to_datetime(start_date)
    target_date = start_date + pd.to_timedelta(doy - 1, unit='d')
    date_str = target_date.strftime('%Y%m%d')
    return date_str


def extract_phenological_dates(file_path, base_dir, station_name, year):
    """
    Extract phenological dates from a time series of a vegetation index.

    Args:
        file_path (str): Path to the Excel file containing the time series data.

    Returns:
        dict: Dictionary containing the phenological dates.

    """
    df = pd.read_excel(file_path)
    # Apply the function to the 'time' column
    df['dates'] = df['time'].apply(lambda doy: convert_doy_to_date(year, doy))

    # Convert the resulting date strings to datetime objects
    df['date'] = pd.to_datetime(df['dates'])

    # Set the start and end dates, and the vegetation index
    vegetation_index = 'calibrated_evi'
    interval = 1

    # Resample the time series to an equal interval with by max value and interpolate
    df.index = df.date
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

    # Reference: Zhong, Gong and Biging (2012)
    D1 = Di + np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D2 = Di - np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D3 = Dd + np.divide(1, q) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D4 = Dd - np.divide(1, q) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)

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

    # save the plot
    plot_path = os.path.join(base_dir, 'phenology_dates', station_name, 'dates')
    os.makedirs(plot_path, exist_ok=True)

    # Construct the complete file path
    file_path = os.path.join(plot_path, f'{station_name}_{year}.png')


    ax.legend()
    # Save the plot to the specified file path
    plt.savefig(file_path)
    plt.show()


    # Convert dates to the desired format (YYYYMMDD)
    phenological_dates = {
        'D1': vi_time_serie.index[int(round(D1))].strftime('%Y%m%d'),
        'Di': vi_time_serie.index[int(round(Di))].strftime('%Y%m%d'),
        'D2': vi_time_serie.index[int(round(D2))].strftime('%Y%m%d'),
        'D3': vi_time_serie.index[int(round(D3))].strftime('%Y%m%d'),
        'Dd': vi_time_serie.index[int(round(Dd))].strftime('%Y%m%d'),
        'D4': vi_time_serie.index[int(round(D4))].strftime('%Y%m%d')
    }

    # Export the phenological dates to a CSV file
    phenological_dates_df = pd.DataFrame(phenological_dates, index=[0])
    phenological_dates_df.to_csv(os.path.join(plot_path, f'{station_name}_{year}_phenological_dates.csv'), index=False)


