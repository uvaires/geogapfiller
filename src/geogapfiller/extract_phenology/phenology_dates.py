import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import pandas as pd
import os
import matplotlib.dates as mdates

# Set Times New Roman as the font
plt.rcParams['font.family'] = 'Times New Roman'


def _asymmetric_dbl_sigmoid_model(t: np.array, Vb: float, Va: float, p: float, Di: float, q: float, Dd: float):
    """A double logistic model, as in Zhong et al 2016"""
    return Vb + 0.5 * Va * (np.tanh(p * (t - Di)) - np.tanh(q * (t - Dd)))


def extract_phenological_dates(file_path, base_dir, station_name, year, database_name):
    df = pd.read_excel(file_path)
    df['date'] = pd.to_datetime(df['date'])

    vegetation_index = 'EVI'
    interval = 1
    df.index = df.date
    vi_time_serie = (df.loc[:, vegetation_index].resample(f'{interval}d').max()
                     .interpolate(method="time", limit_direction='both'))

    xdata = np.array(range(vi_time_serie.shape[0]))
    ydata = np.array(vi_time_serie)

    p0 = [0.2, 0.6, 0.05, 50 / interval, 0.05, 130 / interval]
    bounds = ([0.0, 0.2, -np.inf, 0, 0, 0], [0.5, 0.8, np.inf, xdata.shape[0], 0.4, xdata.shape[0]])

    popt, pcov = opt.curve_fit(_asymmetric_dbl_sigmoid_model, xdata=xdata, ydata=ydata, p0=p0, bounds=bounds,
                               method='trf', maxfev=10000)
    if np.any(np.isinf(pcov)):
        raise RuntimeError("Covariance of the parameters could not be estimated")

    Vb, Va, p, Di, q, Dd = popt
    vi_fitted = _asymmetric_dbl_sigmoid_model(xdata, *popt)

    D1 = Di + np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D2 = Di - np.divide(1, p) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D3 = Dd + np.divide(1, q) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)
    D4 = Dd - np.divide(1, q) * np.log((math.sqrt(6) - math.sqrt(2)) / 2)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df[vegetation_index], marker='.', lw=0, label='Raw GCC', color='gray')
    ax.plot(vi_time_serie.index, vi_fitted, label='Fitted GCC', color='blue')
    ax.set_ylim(0, 1)
    ax.set_ylabel('GCC', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_title(f'Phenological Stages', fontsize=16, fontweight='bold')
    ax.grid(True)

    colors = ['green', 'blue', 'black', 'purple', 'orange', 'red']
    labels = ['Greenup', 'MidGreenup', 'Maturity', 'Senescence', 'MidGreendown', 'Dormancy']
    for i, d in enumerate([D1, Di, D2, D3, Dd, D4]):
        ax.axvline(vi_time_serie.index[int(round(d))], 0, vi_fitted[int(round(d))], color=colors[i],
                   label=f'{labels[i]}: {str(vi_time_serie.index[int(round(d))].date())}', ls='--', lw=1.5)

    ax.legend(frameon=True)

    plot_path = os.path.join(base_dir, 'phenology_dates', database_name, station_name)
    os.makedirs(plot_path, exist_ok=True)
    file_path = os.path.join(plot_path, f'{station_name}_{year}.png')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    for label in ax.get_xticklabels():
        label.set_rotation(0)  # Set rotation to 0 degrees for horizontal text

    plt.savefig(file_path, dpi=300)
    plt.show()

    phenological_dates = {
        'D1': vi_time_serie.index[int(round(D1))].strftime('%Y%m%d'),
        'Di': vi_time_serie.index[int(round(Di))].strftime('%Y%m%d'),
        'D2': vi_time_serie.index[int(round(D2))].strftime('%Y%m%d'),
        'D3': vi_time_serie.index[int(round(D3))].strftime('%Y%m%d'),
        'Dd': vi_time_serie.index[int(round(Dd))].strftime('%Y%m%d'),
        'D4': vi_time_serie.index[int(round(D4))].strftime('%Y%m%d')
    }

    phenological_dates_df = pd.DataFrame(phenological_dates, index=[0])
    phenological_dates_df.to_csv(os.path.join(plot_path, f'{station_name}_{year}.csv'), index=False)


# Read the combined DataFrame
#data_frame = 'arsltarmdcr'
years = ['2021','2022', '2023']
database_name = 'predited_hls'
station_name = 'arsltarmdcr'

for year in years:
    df = rf'C:\crop_phenology\dataframe\{station_name}_{year}_hls.xlsx'

    base_dir = r'C:\crop_phenology'

    # Plot the phenological dates
    extract_phenological_dates(df, base_dir, station_name, year, database_name)