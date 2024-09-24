import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from PIL import Image
import imageio
import matplotlib.image as mpimg
from PIL import Image
import math
import scipy.optimize as opt


def processing_phenocam_data(phenocam_data, interval=3):
    """
    Process the Phenocam data and filter it based on a three-day interval.

    Args:
        phenocam_data (str): Path to the Phenocam data CSV file.


    Returns:
        pd.DataFrame: Filtered Phenocam data.

    """
    # processing the CSV file
    phenocam_df = pd.read_csv(phenocam_data)
    # Convert the date column to datetime format
    phenocam_df['date'] = pd.to_datetime(phenocam_df['date'])
    # Sort the DataFrame by date
    phenocam_df = phenocam_df.sort_values(by='date')
    # Initialize a list to store filtered rows
    filtered_rows = []
    # Initialize variables to keep track of the previous date and the index
    prev_date = None
    index = 0

    # Iterate through each row in the DataFrame
    for _, row in phenocam_df.iterrows():
        # Check if it's the first row or if the current date is within a three-day interval from the previous date
        if prev_date is None or (row['date'] - prev_date).days >= interval or pd.isna(row['date']):
            # Add the row to the filtered rows list
            filtered_rows.append(row)
            # Update the previous date and index
            prev_date = row['date']
            index += 1

    # Create a new DataFrame containing only the filtered rows
    filtered_phenocam = pd.DataFrame(filtered_rows)

    return filtered_phenocam


def processing_hls_images(base_dir: str, station_name: str, polygon_path: str):
    """
    Process HLS images: remove clouds, select bands of interest

    Args:

        base_dir (str): Output directory where processed images are stored.
        station_name (str): Name of the station.

    """
    # reading the polygon data
    polygon_data = gpd.read_file(polygon_path)
    # Filter polygon data based on station name
    selected_polygon = polygon_data[polygon_data['station'] == station_name]
    # Find all HLS images in the directory
    img_dir = glob.glob(os.path.join(base_dir, '**', station_name, '**', 'smoothed_evi', '*.tif'), recursive=True)

    # Reproject the selected polygon to the UTM projection using the CRS of the first image
    with rasterio.open(img_dir[0]) as src:
        target_crs = src.crs
        selected_polygon_utm = selected_polygon.to_crs(target_crs)

    # Iterate over each image and calculate mean EVI
    mean_evi_values = []
    date_doy = []
    for img in img_dir:
        # get the date from the image
        date = os.path.basename(img).split('_')[0]
        # convert the date 'YYYYMMDD' to Day of the Year (DOY)
        doy = pd.to_datetime(date, format='%Y%m%d')
        date_doy.append(doy)
        with rasterio.open(img) as src:
            # Mask the image with the reprojected polygon and calculate mean EVI
            masked_image, _ = mask(src, selected_polygon_utm.geometry, crop=True)
            # Calculate mean EVI, ignoring NaN values
            mean_value = np.nanmean(masked_image)
            if not np.isnan(mean_value):  # Check if the mean value is not NaN
                mean_evi_values.append(mean_value)
            else:
                mean_evi_values.append(np.nan)  # Insert NaN if mean value is NaN

    return mean_evi_values, date_doy


def process_hls_original(base_dir: str, station_name: str, polygon_path: str):
    # reading the polygon data
    polygon_data = gpd.read_file(polygon_path)
    # Filter polygon data based on station name
    selected_polygon = polygon_data[polygon_data['station'] == station_name]
    # Find all HLS images in the directory
    img_dir = glob.glob(os.path.join(base_dir, '**', station_name, '**', 'spectral_index', '**', '*.tif'),
                        recursive=True)

    # Reproject the selected polygon to the UTM projection using the CRS of the first image
    with rasterio.open(img_dir[0]) as src:
        target_crs = src.crs
        selected_polygon_utm = selected_polygon.to_crs(target_crs)

    # Iterate over each image and calculate mean EVI
    date_doy = []
    original_evi_values = []
    for img in img_dir:
        # get the date from the image
        date = os.path.basename(img).split('_')[0]
        # convert the date 'YYYYMMDD' to Day of the Year (DOY)
        doy = pd.to_datetime(date, format='%Y%m%d')
        date_doy.append(doy)
        with rasterio.open(img) as src:
            # Mask the image with the reprojected polygon and calculate mean EVI
            masked_image, _ = mask(src, selected_polygon_utm.geometry, crop=True)
            # Calculate mean EVI, ignoring NaN values
            mean_value = np.nanmean(masked_image)
            if not np.isnan(mean_value):  # Check if the mean value is not NaN
                original_evi_values.append(mean_value)
            else:
                original_evi_values.append(np.nan)  # Insert NaN if mean value is NaN

    return original_evi_values, date_doy


### Sixth try

def plot_phenocam_evi_comparison_with_animation(filtered_phenocam, mean_evi_values, date_doy_mean, original_evi_values,
                                                date_doy,
                                                station_name, base_dir, dates_img, image_paths):
    """
    Plot the GCC Phenocam Mean and EVI HLS Mean with a moving vertical bar indicating the current date,
    and display the timelapse images corresponding to the selected dates.

    Args:
        filtered_phenocam (pd.DataFrame): Filtered Phenocam data.
        mean_evi_values (List[float]): List of mean EVI values.
        date_doy_mean (List[int]): List of Day of Year corresponding to mean EVI values.
        original_evi_values (List[float]): List of mean EVI values from original HLS images.
        date_doy (List[int]): List of Day of Year corresponding to original EVI values.
        station_name (str): Name of the station.
        base_dir (str): Base directory.
        dates_img (pd.Series): Series containing image dates.
        image_paths (List[str]): List of paths to timelapse images.

    Returns:
        anim: Animation object.
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

    # Plot the GCC Phenocam Mean on the first subplot (ax1)
    ax1.plot(filtered_phenocam['date'], filtered_phenocam['gcc_mean'], color='blue', marker='o', linestyle='None',
             markersize=5,
             label='GCC Phenocam Mean')
    ax1.set_xlabel('DOY', fontname='Times New Roman', fontsize=12)
    ax1.set_ylabel('GCC Mean', color='black', fontname='Times New Roman', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper left', fontsize=10)

    # Create a secondary y-axis for the EVI values
    ax1_evi = ax1.twinx()
    ax1_evi.plot(date_doy_mean, mean_evi_values, color='green', marker='o', markersize=5, label='EVI HLS Synthetic')
    ax1_evi.plot(date_doy, original_evi_values, color='red', marker='^', linestyle='None', markersize=7,
                 label='EVI HLS Original')
    ax1_evi.set_ylabel('EVI Mean', color='black', fontname='Times New Roman', fontsize=12)
    ax1_evi.tick_params(axis='y', colors='black')
    ax1_evi.legend(loc='upper right', fontsize=10)

    # Initialize the vertical line for current date on ax1
    line_ax1 = ax1.axvline(dates_img.iloc[0], color='black')

    # Function to update the plot with the vertical line at the specified date on ax1
    def update_plot(frame):
        # Update the position of the vertical line on ax1
        line_ax1.set_xdata([dates_img.iloc[frame], dates_img.iloc[frame]])

        # Clear previous date annotation on ax1
        for annotation in ax1.texts:
            annotation.remove()

        # Add new text annotation for current date on ax1
        ax1.text(dates_img.iloc[frame], ax1.get_ylim()[1], dates_img.iloc[frame].strftime('%Y-%m-%d'), ha='center',
                 va='bottom')

        # Clear previous image on ax2
        ax2.clear()

        # Plot the current image on ax2
        img = Image.open(image_paths[frame])
        ax2.imshow(img)
        ax2.axis('off')

        plt.draw()

    # Create animation
    anim = FuncAnimation(fig, update_plot, frames=len(dates_img), interval=200)

    # create a directory to store the animation
    anim_dir = os.path.join(base_dir, 'animations', station_name)
    os.makedirs(anim_dir, exist_ok=True)

    # Save animation as GIF
    gif_path = os.path.join(anim_dir, f'{station_name}.gif')
    anim.save(gif_path, writer='pillow')

    # Save animation as MP4
    mp4_path = os.path.join(anim_dir, f'{station_name}.mp4')
    anim.save(mp4_path, writer='ffmpeg')

    return anim


def exctrat_phenocam_dates(base_dir: str, station_name: str):
    """
    Extract the dates of the Phenocam images.

    Returns:
        List[str]: List of dates in 'YYYYMMDD' format.
    """
    # read image paths
    image_paths = glob.glob(os.path.join(base_dir, 'phenocam_imgs', station_name, '*.jpg'), recursive=True)
    # extract the dates of the images
    dates_img = []
    for img in image_paths:
        # Split the filename by underscores
        parts = os.path.basename(img).split('_')
        # Select indices for year, month, and day
        year = parts[1]
        month = parts[2]
        day = parts[3]
         # Format the date
        date = f"{year}{month}{day}"
        dates_img.append(date)

    return dates_img, image_paths
