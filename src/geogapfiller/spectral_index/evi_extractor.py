# Import libraries
import os
import glob
import pyproj
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_evi_coords(base_dir: str, lat: float, lon: float, year: str, station_name: str):
    # Read the images
    evi_img = glob.glob(os.path.join(base_dir,'**', station_name, year, 'smoothed_evi', '*.tif'), recursive=True)
    print(evi_img)

    # Stack the EVI layers using np.dstack
    evi_layers = _extract_img_profile(evi_img)
    stacked_evi = np.dstack(evi_layers)
    stacked_evi.shape

    # Specified latitude and longitude
    lat, lon = lat, lon

    # Reproject the coordinates from WG84 to UTM
    coords_preproj = _reproject_coords(evi_img, lon, lat)
    print(coords_preproj)

    # Open the first EVI raster to obtain its CRS and transform
    with rasterio.open(evi_img[0]) as src:
        crs = src.crs
        transform = src.transform

    # Transform latitude and longitude to pixel coordinates (row and column)
    col, row = ~transform * (coords_preproj[0], coords_preproj[1])
    # Extract the EVI time series at the specified pixel location
    evi_values = stacked_evi[int(row), int(col), :]
    print(evi_values)

    # Create an empty list to store the extracted dates
    evi_dates = _extract_dates(evi_img)

    # Plot the complete EVI time series
    plt.scatter(evi_dates, evi_values, label='EVI Values', marker='o', color='green')

    plt.title(f'EVI Time Series for Lat: {lat}, Lon: {lon}', fontname='Times New Roman', fontsize=16)
    plt.xlabel('Time', fontname='Times New Roman', fontsize=14)
    plt.ylabel('EVI Value', fontname='Times New Roman', fontsize=14)
    plt.xticks(rotation=45, fontname='Times New Roman', fontsize=12)
    plt.xticks(evi_dates[::4])
    plt.yticks(fontname='Times New Roman', fontsize=12)
    plt.legend()
    plt.grid(True)  # Add grid lines

    # Set the size
    fig = plt.gcf()
    fig.set_size_inches(10, 6)

    ouput_dir = os.path.join(base_dir,'data_processed', station_name, 'validation_plots')
    os.makedirs(ouput_dir, exist_ok=True)
    save_plot = os.path.join(ouput_dir, f'evi_smooth_{year}.png')
    # Save the plot to the specified output path
    plt.savefig(save_plot, bbox_inches='tight', dpi=300)
    plt.show()


### Private function

def _reproject_coords(input_raster, lon, lat):
    # Open the input raster to obtain its coordinate reference system (CRS)
    with rasterio.open(input_raster[0]) as src:
        proj = src.crs.to_epsg()  # Get the EPSG code of the raster's CRS

    # Define the original coordinate reference system as WGS 84 (EPSG:4326)
    original_proj = pyproj.CRS("EPSG:4326")

    # Create the target coordinate reference system based on the EPSG code of the raster's CRS
    target_proj = pyproj.CRS("EPSG:" + str(proj))

    # Create a transformer object for coordinate transformation
    transformer = pyproj.Transformer.from_crs(original_proj, target_proj, always_xy=True)

    # Use the transformer to convert the input latitude and longitude to UTM coordinates
    lon_utm, lat_utm = transformer.transform(lon, lat)

    # Return the UTM easting (X) and northing (Y) coordinates
    return lon_utm, lat_utm


# Create an empty list to store the loaded layers
def _extract_img_profile(evi_img):
    evi_layers = []
    # Loop through the EVI images and load them

    for evi_path in evi_img:
        with rasterio.open(evi_path) as src:
            evi_layer = src.read(1)
            evi_layers.append(evi_layer)

    return evi_layers


def _extract_dates(evi_img):
    evi_dates = []
    # Extract dates from the filenames and add them to the list
    for dates in evi_img:
        date = os.path.basename(dates).split('_')[0].split('.')[0]
        evi_dates.append(date)
    return evi_dates


