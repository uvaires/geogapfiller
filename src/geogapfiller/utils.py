import os
import numpy as np
import rasterio

def _stack_raster(img_list):
    raster_layers = []
    for img in img_list:
        with rasterio.open(img) as src:
            raster_layer = src.read(1)
            raster_layers.append(raster_layer)

    # stack the EVI layers
    stacked_raster = np.stack(raster_layers, axis=0)
    stacked_raster = np.squeeze(stacked_raster)

    return stacked_raster

def _img_metadata(img_list):
    with rasterio.open(img_list[0]) as src:  # Use the first image in the list
        img_profile = src.profile  # Extract the metadata profile

    return img_profile

# Convert dates to string
def _convert_dates(dates):
    base_dates = [date.strftime("%Y%m%d") for date in dates]
    return base_dates

def _export_raster(outputdir, img_list, raster_filled, img_dates, method, pattern):
    # Open the first image to get the profile
    img_profile = _img_metadata(img_list)
    # convert dates into string
    img_dates = _convert_dates(img_dates)
    # Output the filled images
    outpupath = os.path.join(outputdir,'data_processed', method)

    for layer_data, current_dates in zip(raster_filled, img_dates):
             # Construct the output file path for the current layer
            dir_output_data = os.path.join(outpupath)
            os.makedirs(dir_output_data, exist_ok=True)
            output_filename = f"{current_dates}_{pattern}.tif"
            output_filepath = os.path.join(dir_output_data, output_filename)

            # Write the filled EVI image to a GeoTIFF file
            with rasterio.open(output_filepath, 'w', **img_profile) as dst:
                dst.write(layer_data, 1)

