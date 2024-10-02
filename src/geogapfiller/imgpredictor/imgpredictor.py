from abc import ABC, abstractmethod
import numpy as np
from datetime import timedelta
from collections import Counter
from geogapfiller import utils
from geogapfiller.gapfiller import gapfiller

# Abstract Base Class for Fillers
class Predictor(ABC):
    @abstractmethod
    def img_predictor(self, list_raster, base_date, n_jobs) -> np.array:
        pass


# Concrete Class for MedianFiller
class MedianPredictor(Predictor):
    def __init__(self, window_size=1):
        self.window_size = window_size

    def img_predictor(self, original_arr, date_range, n_jobs=-1):

        filler = gapfiller.MedianFiller(self.window_size)

        # Use median filling to fill the gaps
        predicted_raster = gapfiller.run_method(filler, original_arr, date_range)
        predicted_raster = predicted_raster[:,:,:]

        return predicted_raster

class PolynomialPredictor(Predictor):
    def __init__(self, poly_degree=2, window_size=15):
        self.poly_degree = poly_degree
        self.window_size = window_size

    def img_predictor(self, original_arr, date_range, n_jobs=-1):

        filler = gapfiller.PolynomialFiller(self.poly_degree, self.window_size)

        # Use median filling to fill the gaps
        predicted_raster = gapfiller.run_method(filler, original_arr, date_range)
        predicted_raster = predicted_raster[:,:,:]

        return predicted_raster

class HarmonicPredictor(Predictor):
    def __init__(self, window_size=15):
           self.window_size = window_size

    def img_predictor(self, original_arr, date_range, n_jobs=-1):

        filler = gapfiller.HarmonicFiller(self.window_size)

        # Use median filling to fill the gaps
        predicted_raster = gapfiller.run_method(filler, original_arr, date_range)
        predicted_raster = predicted_raster[:,:,:]

        return predicted_raster

class LightGBMPredictor(Predictor):
    def __init__(self, window_size=15, n_estimators=50, random_state=0):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.random_state = random_state

    def img_predictor(self, original_arr, date_range, n_jobs=-1):

        filler = gapfiller.LightGBMFiller(self.window_size, self.n_estimators, self.random_state)

        # Use median filling to fill the gaps
        predicted_raster = gapfiller.run_method(filler, original_arr, date_range)
        predicted_raster = predicted_raster[:,:,:]

        return predicted_raster

def get_method(method_name: str) -> Predictor:
    """
    Factory function to return the appropriate filler class based on the method name.
    :param method_name: Name of the gap-filling method (e.g., 'median', 'polynomial', 'harmonic', 'lightgbm')
    :return: Instance of the corresponding Filler subclass
    """
    if method_name == 'median':
         return MedianPredictor()
    elif method_name == 'polynomial':
         return PolynomialPredictor()
    elif method_name == 'harmonic':
            return HarmonicPredictor()
    elif method_name == 'lightgbm':
          return LightGBMPredictor()
    else:
        raise ValueError(
            f"Filler {Predictor} not implemented. Available methods are: [median, polynomial, harmonic, and 'lightgbm']"
        )

def run_method(Predictor, img_list: list, base_dates: list, interval: int):

    dates_counter = Counter(base_dates)
    start_date = min(base_dates)
    end_date = max(base_dates)

    # Calculate the date range based on the specified interval
    date_range = [start_date + timedelta(days=i) for i in range(0, (end_date - start_date).days + 1, interval)]

    # Stack the list of rasters
    stack_rasters = utils._stack_raster(img_list)

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

    predicted_raster = Predictor.img_predictor(original_arr, date_range)

    return predicted_raster, date_range