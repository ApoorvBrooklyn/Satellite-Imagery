from glob import glob

import earthpy.spatial as es
import earthpy.plot as ep

import rasterio as rio
import numpy as np

# Function to calculate NDWI
def calculate_ndwi(band_green, band_nir):
    return (band_green - band_nir) / (band_green + band_nir)

# Function to detect floods
def detect_flood(before_floods_path, during_floods_path):
    # Load before flood data
    before_floods = glob(before_floods_path + "/*B?*.tiff")
    before_floods.sort()

    # Load during flood data
    during_floods = glob(during_floods_path + "/*B?*.tiff")
    during_floods.sort()

    # Read before flood data
    l = []
    for i in before_floods:
        with rio.open(i, 'r') as f:
            l.append(f.read(1))
    arr_bef = np.stack(l)

    # Read during flood data
    dl = []
    for i in during_floods:
        with rio.open(i, 'r') as f:
            dl.append(f.read(1))
    arr_dur = np.stack(dl)

    # Calculate NDWI for before and during flood data
    ndwi_bef = calculate_ndwi(arr_bef[5, :, :], arr_bef[7, :, :])
    ndwi_dur = calculate_ndwi(arr_dur[5, :, :], arr_dur[7, :, :])

    # Threshold NDWI to detect water bodies
    mask_bef = (ndwi_bef > 0.6).astype(int)
    mask_dur = (ndwi_dur > 0.6).astype(int)

    # Calculate the difference between water presence before and during flood
    flood_diff = np.sum(mask_dur) - np.sum(mask_bef)

    # Determine if flood is occurring based on the difference
    if flood_diff > 0:
        print("Flood is occurring!")
    else:
        print("No flood detected.")

# Paths to before and during flood data
before_floods_data_path = r"C:\Users\HP\Desktop\Final EL\Satellite_Imagery_Analysis-main\Data\Madagascar_data\Madagascar_18_01_2017"
during_floods_data_path = r"C:\Users\HP\Desktop\Final EL\Satellite_Imagery_Analysis-main\Data\Madagascar_data\Madagascar_27_01_2020"

# Call the detect_flood function
detect_flood(before_floods_data_path, during_floods_data_path)
