"""!
Common utility functions and shared constants/instantiations.
This module contains common utility functions used throughout the package.
"""

import datetime
import glob
import os
import seaborn as sns
from matplotlib import pyplot as plt

import dask
import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr


# CONSTANTS
MM_TO_IN = 0.03937008


def screen_times_wrf(data, date_start, date_end):
    # Dimensions should be "day"
    dask.config.set(**{"array.slicing.split_large_chunks": True})

    datedata = pd.to_datetime(data.day)
    data = data.sel(day=~((datedata.month < date_start[1]) & (datedata.year <= date_start[0])))

    datedata = pd.to_datetime(data.day)
    data = data.sel(day=~(datedata.year < date_start[0]))

    datedata = pd.to_datetime(data.day)
    data = data.sel(day=~((datedata.month >= date_end[1]) & (datedata.year >= date_end[0])))

    datedata = pd.to_datetime(data.day)
    data = data.sel(day=~(datedata.year > date_end[0]))

    return data

def _read_wrf_meta_data(dir_meta: str, domain: str):
    """Read wrf meta data from nc4 files, and return lat, lon, height, and the filename"""
    infile = os.path.join(dir_meta, f"wrfinput_{domain}")
    data = xr.open_dataset(infile, engine="netcdf4")
    lat = data.variables["XLAT"]
    lon = data.variables["XLONG"]
    z = data.variables["HGT"]
    return (lat, lon, z, infile)

