
import gc
import glob
import dask
import fsspec
import intake


import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl

from collections import defaultdict
from matplotlib import pyplot as plt

from dask.diagnostics import progress
# import noresmfunctions as fct
from scipy.stats import norm
from tqdm.autonotebook import tqdm
import xml.etree.ElementTree as ET

from params import allnames
from params import homedir
from params import experiment_ids, years, table_ids, labels, variables, savepath

figdir = homedir + 'figures/'
#Functions and otherwise...

from netCDF4 import Dataset
import pylab as P
import numpy as np
from numpy import ma
from numpy import dtype
import matplotlib
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import xarray as xr
import pandas as pd
import os
from scipy.interpolate import griddata
import wrf
from wrf import (getvar, to_np, vertcross, smooth2d, CoordPair, GeoBounds,
                 get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim)# 
# import xesmf as xe
import dask
import gc
import datetime
import time
import scipy
import scipy.stats

states = cfeature.NaturalEarthFeature(category='cultural', scale='50m',
                                      facecolor='none',
                                      name='admin_1_states_provinces_shp')


roll = 122  #days between 1 Sept and 1 Jan
bad = -999
dir_out = "/glade//"

#Read meta
def _metaread(dir_meta,domain):
 file = "%swrfinput_%s" %(dir_meta,domain)
 data = xr.open_dataset(file)
 lat = data.variables["XLAT"]
 lon = data.variables["XLONG"]
 z = data.variables["HGT"]

 return (lat,lon,z,file)

#WRF reader function
def _wrfread(prefix,dir,var,domain,calendar):

 all_files = sorted(os.listdir(dir))

 anal_files = []
 for ii in all_files:
  if ii.startswith(var+"."):
   if domain in ii:
    if prefix in ii:
     anal_files.append(dir+str(ii))

 del all_files

 nf  = len(anal_files)

 data = xr.open_mfdataset(anal_files, combine="by_coords")
 var_read = data.variables[var]
 day = data.variables["day"].values
 nt = len(day)

 day1 = str ( int ( day[0] ) )
 val1 = day1[0:4]
 val2 = int (day1[4:6])
 val3 = int (day1[6:8])
 print (val1,val2,val3)
 day1_str = "%s-%s-%s" %(val1, "{:0=2d}".format(val2),
                                        "{:0=2d}".format(val3) )

 day2 = str ( int ( day[nt-1] ) )
 val1 = day2[0:4]
 val2 = int (day2[4:6])
 val3 = int (day2[6:8])
 print (val1,val2,val3)
 day2_str = "%s-%s-%s" %(val1, "{:0=2d}".format(val2),
                                        "{:0=2d}".format(val3) )
 time_array = xr.cftime_range(start=day1_str, end=day2_str,
                              freq="1D", calendar=calendar)


 var_read = xr.DataArray(var_read)
 var_read['day'] = time_array    #year doesn't matter here

 return (var_read.rename({'day': 'time'}))

#WRF tier 3 reader function
def _wrfread_gcm(model,gcm,variant,dir,var,domain,calendar):

 all_files = sorted(os.listdir(dir))

 anal_files = []
 for ii in all_files:
  if ii.startswith(var+".") and model in ii and gcm in ii \
        and variant in ii and domain in ii:
   if domain in ii:
     anal_files.append(dir+str(ii))

 del all_files

 nf  = len(anal_files)

 data = xr.open_mfdataset(anal_files, combine="by_coords")
 var_read = data.variables[var]
 day = data.variables["day"].values
 nt = len(day)

 day1 = str ( int ( day[0] ) )
 val1 = day1[0:4]
 val2 = int (day1[4:6])
 val3 = int (day1[6:8])
 day1_str = "%s-%s-%s" %(val1, "{:0=2d}".format(val2),
                                        "{:0=2d}".format(val3) )

 day2 = str ( int ( day[nt-1] ) )
 val1 = day2[0:4]
 val2 = int (day2[4:6])
 val3 = int (day2[6:8])
 day2_str = "%s-%s-%s" %(val1, "{:0=2d}".format(val2),
                                        "{:0=2d}".format(val3) )

 time_array = xr.cftime_range(start=day1_str, end=day2_str,
                              freq="1D", calendar=calendar)   

 var_read = xr.DataArray(var_read)
 var_read['day'] = time_array    #year doesn't matter here

 return (var_read)

def _read_merge(dir,domain,var,gcm,variant,date_start_hist, \
                date_start_ssp,date_end_hist,date_end_ssp,exp,bc,calendar):
    
    dir_x = "%s/%s_%s_historical/postprocess/" %(dir,gcm,variant) + domain + "/"
    
    if bc == True:
        dir_x = "%s/%s_%s_historical_bc/postprocess/" %(dir,gcm,variant) + domain + "/"

    if calendar == '360_day':
        date_end_hist = date_end_hist.split("-")
        date_end_hist = "%s-%s-%s" %(date_end_hist[0],
                                     date_end_hist[1],
                                     int( int(date_end_hist[2]) + 0.0001 - 1) )
        
        date_end_ssp = date_end_ssp.split("-")
        date_end_ssp = "%s-%s-%s" %(date_end_ssp[0],
                                     date_end_ssp[1],
                                     int( int(date_end_ssp[2]) + 0.0001 - 1) )
        
    print (calendar,date_end_hist,date_end_ssp)
    var_wrf = _wrfread_gcm("hist",gcm,variant,dir_x,var,domain,calendar)
    var_hist = var_wrf.sel(day=slice(date_start_hist,date_end_hist))
    dir_x = "%s/%s_%s_%s/postprocess/" %(dir,gcm,variant,exp) + domain + "/"
    
    print (gcm,variant,bc,exp)
    
    if bc == True:
        dir_x = "%s/%s_%s_%s_bc/postprocess/" %(dir,gcm,variant,exp) + domain + "/"

    var_wrf = _wrfread_gcm(exp,gcm,variant,dir_x,var,domain,calendar)
    var_ssp = var_wrf.sel(day=slice(date_start_ssp,date_end_ssp) )
    
    var_combine = xr.concat([var_hist,var_ssp],dim="day").rename({'day': 'time'})
    
    return (var_combine)

def _metaread(dir_meta,domain):
 file = "%swrfinput_%s" %(dir_meta,domain)
 data = xr.open_dataset(file)
 lat = data.variables["XLAT"]
 lon = data.variables["XLONG"]
 z = data.variables["HGT"]

 return (lat,lon,z,file)

def normalize(data):
    mean = np.nanmean(data)
    norm = [(val-mean)/mean for val in data]
    return norm

    

def process_variable(var, experiment):
    ## static
    cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(cat_url)
    label = labels[var]
    table_id = table_ids[var]  
    starty = years[experiment][0]
    endy = years[experiment][1]
    cat = col.search(source_id=allnames, table_id = table_id, experiment_id=[experiment], variable_id=var)
    ## create dict 
    dset_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True,}, )
    # metadata of the historical run:
    _d2 = pd.Series(["calendar",
                     "branch_time_in_parent", #"parent_activity_id", "parent_experiment_id",	"parent_mip_era",
                     "parent_source_id",#"parent_sub_experiment_id", 
                     "parent_time_units",# "parent_variant_label"
                      ])

    for i in dset_dict.keys():
        _data = []
        _names =[]
        _data.append(dset_dict[i].time.to_index().calendar)
        for k, v in dset_dict[i].attrs.items():
            if 'parent_time_units' in k or 'branch_time_in_parent' in k or 'parent_source_id' in k:
                _data.append(v)
                _names.append(k)
        _d2 = pd.concat([_d2,   pd.Series(_data)], axis=1)
        _d2.rename(columns={1:i.split('.')[2]}, inplace=True)
        _d2.rename(columns={0:i.split('.')[2]}, inplace=True)

    _d2.dropna(how='all', axis=1, inplace=True)

    tmp = np.where(['CESM2' in val for val in list(dset_dict.keys())])
    tmp = [int(val) for val in tmp]
    base_key = list(dset_dict.keys())[tmp[0]]

    ds = dset_dict[base_key]

    _month = ds[var].groupby('time.month').mean('time', keep_attrs = True)
    _jan  = _month.sel(month = 1)

    year_range = range(starty, endy+1)

    # create dictionary for reggridded data
    ds_gridded_dict = dict()
    # Read in the output grid from the base
    ds_out = dset_dict[base_key].isel(member_id = 0)
    ds_out = ds_out.sel(time = ds_out.time.dt.year.isin(year_range)).squeeze()
    standard_cal = ds_out['time']

    counter = 0
    for keys in dset_dict.keys():
        amon = keys.split('.')[-2]
        model = keys.split('.')[2]
        if model in allnames:
            try:
                # select where data should be saved
                filename = f'{var}_{model}_{experiment}.nc'
                nc_out = savepath + filename
                files = glob.glob(nc_out)

                # Input data from CMIP6 model to be regridded
                ds_in = dset_dict[keys].mean(['member_id'], keep_attrs=True).drop('dcpp_init_year') ## mean of variants
                ds_in = ds_in.sel(time = ds_in.time.dt.year.isin(year_range)).squeeze()

                # they may not be on the same calendar and that's ok
                if len(ds_in['time']) == len(standard_cal):
                    ds_in['time'] = standard_cal
                else:
                    ds_in['time'] = standard_cal[1:]

                # Regrid data
                ds_in_regrid = fct.regrid_data(ds_in, ds_out)

                # Shift the longitude from 0-->360 to -180-->180 and sort by longitude and time
                ds_in_regrid = ds_in_regrid.assign_coords(lon=(((ds_in_regrid.lon + 180) % 360) - 180)).sortby('lon').sortby('time')
                ds_in_regrid = ds_in_regrid.reset_coords(names=['time_bnds', ], drop=True)

                # create dataset with all models
                ds_gridded_dict[model] = ds_in_regrid

                if nc_out in files:
                    print('{} is downloaded'.format(nc_out))
                    counter += 1
                    print('Have regridded in total: {:} files'.format(str(counter)))

                else:    
                    # Save to netcdf file
                    try:
                        ds_in_regrid.to_netcdf(nc_out)
                        print('file written: {}'.format(nc_out))
                    except:
                        print('could not write {}'.format(nc_out))
            except:
                print(f'could not compute {var} for {model}')
                
    '''
    _ds = list(ds_gridded_dict.values())
    _coord = list(ds_gridded_dict.keys())
    ds_cmip = xr.concat(objs=_ds, dim=_coord, coords="all").rename({'concat_dim':'model'})
    ds_cmip = ds_cmip.drop('bnds')
    ## this would mask all non-seasonal snow ## 

    ds_cmip[var+'_season_mean'] = ds_cmip[var].groupby('time.season').mean('time', keep_attrs=True)
    ds_cmip[var+'_model_mean'] = ds_cmip[var].groupby('time').mean('model', keep_attrs=True, skipna = True)
    ds_cmip[var+'_season_model_mean'] = ds_cmip[var+'_season_mean'].mean('model', keep_attrs=True, skipna = True)
    ds_cmip[var+'_season_model_std']  = ds_cmip[var+'_season_mean'].std('model', keep_attrs=True, skipna = True)
    if var == 'snw':
        num_years = len(ds_in['time']) // 12
        low_snw = ds_cmip['snw_model_mean'] < 0.1
        low_snw_count = low_snw.sum(dim='time')
        low_snw_avg = low_snw_count / num_years
        mask = low_snw_avg > 1
        ds_cmip['snw_model_mean_mask'] = ds_cmip['snw_model_mean'].where(mask, drop=True)
        nc_out = savepath +f'{var}_model_mean_{experiment}.nc'
        ds_cmip['snw_model_mean_mask'].to_netcdf(nc_out)
    else:
        nc_out = savepath +f'{var}_model_mean_{experiment}.nc'
        ds_cmip[var+'_model_mean'].to_netcdf(nc_out)
    print('wrote', nc_out)
    ## now select for the snotel sites
    dsets_snotel = {}
    dsets_ = dask.compute(dict(dsets))[0]
    print('getting SNOTEL data')
    for k, v in tqdm(dsets_.items()):
        expt_dsets = v.values()
        if any([d is None for d in expt_dsets]):
            print(f"Missing experiment for {k}")
            continue

        dsets_snotelonly = [v[expt].pipe(snotel_pipe) # .swap_dims({'time':'month'}).groupby('month').mean()
                              for expt in experiment_ids]
        dsets_snotel[k] = dsets_snotelonly

    with progress.ProgressBar():
        dsets_aligned_snotel = dask.compute(dsets_snotel)[0]

    dsets_aligned_snotel.to_netcdf(savepath + f'snotel_{var}.nc')
    print(f'wrote {savepath}/snotel_{var}.nc')
    '''
    return

    
def drop_all_bounds(ds):
    drop_vars = [vname for vname in ds.coords
                 if (('_bounds') in vname ) or ('_bnds') in vname]
    return ds.drop(drop_vars)

def open_dset(df):
    assert len(df) == 1
    ds = xr.open_zarr(fsspec.get_mapper(df.zstore.values[0]), consolidated=True)
    return drop_all_bounds(ds)

def open_delayed(df):
    return dask.delayed(open_dset)(df)

def get_lat_name(ds):
    """Figure out what is the latitude coordinate for each dataset."""
    for lat_name in ['lat', 'latitude']:
        if lat_name in ds.coords:
            return lat_name
    raise RuntimeError("Couldn't find a latitude coordinate")

def global_mean(ds):
    """Return global mean of a whole dataset."""
    lat = ds[get_lat_name(ds)]
    weight = np.cos(np.deg2rad(lat))
    weight /= weight.mean()
    other_dims = set(ds.dims) - {'time'}
    return (ds * weight).mean(other_dims)

def snotel_pipe(ds):
    """Return snotel location time series."""
    snotelmeta = pd.read_csv(homedir + 'data/snotelmeta.csv')
    var = ds.variable_id
    snotellat = snotelmeta.lat
    snotellon = snotelmeta.lon
    snotelonly = []
    for i in range(len(snotelmeta.lat)):
        lat = snotellat[i]
        lon = snotellon[i] + 180
        try:
            j = np.nanargmin(np.abs(ds.lat-lat))
            k = np.nanargmin(np.abs(ds.lon-lon))
        except:
            j = np.nanargmin(np.abs(ds.latitude-lat))
            k = np.nanargmin(np.abs(ds.longitude-lon))
        snotelonly.append(ds[var][:,j,k])
    return snotelonly


def clip_ds_monthly(xds, basin, lon_name):
    # Adjust lon values to make sure they are within (-180, 180)
    xds['_longitude_adjusted'] = xr.where(
        xds[lon_name] > 180,
        xds[lon_name] - 360,
        xds[lon_name])
    xds = (
        xds
        .swap_dims({lon_name: '_longitude_adjusted'})
        .sel(**{'_longitude_adjusted': sorted(xds._longitude_adjusted)})
        .drop(lon_name))

    xds = xds.rename({'_longitude_adjusted': lon_name})
    xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    xds.rio.write_crs("EPSG:4326", inplace=True)
    clipped_xds = xds.rio.clip(basin.geometry, all_touched = True, crs = basin.crs)
    return clipped_xds

def clip_ds(xds, basin, lon_name, var):
    import xarray
    # Adjust lon values to make sure they are within (-180, 180)
    xds['_longitude_adjusted'] = xarray.where(
        xds[lon_name] > 180,
        xds[lon_name] - 360,
        xds[lon_name])

    # reassign the new coords to as the main lon coords
    # and sort DataArray using new coordinate values
    xds = (
        xds
        .swap_dims({lon_name: '_longitude_adjusted'})
        .sel(**{'_longitude_adjusted': sorted(xds._longitude_adjusted)})
        .drop(lon_name))

    xds = xds.rename({'_longitude_adjusted': lon_name})
    xds = xds[[var]].transpose('time', 'lat', 'lon')
    xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    xds.rio.write_crs("EPSG:4326", inplace=True)
    clipped_xds = xds.rio.clip(basin.geometry, all_touched = True, crs = basin.crs)
    return clipped_xds


import netCDF4 as nc
from scipy.stats import norm

def droughtindx(nsample):
    indx = []
    for i in range(nsample):
        px = (i+1-0.44)/(nsample+0.12)
        indx.append(norm.ppf(px))
    return indx


def get_swei(ds):
    swe = ds['snw']
    ntime = swe.shape[0]
    nlat = swe.shape[2]
    nlon = swe.shape[1]
    nyr = int(ntime / 12)
    nd = nlat * nlon
    nm = 12

    # Compute the 3-month cumulative sum for each pixel
    ds_cumsum = ds.rolling(time=3, min_periods=3).sum()
    #more than 75% of years have swe in that range
    mask = (np.nansum((cumswe > 0), axis=(0)) > 0.75*nyr)
    ds_cumsum[:,~mask] = np.nan
    # mask high values
    ds_cumsum[ds_cumsum>1e19] = np.nan

    years = np.unique(ds.time.dt.year)
    months = np.unique(ds.time.dt.month)

    # Reshape the data back into a 4D array of (year, month, lat, lon)
    ds_new = xr.DataArray(
        ds_cumsum.snw.data.reshape((-1, 12, ds.sizes['lat'], ds.sizes['lon'])),
        dims=('year', 'month', 'lat', 'lon'),
        coords={'year': years, 'month': months, 'lat': ds['lat'], 'lon': ds['lon']}
    )

    
    categ = np.zeros((nyr, nm, nlon, nlat))
    nsample = nyr
    sweix = droughtindx(nsample)  # all values for each pixel.
    sweix = np.array(sweix)

    aindx = np.argsort(ds_new.data, axis=0)

    # Create a broadcasting version of sweix
    sweix_broadcasted = sweix[:, np.newaxis, np.newaxis, np.newaxis]

    # Assign sorted sweix values to categ based on sorted indices (array, indices, values, axis)
    np.put_along_axis(categ, aindx,sweix_broadcasted, axis=0)

    # Create the new xarray Dataset
    ds_swei = xr.Dataset(
        {'swei':(('year','month','lat','lon'), categ)},
        coords={'year': years, 'month': months,'lon': ds['lon'], 'lat': ds['lat'], }
    )
    return ds_swei

'''
def get_swei(swe):
    ntime = swe.shape[0]
    nlat = swe.shape[1]
    nlon = swe.shape[2]
    nm = 12
    nyr = int(ntime/nm)
    nd = nlat*nlon
    mon = []
    for j in range(1,13):
            mon.append([j]*nd)
    mon = mon * int(ntime/12)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        mon = np.array(mon).reshape(ntime,nlat,nlon)

    df = pd.DataFrame({"swe":swe.data.flatten(),"mon":mon.flatten()})
    swe = df['swe'].to_numpy()
    nm = 12
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        swe = swe.reshape(-1, nlat,nlon)
        swe = swe[9:-3,:,:].reshape((-1,12,nlat,nlon))
    cumswe = np.zeros((swe.shape[0],12,nlat,nlon))
    for im in range(1,13):
        cumswe[:,im-1,:,:] = np.nansum(swe[:,im-1:im+2,:,:],axis=1)

    #more than 75% of years have swe in that range
    mask = (np.nansum((cumswe > 0), axis=(0)) > 0.75*nyr)
    cumswe[:,~mask] = np.nan
    # mask high values
    cumswe[cumswe>1e19] = np.nan

    categ = np.zeros((nyr,nm,nlat,nlon))
    nsample = nyr
    sweix = droughtindx(nsample) ## all values for each pixel.
    for i in range(nlat):
        for j in range(nlon):
            for k in range(12):
                aindx = np.argsort(cumswe[:,k-1,i,j]) # puts them in order by year
                    categ[aindx,k,i,j] = (sweix[:]) # then puts in the swei where that goes
    for yr in range(nyr):
        categ[yr,:,:,:][~mask] = np.nan
        categ[np.isnan(cumswe)] = np.nan
    return categ
    '''