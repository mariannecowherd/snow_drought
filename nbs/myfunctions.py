
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
import noresmfunctions as fct
from scipy.stats import norm
from tqdm.autonotebook import tqdm
import xml.etree.ElementTree as ET

from params import allnames
from params import homedir
from params import experiment_ids, years, table_ids, labels, variables, savepath

figdir = homedir + 'figures/'

def normalize(data):
    mean = np.nanmean(data)
    norm = [(val-mean)/mean for val in data]
    return norm

def process_variable(var, experiment):
    cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(cat_url)

    add_colorbar = True
    label = labels[var]
    table_id = table_ids[var]  
    starty = years[experiment][0]
    endy = years[experiment][1]
    cat = col.search(source_id=allnames, table_id = table_id, experiment_id=[experiment], variable_id=var)
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
    
    tmp = np.where(['CSM2' in val for val in list(dset_dict.keys())])
    tmp = [int(val) for val in tmp]
    base_key = list(dset_dict.keys())[tmp[0]]
    
    ds = dset_dict[base_key]
    
    _month = ds[var].groupby('time.month').mean('time', keep_attrs = True)
    _jan  = _month.sel(month = 1)
    
    year_range = range(starty, endy+1)
    
    # create dictionary for reggridded data
    ds_gridded_dict = dict()

    # Read in the output grid from NorESM
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
                filename = f'{var}_{model}-{starty}01_{endy}12.nc'
                nc_out = savepath + filename
                files = glob.glob(nc_out)

                # Input data from CMIP6 model to be regridded
                ds_in = dset_dict[keys].isel(member_id = 0)
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
                print(model)
    _ds = list(ds_gridded_dict.values())
    _coord = list(ds_gridded_dict.keys())
    ds_cmip = xr.concat(objs=_ds, dim=_coord, coords="all").rename({'concat_dim':'model'})
    ds_cmip = ds_cmip.drop('bnds')
    ds_cmip[var].attrs

    ds_cmip[var+'_season_mean'] = ds_cmip[var].groupby('time.season').mean('time', keep_attrs=True)
    ds_cmip[var+'_model_mean'] = ds_cmip[var].groupby('time').mean('model', keep_attrs=True, skipna = True)

    for model in ds_cmip.model.values:
        fig,_,_ = fct.plt_spatial_seasonal_mean(ds_cmip[var+'_season_mean'].sel(model=model),var, title='{} MEAN ({} - {})'.format(model,starty, endy))
        plt.savefig(figdir + f'{model}_{var}_seasonal_mean.jpg')
    ds_cmip[var+'_season_model_mean'] = ds_cmip[var+'_season_mean'].mean('model', keep_attrs=True, skipna = True)
    ds_cmip[var+'_season_model_std']  = ds_cmip[var+'_season_mean'].std('model', keep_attrs=True, skipna = True)
    nc_out = savepath +f'{var}_model_mean_{experiment}.nc'
    ds_cmip[var+'_model_mean'].to_netcdf(nc_out)
    print('wrote', nc_out)

    fig, axs, im = fct.plt_spatial_seasonal_mean(ds_cmip[var+'_season_model_mean'], var, add_colorbar = False, title='Inter-Model Mean {}, {}'.format(var,experiment))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1, 0.15, 0.025, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="vertical", fraction=0.046, pad=0.04)
    cb.set_label(label=f'MEAN - {label}', weight='bold')
    plt.tight_layout()

    # save figure to png
    figname = f'{var}_season_mean_{starty}_{endy}.png'
    plt.savefig(figdir + figname, format = 'png', bbox_inches = 'tight', transparent = False)
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


def global_mean_60s(ds):
    """Return global mean of a whole dataset from 60 n to 60 s."""
    try:
        ds = ds.sel(lat = slice(-60,60))
    except:
        ds = ds.sel(latitude = slice(-60,60))
        

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



def load_global(variable, table_id, experiment_ids, rm_arc = False):
    """
    function to load all models for a given variable and experiment id
    rm_arc tells whether to clip all data to the 60, -60 latitude range
    """
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    query = dict(
        experiment_id=experiment_ids,
        table_id=table_id,
        variable_id=[variable],
        source_id = allnames,
    )

    col_subset = col.search(require_all_on=["source_id"], **query)
    col_subset.df.groupby("source_id")[
        ["experiment_id", "variable_id", "table_id"]
    ].nunique()
    
    ## this gets us the first member_id for each source and experiment ## 
    col_subset_first = col_subset.df.groupby(['source_id','experiment_id']).first()

    dsets = defaultdict(dict)

    #for group, df in col_subset.df.groupby(by=['source_id', 'experiment_id']):
    #    dsets[group[0]][group[1]] = open_delayed(df)
    
    for group, df in col_subset_first.groupby(by=['source_id', 'experiment_id']):
        dsets[group[0]][group[1]] = open_delayed(df)
        
    dsets_ = dask.compute(dict(dsets))[0]
        
    expt_da = xr.DataArray(experiment_ids, dims='experiment_id', name='experiment_id',
                        coords={'experiment_id': experiment_ids})

    dsets_aligned = {}
    dsets_aligned_monthly = {}
    dsets_snotel = {}

    for k, v in tqdm(dsets_.items()):
        expt_dsets = v.values()
        if any([d is None for d in expt_dsets]):
            print(f"Missing experiment for {k}")
            continue

        for ds in expt_dsets:
            ds.coords['year'] = ds.time.dt.year

        # workaround for
        # https://github.com/pydata/xarray/issues/2237#issuecomment-620961663
        if rm_arc: pipe = global_mean_60s
        else: pipe = global_mean
            
        
        dsets_ann_mean = [v[expt].pipe(pipe)
                                .swap_dims({'time': 'year'})
                                .drop('time')
                                .coarsen(year=12).mean()
                        for expt in experiment_ids]

        # align everything with the 4xCO2 experiment
        dsets_aligned[k] = xr.concat(dsets_ann_mean, join='outer',
                                    dim=expt_da)
        for ds in expt_dsets:
            ds.coords['month'] = ds.time.dt.month
            ds.coords['year'] = ds.time.dt.year
            ds.coords['month'] = ds.time.dt.month
            if ds.activity_id != 'ScenarioMIP':
                tmp = ds.swap_dims({'time':'year'}).sel(year=slice(1900, 1980))
                years = [val.year for val in tmp.time]
                tmpdata = np.array(tmp[variable].data)
                newtmpdata = tmpdata.reshape((12,-1,tmpdata.shape[1],tmpdata.shape[2]))

                monthly_da = xr.Dataset(
                    data_vars = dict(
                        tas = (['month','year','lat','lon'],newtmpdata),
                    ),
                    coords = dict(
                        lon = tmp.lon,
                        lat = tmp.lat,
                        time = years,
                        month = range(1,13),
                    )
                )
                norm = monthly_da.mean(dim='year')
                norm.to_netcdf(savepath + f'{variable}_norm_{ds.source_id}.nc')
                
                tmp = ds.swap_dims({'time':'year'})
                years = [val.year for val in tmp.time]
                tmpdata = np.array(tmp[variable].data)
                newtmpdata = tmpdata.reshape((12,-1,tmpdata.shape[1],tmpdata.shape[2]))

                monthly_da = xr.Dataset(
                    data_vars = dict(
                        tas = (['month','year','lat','lon'],newtmpdata),
                    ),
                    coords = dict(
                        lon = tmp.lon,
                        lat = tmp.lat,
                        time = years,
                        month = range(1,13),
                    )
                )
                norm = nc.Dataset(savepath + f'{variable}_norm_{ds.source_id}.nc')
                normdata = norm.variables['tas'][:]
                normlon = norm.variables['lon'][:]
                normlat = norm.variables['lat'][:]
                norm_da = xr.Dataset(
                        data_vars = dict(
                            norm = (['month','lat','lon'],normdata),
                        ),
                        coords = dict(
                            lon = normlon,
                            lat = normlat,
                            month = range(1,13),
                        )
                    )
                anom = monthly_da.tas - norm_da.norm
                anom.to_netcdf(savepath + f'{variable}_anom_{ds.source_id}_{ds.experiment_id}.nc')

        dsets_monthly_mean = [v[expt].pipe(pipe).swap_dims({'time':'month'}).groupby('month').mean()
                              for expt in experiment_ids]
        dsets_aligned_monthly[k] = dsets_monthly_mean
        
        dsets_snotelonly = [v[expt].pipe(snotel_pipe) # .swap_dims({'time':'month'}).groupby('month').mean()
                              for expt in experiment_ids]
        dsets_snotel[k] = dsets_snotelonly
        
    with progress.ProgressBar():
        dsets_aligned_ = dask.compute(dsets_aligned)[0]
        dsets_aligned_monthly_ = dask.compute(dsets_aligned_monthly)[0]
        dsets_aligned_snotel = dask.compute(dsets_snotel)[0]
        
    source_ids = list(dsets_aligned_.keys())
    source_da = xr.DataArray(source_ids, dims='source_id', name='source_id',
                            coords={'source_id': source_ids})

    big_ds = xr.concat([ds.reset_coords(drop=True)
                        for ds in dsets_aligned_.values()],
                        dim=source_da)
    monthly_ds = dsets_aligned_monthly_
    
    df_all = big_ds.sel(year=slice(1900, 2100)).to_dataframe().reset_index()
    ds_mean = big_ds[[variable]].sel(experiment_id='historical').mean(dim='year')
    ds_anom = big_ds[[variable]] - ds_mean
    return df_all, ds_mean, ds_anom, monthly_ds, dsets_aligned_snotel



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

def get_swei(swe):
    ntime = swe.shape[0]
    nlat = swe.shape[1]
    nlon = swe.shape[2]
    nm = 12
    nyr = int(ntime/nm)
    nyr2 = nyr-1
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
    mask = (np.nansum((cumswe > 0), axis=(0)) > 0.75*nyr2)
    cumswe[:,~mask] = np.nan
    # mask high values
    cumswe[cumswe>1e19] = np.nan

    categ = np.zeros((nyr2,nm,nlat,nlon,))
    nsample = nyr2
    sweix = droughtindx(nsample) ## all values for each pixel.
    for i in range(nlat):
            for j in range(nlon):
                    for k in range(12):
                            aindx = np.argsort(cumswe[:,k-1,i,j]) # puts them in order by year
                            categ[aindx,k,i,j] = (sweix[:]) # then puts in the swei where that goes
    for yr in range(nyr2):
            categ[yr,:,:,:][~mask] = np.nan
            categ[np.isnan(cumswe)] = np.nan
    return categ