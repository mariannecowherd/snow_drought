## define global parameters ##

homedir = '/global/u1/c/cowherd/e3sm_eval/snow_drought/'
savepath = '/global/cfs/cdirs/e3sm/cowherd/data/test/'

## names of all models to query
allnames = ['ACCESS-CM2', 
             'BCC-CSM2-MR', 
             'CESM2-WACCM', 
             'GFDL-ESM4', 
             'GFDL-CM4',
             'IPSL-CM6A-LR', 
             'MIROC6',
             'MRI-ESM2-0',
             'UKESM1-0-LL']
## which experiment ids?
experiment_ids = ['historical', 'ssp245', 'ssp585']
## which variables? 
variables = ['snw','tas','pr']

years = {'historical': [1850,2014],
         'ssp245':[2015,2099],
         'ssp585':[2015,2099]}
table_ids = {'tas':'Amon', 
             'pr': 'Amon',
             'snw':'LImon'}
labels =  {'snw':'SWE [kg m$^{-2}$]', 
             'pr': 'Precip [kg $d^{-1}$]',
             'tas':' Temp [K]'}

