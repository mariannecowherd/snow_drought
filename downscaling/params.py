## define global parameters ##

homedir = '/global/u1/c/cowherd/snow_drought/'
savepath = '/global/cfs/cdirs/m4099/cowherd/'
datadir = '/global/cfs/cdirs/m4099/fate-of-snotel/wrfdata/'
## names of all models to query
allnames = [ 'ukesm1-0-ll_bc']
## which experiment ids?
experiment_ids = ['historical', 'ssp370']
## which variables? 
variables = ['snow','t2','prec_c']

years = {'historical': [1850,2014],
         'ssp370':[2015,2099],
         'ssp585':[2015,2099]}

table_ids = {'tas':'Amon', 
             'pr': 'Amon',
             'snw':'LImon'}
labels =  {'snw':'SWE [kg m$^{-2}$]', 
             'pr': 'Precip [kg $d^{-1}$]',
             'tas':' Temp [K]'}


colors = {'historical':'black',
          'ssp245':'darkblue',
          'ssp585':'darkred'}

