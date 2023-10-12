## define global parameters ##

homedir = '/glade/u/home/mcowherd/snow_drought/'
savepath = '/glade/campaign/uwyo/wyom0112/berkeley/'
datadir = '/glade/campaign/uwyo/wyom0112/postprocess/'
## names of all models to query
allnames = [ 'ukesm1-0-ll_bc']
## which experiment ids?
experiment_ids = ['historical', 'ssp370']
## which variables? 
variables = ['snow','t2','prec']

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

