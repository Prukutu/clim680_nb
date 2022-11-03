import intake
import xarray as xr
import numpy as np

from dask.diagnostics import ProgressBar


# Using the CMIP6 ensemble stored in google cloud by Pangeo project and opening the data store object
col_url = 'https://storage.googleapis.com/cmip6/pangeo-cmip6.json'
col = intake.open_esm_datastore(col_url)

# Scenarios we're interested in.
scenarios = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
# scenarios = ['ssp585',]

# Create dict to hold the extracted dataset objects for each scenario.
print('Loading ensemble...')
dset_dict = {}
for sce in scenarios:

    col_subset = col.search(experiment_id=sce,
                            variable_id='pr',
                            member_id='r1i1p1f1',
                            table_id='day',
                            )
    # Convert to xarray dataset objects
    dset_dict[sce] = col_subset.to_dataset_dict(zarr_kwargs={'consolidated': True})

print('Computing climate indices...')
rx1day = {}
for sce in scenarios:
    datadict = {}

    for model in dset_dict[sce].keys():
        
        datadict[model] = dset_dict[sce][model]['pr'].groupby('time.year').max()
        
        if sce == 'historical':
            datadict[model] = datadict[model].sel(year=slice(1985, 2014))
        else:
            datadict[model] = datadict[model].sel(year=slice(2015, 2099))
        if datadict[model]['year'].shape == 0:
            del datadict[model]


    rx1day[sce] = datadict

print('Ensemble loaded into dict!')
# Regrid all models to common 1 degree grid
dataout = {}
resx, resy = .5, .5
del rx1day['historical']['CMIP.MPI-M.ICON-ESM-LR.historical.day.gn']

out = xr.Dataset(
            {
                'lat': (['lat'], np.arange(-90, 90, resy)),
                'lon': (['lon'], np.arange(0, 360, resx)),
            }
        )



print('Regridding data and standardizing time dim')
for sce in scenarios:
    print(sce)
    for model in rx1day[sce].keys():
        print(model)

        # Create an empty dataset that will have the grid that we want.

        rx1day[sce][model] = rx1day[sce][model].interp_like(out)
        rx1day[sce][model] = rx1day[sce][model].assign_coords(coords={'model': model.split('.')[2]})
    rx1day[sce] = xr.concat(list(rx1day[sce].values()), 'model', coords='minimal')



for sce in scenarios:
    # Save to file
    print('Writing to file')

    write_job = rx1day[sce].to_netcdf('/scratch/lortizur/rx1day' + sce + '.nc', compute=False)

    with ProgressBar():
        write_job.compute()


    print('Writing to NetCDF Complete!')
