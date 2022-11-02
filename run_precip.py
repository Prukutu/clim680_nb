import gc

import intake
import xarray as xr
import xesmf as xe

from dask.diagnostics import ProgressBar

xclim.set_options(data_validation='warn')

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

print('Ensemble loaded into dict!')

# These dataset is missing geographical coordinates or are outside our time of interest (> 2100).
del dset_dict['historical']['CMIP.MPI-M.ICON-ESM-LR.historical.day.gn']
del dset_dict['ssp126']['ScenarioMIP.CSIRO-ARCCSS.ACCESS-CM2.ssp126.day.gn']
del dset_dict['ssp585']['ScenarioMIP.CSIRO-ARCCSS.ACCESS-CM2.ssp585.day.gn']

# Get the maximum 1 day precipitation (RX1Day index) per year
# Models in common between all ensemble members
allowed_models = ['ACCESS-CM2', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'CanESM5',
                  'EC-Earth3', 'EC-Earth3-Veg-LR','GFDL-ESM4', 'IITM-ESM', 'INM-CM4-8' ,
                  'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC6', 'MPI-ESM1-2-HR',
                  'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
print('Computing climate indices...')
rx1day = {}
for sce in scenarios:
#    print(sce)
    modelvals = {}
    cdd = {}
    for model in dset_dict[sce].keys():


        modname = model.split('.')[2]

        # Convert precip to mm/day
        # dset_dict[sce][model]['pr'] = dset_dict[sce][model]['pr']

        if modname in allowed_models:
#            print(model)
            modelvals[model] = dset_dict[sce][model]['pr'].resample(time='AS').max('time')

            if sce == 'historical':
                mask = (modelvals[model].time.dt.year >=1990) & (modelvals[model].time.dt.year <= 2010)
                ds = modelvals[model][:, mask, :, :]
            else:
                mask = (modelvals[model].time.dt.year <= 2100) & (modelvals[model].time.dt.year >= 2015)
                ds = modelvals[model][:, mask, :, :]

            modelvals[model] = ds.to_dataset()

    rx1day[sce] = modelvals
print('RX1Day and CDD computed and stored to dict')

# Regrid all models to common 1 degree grid
rx1day_1deg = {}
resx, resy = 1, 1

print('Regridding data and standardizing time dim')
for sce in scenarios:
    print(sce)
    newvals = {}
    for model in rx1day[sce].keys():
#        print(model)

        # Create an empty dataset that will have the grid that we want.
        out = xr.Dataset(
            {
                'lat': (['lat'], np.arange(-90, 90, resy)),
                'lon': (['lon'], np.arange(0, 360, resx)),
            }
        )

        regridder = xe.Regridder(rx1day[sce][model], out, 'bilinear', periodic=True)
        newvals[model] = regridder(rx1day[sce][model])

    rx1day_1deg[sce] = newvals

# Convert all times to year ints to avoid calendar mismatch nonsense (which doesn't apply to our case)
for sce in scenarios:
    for model in rx1day_1deg[sce]:
#        print(model)
        if rx1day_1deg[sce][model]['time'].dtype != 'int64':
            rx1day_1deg[sce][model]['time'] = rx1day_1deg[sce][model]['time'].dt.year
        else:
            pass


print('Deleting older data...')

ensemblemeans = {}
for sce in scenarios:
    ensemblemeans[sce] = xr.concat(rx1day_1deg[sce].values(), 'member_id').mean('member_id')
print('Concatenating files complete')


print('preparing final files...')
# Concatenate individual models into single dataset for each scenario
resx, resy = .25, .25

for sce in scenarios:
    print('Processing ' + sce)
#    outds = xr.concat(rx1day_1deg[sce].values(), 'member_id').mean('member_id')
#    print('Concat complete!')
    out = xr.Dataset(
            {
                'lat': (['lat'], np.arange(-90, 90, resy)),
                'lon': (['lon'], np.arange(0, 360, resx)),
            }
        )

    regridder = xe.Regridder(ensemblemeans[sce], out, 'bilinear', periodic=True)
    ds_out = regridder(ensemblemeans[sce])
    print('Regridding complete!')
    # Save to file
    print('Writing to file')

    write_job = ds_out.to_netcdf('/scratch/lortizur/precip_' + sce + '.nc', compute=False)

    with ProgressBar():
        write_job.compute()


    print('Writing to NetCDF Complete!')


