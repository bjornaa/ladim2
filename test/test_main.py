"""Module for testing config file API"""


import pytest
import xarray as xr
import pandas as pd

import uuid
import shutil
import contextlib
from pathlib import Path
from ladim2.main import main
import yaml
import numpy as np


@contextlib.contextmanager
def tempfile(num=1):
    d = Path(__file__).parent.joinpath('temp')
    d.mkdir(exist_ok=True)
    paths = [d.joinpath(uuid.uuid4().hex + '.tmp') for _ in range(num)]

    try:
        if len(paths) == 1:
            yield str(paths[0])
        else:
            yield [str(p) for p in paths]

    finally:
        for p in paths:
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except IOError:
                pass

        try:
            d.rmdir()
        except IOError:
            pass


@contextlib.contextmanager
def runladim(conf):
    with tempfile(4) as fnames:
        conf_fname, out_fname, gridforce_fname, rls_fname = fnames

        conf['grid']['filename'].to_netcdf(gridforce_fname)
        conf['grid']['filename'] = gridforce_fname

        conf['forcing']['filename'].to_netcdf(gridforce_fname)
        conf['forcing']['filename'] = gridforce_fname

        conf['release']['release_file'].to_csv(rls_fname, sep='\t', index=False)
        conf['release']['release_file'] = rls_fname

        conf['output']['filename'] = out_fname

        with open(conf_fname, 'w', encoding='utf-8') as conf_file:
            yaml.safe_dump(conf, conf_file)

        main(conf_fname)

        with xr.open_dataset(out_fname) as dset:
            yield dset


def make_gridforce():
    x = xr.Variable('xi_rho', np.arange(5))
    y = xr.Variable('eta_rho', np.arange(7))
    x_u = xr.Variable('xi_u', np.arange(len(x) - 1))
    y_u = xr.Variable('eta_u', np.arange(len(y)))
    x_v = xr.Variable('xi_v', np.arange(len(x)))
    y_v = xr.Variable('eta_v', np.arange(len(y) - 1))
    s = xr.Variable('s_rho', np.arange(2))
    s_w = xr.Variable('s_w', np.arange(len(s) + 1))
    ocean_t = xr.Variable(
        'ocean_time',
        np.datetime64('2000-01-02T03')
        + np.arange(3) * np.timedelta64(1, 'm')
    )
    t = xr.Variable('ocean_time', np.arange(len(ocean_t)))

    return xr.Dataset(
        data_vars=dict(
            ocean_time=ocean_t,
            h=(y*0 + x*0) + 3.,
            mask_rho=(y*0 + x*0) + 1,
            pm=(y*0 + x*0) + 1.,
            pn=(y*0 + x*0) + 1.,
            angle=(y*0 + x*0) + 0.,
            hc=0.,
            Cs_r=1. - (s + 0.5)/len(s),
            Cs_w=1. - s_w/(len(s_w) - 1),
            u=t*0. + s*0 + y_u + x_u,
            v=t*0. + s*0 - y_v + x_v,
        ),
        coords=dict(
            lon_rho=(x*1 + y*0),
            lat_rho=(x*0 + y*1),
        ),
    )


def make_release():
    t = ['2000-01-02T03:00', '2000-01-02T03:01', '2000-01-02T03:02']

    return pd.DataFrame(
        data=dict(
            release_time=t,
            X=np.ones(len(t)),
            Y=np.ones(len(t)),
            Z=np.zeros(len(t)),
        )
    )


class Test_minimal:
    @pytest.fixture(scope='class')
    def result(self):

        gridforce = make_gridforce()
        release = make_release()

        conf = dict(
            version=2,

            time=dict(
                dt=[30, 's'],
                start=str(gridforce.ocean_time[0].values),
                stop=str(gridforce.ocean_time[-1].values),
            ),
            grid=dict(
                module='ladim2.grid_ROMS',
                filename=gridforce,
            ),
            forcing=dict(
                module='ladim2.forcing_ROMS',
                filename=gridforce,
            ),
            release=dict(
                release_file=release,
            ),
            tracker=dict(
                advection=dict(),
            ),
            output=dict(
                output_period=[30, 's'],
                instance_variables=dict(),
            ),
        )

        with runladim(conf) as gridforce:
            yield gridforce

    def test_minimal(self, result):
        assert isinstance(result, xr.Dataset)
