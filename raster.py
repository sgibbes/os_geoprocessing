import subprocess
from inputs import util
from .write_numpy_to_raster import *
from inputs import geop
import pandas as pd
import os
import numpy as np


def func_name(d):

    # calculate percent slope using a dem
    output_slope = os.path.join(d.out_dir, 'dem', 'slope.tif')
    cmd = ['gdaldem', 'slope', '-p', '-co', 'COMPRESS=LZW', d.dem, output_slope]
    subprocess.check_call(cmd)

    # clip slope to state raster
    state_slope_tmp = os.path.join(d.out_dir, 'dem', '{}_slope_temp.tif'.format(d.state))
    geop.clip_raster(d.state_raster, d.state_poly, output_slope, state_slope_tmp)
    # gdalwarp to get pixels to align
    ulx, uly, lrx, lry = get_extent(d.state_raster)

    slope = os.path.join(d.out_dir, 'dem', 'de_slope.tif')
    cmd = ['gdalwarp', '-te', ulx, lry, lrx, uly, '-overwrite', state_slope_tmp, slope]
    subprocess.check_call(cmd)
    d.slope = slope
    slope_array = get_array(slope)
    imp11_array = get_array(d.nlcd11_imp_de)

    # create filename
    slope_imp11_ext = os.path.join(d.out_dir, 'suitability', 'slope_extract.tif')

    # return slope array where we have cat1 data
    slope_cat1_ext_arr = np.where((imp11_array > 0) & (imp11_array <= 100), slope_array, -9999)
    array_to_rast(slope, slope_cat1_ext_arr, slope_imp11_ext, -9999)
    print("done with extracting slope")

    print("clipping slope to each county")
    projected_county = os.path.basename(d.county).replace('.shp', '_proj.shp')
    projected_county = os.path.join(d.out_dir, 'census', projected_county)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(projected_county, 0)
    layer = dataSource.GetLayer()

    county_list = []

    # create list of counties by reading in their geoid10
    for feature in layer:
        county_list.append(feature.GetField('GEOID10'))
    print("clipping projected dem to state poly")

    slope99_array_list = []

    for county in county_list:
        outname = 'slope_ext_{}_{}.tif'.format(d.state, county)
        county_slope_imp11_ext = os.path.join(d.out_dir, 'dem', outname)

        sql = "GEOID10 = '{}'".format(county)

        # clip cat1 to each county using field to filter
        cmd = ['gdalwarp', '-cutline', projected_county, '-tr', '30', '30', '-tap', '-crop_to_cutline',
               '-cwhere', sql, '-ot', 'Float32', '-overwrite', slope_imp11_ext, county_slope_imp11_ext]

        subprocess.check_call(cmd)

        # clip state slope to each county
        outname = 'slope_{}_{}.tif'.format(d.state, county)
        county_slope = os.path.join(d.out_dir, 'dem', outname)
        cmd = ['gdalwarp', '-cutline', projected_county, '-tr', '30', '30', '-tap', '-crop_to_cutline',
               '-cwhere', sql, '-ot', 'Float32', '-overwrite', slope, county_slope]

        subprocess.check_call(cmd)

        print('binning')
        # slice into 100 bins of equal area. save as <filename>
        # qcut: https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.qcut.html#pandas.qcut

        # read in raster as array
        cty_slope_imp_arr = get_array(county_slope_imp11_ext)
        cty_slope_arr = get_array(county_slope)

        # flatten
        cty_slope_arr_flat = cty_slope_imp_arr.flatten()

        # remove no data value -9999 by getting everything >= 0
        cty_slope_arr_flat = cty_slope_arr_flat[cty_slope_arr_flat > 0]

        print('min: {}, max: {}'.format(cty_slope_arr_flat.min(), cty_slope_arr_flat.max()))
        ser, bins = pd.qcut(cty_slope_arr_flat, 100, retbins=True)

        # get 100th bin. this is the 2nd to last item in array. the last one is the max value of array
        s_threshold = bins[-2]

        co_slope99_array = np.where((cty_slope_arr < s_threshold) & (cty_slope_arr >= 0), 1, -9999)

        co_slope99 = os.path.join(d.out_dir, 'dem', 'slope99_{}_{}.tif'.format(d.state, county))

        array_to_rast(county_slope, co_slope99_array, co_slope99, -9999)

        # make list of each slope99 array so we can add them all together to create statewide raster
        slope99_array_list.append(co_slope99)

    # combine each array to make statewide array
    state_file = '{}_cat9.tif'.format(d.state)
    state_data = os.path.join(d.out_dir, 'dem', state_file)

    cmd = ['gdal_merge.py', '-o', state_data, '-co', 'COMPRESS=LZW', '-a_nodata', '-9999', '-ot', 'Int32']

    for co in slope99_array_list:
        cmd.append(co)

    subprocess.check_call(cmd)

    d.slope99 = state_data