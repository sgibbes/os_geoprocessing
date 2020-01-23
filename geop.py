import os
import shutil
import subprocess
import pysal
import csv
import rasterio
import math
import numpy as np
from osgeo import gdal
import geopandas as gpd
import pandas as pd
import sys
from simpledbf import Dbf5

from utilities import write_numpy_to_raster
from utilities import whitebox


def wkt():
    return 'PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",' \
          'GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.2572221010042,AUTHORITY' \
          '["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],' \
          'AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",29.5],' \
          'PARAMETER["standard_parallel_2",45.5],PARAMETER["latitude_of_center",23],PARAMETER' \
          '["longitude_of_center",-96],PARAMETER["false_easting",0],PARAMETER["false_northing",0],' \
          'UNIT["metre",1,AUTHORITY["EPSG","9001"]]]'


def project_rast_102039(input, output):
    input_r = gdal.Open(input)

    gdal.Warp(output, input_r, dstSRS=wkt())


def proj_rast_102039_w_tr(in_raster, extent=None):

    projected = in_raster.replace('.tif', '_proj.tif')
    cmd = ['gdalwarp', '-t_srs', 'EPSG:102039', '-co', 'COMPRESS=LZW', '-tr', '30', '30', '-tap', '-overwrite']

    if extent:
        ulx, uly, lrx, lry = write_numpy_to_raster.get_extent(extent)

        cmd += ['-te', ulx, lry, lrx, uly]

    cmd += [in_raster, projected]
    print(cmd)

    subprocess.check_call(cmd)

    return projected


def assign_srs_102039(input):
    cmd = ['gdal_edit.py', '-a_srs', wkt(), input]
    subprocess.check_call(cmd)


def project_rast(in_raster, t_srs, res):
    projected = in_raster.replace('.tif', '_proj.tif')

    cmd = ['gdalwarp', '-s_srs', 'EPSG:4326', '-t_srs', 'EPSG:102039', '-co', 'COMPRESS=LZW', '-tr', res, res, '-tap', '-overwrite',
           in_raster, projected]

    print(cmd)

    subprocess.check_call(cmd)


    return projected


def project(in_file, out_dir, newname=None, roads=False):

    print('projecting: {}'.format(in_file))
    infile_basename = os.path.basename(in_file)
    infile_name = infile_basename.split('.')[0]

    if newname:
        new_filename = newname
    else:
        new_filename = infile_name

    projected_file = '{}_proj.shp'.format(new_filename)
    projected_file_path = os.path.join(out_dir, projected_file)

    # try:
    # projection cmd
    cmd = ['ogr2ogr', '-t_srs', 'EPSG:102039']

    # don't keep any fields but geoid10
    if 'tract10' in in_file:

        cmd += ['-sql', 'SELECT GEOID10 from {}'.format(infile_name)]

    if roads:
        cmd += ['-sql', 'SELECT FUNC_CLASS, FERRY_TYPE FROM {}'.format(infile_name)]
    # add on the in/out file
    cmd += [projected_file_path, in_file]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print('error')

    return projected_file_path


def diss_shp(input_shape, output_diss):
    basename = os.path.basename(input_shape).split('.')[0]
    cmd = ['ogr2ogr', output_diss, input_shape, '-dialect', 'sqlite', '-sql',
           'SELECT ST_Union(geometry) FROM {}'.format(basename)]

    subprocess.check_call(cmd)


def export_state(d, census_outdir):
    # dissolve d.county to make state
    county = d.county

    # get the projected county file
    county_proj = os.path.basename(county).replace('.shp', '_proj.shp')
    county_proj = os.path.join(census_outdir, county_proj)

    # create the output state name
    out_state_name = '{}.shp'.format(d.state)  # de.shp
    out_state_file = os.path.join(census_outdir, out_state_name)  # /data/output_data/census/de.shp

    # get the basename of the county file
    input = os.path.basename(county_proj).split('.')[0]
    cmd = ['ogr2ogr', out_state_file, county_proj, '-dialect', 'sqlite', '-sql',
           'SELECT ST_Union(geometry) FROM {}'.format(input)]

    subprocess.check_call(cmd)

    return out_state_file


def poly_to_raster_already102039(poly, outname, attribute=None, sql=None, extent=None, ot=None):
    # the other function- poly_to_raster was throwing an error when running block_density2. if take out
    # the -a_srs, it works.
    # rasterize the polygon. If we want to burn in values from the poly, use attribute
    cmd = ['gdal_rasterize', '-tap', '-tr', '30', '30', '-a_nodata', '0']

    if attribute:
        cmd += ['-a', attribute]
    else:
        cmd += ['-burn', '1']

    if sql:
        cmd += ['-dialect', 'sqlite', '-sql', sql]

    if extent:
        ulx, uly, lrx, lry = write_numpy_to_raster.get_extent(extent)
        print(extent)

        cmd += ['-te', ulx, lry, lrx, uly]

    if ot:
        cmd += ['-ot', ot]

    layername = os.path.basename(poly).split('.')[0]
    cmd += ['-l', layername, poly, outname]

    subprocess.check_call(cmd)

    return outname


def get_proj_file_name(out_dir, in_file):
    infile_basename = os.path.basename(in_file)
    infile_name = infile_basename.split('.')[0]

    projected_file = '{}_proj.shp'.format(infile_name)

    return os.path.join(out_dir, projected_file)


def csv_of_blocks(d, in_file):
    print('creating csv of blocks')
    projected_file_path = get_proj_file_name(os.path.join(d.out_dir, 'census'), in_file)
    print(projected_file_path)
    blocks_dbf = projected_file_path.replace('.shp', '.dbf')

    # set proj
    print('reading in file')
    dbf = Dbf5(blocks_dbf)
    df = dbf.to_dataframe()

    # create dict to hold dtypes for columns
    convert_dict = {}

    # convert these to int64
    columns_list = ['STATEFP10', 'COUNTYFP10', 'TRACTCE10', 'BLOCKCE10', 'GEOID10']

    for c in columns_list:
        convert_dict[c] = 'int64'

    # conver these to float64
    columns_list = ['INTPTLAT10', 'INTPTLON10']
    for c in columns_list:
        convert_dict[c] = 'float64'

    # use conversion dict to set dtype of columns
    df = df.astype(convert_dict)

    # save df to csv
    out_csv = projected_file_path.replace('.shp', '.csv')
    df.to_csv(out_csv, index=False)


def poly_to_raster(poly, outname, attribute=None, sql=None, extent=None, ot=None):

    # rasterize the polygon. If we want to burn in values from the poly, use attribute
    cmd = ['gdal_rasterize', '-a_srs', 'EPSG:102039', '-tap', '-tr', '30', '30',
           '-a_nodata', '0']

    if attribute:
        cmd += ['-a', attribute]
    else:
        cmd += ['-burn', '1']

    if sql:
        cmd += ['-dialect', 'sqlite', '-sql', sql]

    if extent:
        ulx, uly, lrx, lry = write_numpy_to_raster.get_extent(extent)
        print(extent)

        cmd += ['-te', ulx, lry, lrx, uly]

    if ot:
        cmd += ['-ot', ot]

    layername = os.path.basename(poly).split('.')[0]
    cmd += ['-l', layername, poly, outname]

    subprocess.check_call(cmd)

    return outname


def proj_wgs84(in_tif, out_tif, extent=None, nlcd=False):

    cmd = ['gdalwarp', '-t_srs', 'EPSG:4326', '-co', 'COMPRESS=LZW', '-overwrite']

    if extent:
        ulx, uly, lrx, lry = write_numpy_to_raster.get_extent(extent)
        cmd += ['-te', ulx, lry, lrx, uly]

    if not nlcd:
        cmd += ['-tr', '.00025', '.00025', '-tap']

    cmd += [in_tif, out_tif]

    subprocess.check_call(cmd)


def clip_raster(template_ras, clipsrc, in_raster, clipped_raster, nlcd=False):

    ulx, uly, lrx, lry = write_numpy_to_raster.get_extent(template_ras)
    cmd = ['gdalwarp', '-cutline', clipsrc, '-crop_to_cutline', '-te', ulx, lry, lrx, uly, '-tr', '30', '30', '-tap',
           '-t_srs', 'EPSG:102039', '-dstnodata', '0', '-overwrite', '-co', 'COMPRESS=LZW']

    cmd += [in_raster, clipped_raster]

    print(cmd)
    subprocess.check_call(cmd)


def neighbors_pysal(blocks, d):
    # uses pysal to get all touching neighbors for each shapefile. w.neighbors is the dictionary
    # https://stackoverflow.com/a/27993912/5272945
    print('calculating queen neighbors')
    w = pysal.queen_from_shapefile(blocks, idVariable='GEOID10')

    neighbors_dict = w.neighbors

    write_csv(neighbors_dict, d)


def write_csv(input_dict, d):
    print('writing csv')
    fips = d.fips
    out_dir = d.out_dir

    # create the name of the csv
    neighbors_csv = 'block_neighbors_{}.csv'.format(fips)

    neighbors_csv = os.path.join(out_dir, 'census', neighbors_csv)

    print('polygon neighbors csv: {}'.format(neighbors_csv))
    # open the csv for writing
    with open(neighbors_csv, 'w') as f:
        writer = csv.writer(f)
        # dictionary is form of {key: [val, val, val], key: [val, val, val]}
        writer.writerow(('src_OBJECTID', 'nbr_OBJECTID'))

        for row in input_dict.items():
            # get the source geoid
            key = row[0]
            # iterate over all the neighbors and write those
            for v in row[1]:
                writer.writerow((key, v))


def calc_imp01_imp11(imp01, imp11, output_tif):
    print('calculating imp01, imp11')
    print('reading in {}'.format(imp01))
    with rasterio.open(imp01) as src:
        imp01_data = src.read(1)
        profile = src.profile

    print('reading in {}'.format(imp11))
    with rasterio.open(imp11) as src:
        imp11_data = src.read(1)

    new_data = 1 * ((imp01_data == 0) | (imp11_data > 0))

    # array was saved as 64bit when values are just 0, 1. So, need to re-define that it is int
    new_data = new_data.astype(rasterio.uint8)

    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(new_data, 1)

    print('setting nodata')
    # because I don't know how to set nodata in rio yet:
    cmd = ['gdal_edit.py', '-a_nodata', '0', output_tif]
    subprocess.check_call(cmd)


def rasterio_recalc(d, input_tif, output_tif, new_val=None, lower_val=None, higher_val=None):
    # https://gis.stackexchange.com/questions/269610/raster-calculation-using-rasterio

    # read the input tif. this sets properties for writing output file
    print('reading in {}'.format(input_tif))
    with rasterio.open(input_tif) as src:
        data = src.read(1)
        profile = src.profile

    # if the new value is set by the user, ex:1, make this the value that the evalutation is multiplied by
    if new_val:
        new_value = new_val
    else:
        new_value = data

    print('min value of input raster: {}'.format(data.min()))
    print('max value of input raster: {}'.format(data.max()))
    # perform statement to decide which values to keep, which go null
    # this one is for lc01, lc11
    print('setting new values: lower: {}, higher: {}'.format(lower_val, higher_val))

    if lower_val and higher_val:

        new_data = new_value * ((data > lower_val) & (data < higher_val))

    # this one is for impervious
    elif higher_val:

        new_data = new_value * (data < higher_val)

    else:

        new_data = new_value * (data > lower_val)

    # for some reason, array was saved as 64bit when values are just 0, 1. So, need to re-define that it is int
    if type(new_val) is int:
        new_data = new_data.astype(rasterio.uint8)

    print('min value in new data: {}'.format(new_data.min()))
    print('max value in new data: {}'.format(new_data.max()))
    print('writing new data: {}'.format(output_tif))
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(new_data, 1)

    # print 'setting nodata'
    # # because I don't know how to set nodata in rio yet:
    # cmd = ['gdal_edit.py', '-a_nodata', '0', output_tif]
    # subprocess.check_call(cmd)


def get_pixel_area(lat):
    # borrowing from: https://github.com/wri/gfw-annual-loss-processing/blob/

    """
    Calculate geodesic area for Hansen data, assuming a fix pixel size of 0.00025 * 0.00025 degree
    using WGS 1984 as spatial reference.
    Pixel size various with latitude, which is it the only input parameter
    """
    a = 6378137.0  # Semi major axis of WGS 1984 ellipsoid
    b = 6356752.314245179  # Semi minor axis of WGS 1984 ellipsoid

    d_lat = 0.00025  # pixel hight
    d_lon = 0.00025  # pixel width

    pi = math.pi

    q = d_lon / 360
    e = math.sqrt(1 - (b / a) ** 2)

    area = abs(
        (pi * b ** 2 * (
            2 * np.arctanh(e * np.sin(np.radians(lat + d_lat))) /
            (2 * e) +
            np.sin(np.radians(lat + d_lat)) /
            ((1 + e * np.sin(np.radians(lat + d_lat))) * (1 - e * np.sin(np.radians(lat + d_lat)))))) -
        (pi * b ** 2 * (
            2 * np.arctanh(e * np.sin(np.radians(lat))) / (2 * e) +
            np.sin(np.radians(lat)) / ((1 + e * np.sin(np.radians(lat))) * (1 - e * np.sin(np.radians(lat))))))) * q

    return area


def create_vrt(input_tsv, output_vrt):
    ogr_layer_name = os.path.splitext(os.path.basename(input_tsv))[0]
    input_tsv_base = os.path.basename(input_tsv)

    vrt_text = '''<OGRVRTDataSource>
            <OGRVRTLayer name="data">
                <SrcDataSource>{}</SrcDataSource>
                <SrcLayer>{}</SrcLayer>
                <GeometryType>wkbPoint</GeometryType>
                <LayerSRS>WGS84</LayerSRS>
                <GeometryField encoding="PointFromColumns" x="lon" y="lat"/>
            </OGRVRTLayer>
        </OGRVRTDataSource>'''.format(input_tsv_base, ogr_layer_name)

    with open(output_vrt, 'w') as thefile:
        thefile.write(vrt_text)


def clip_poly(in_poly, clip_src, clipped_poly, sql=None):
    print('clipping poly')

    cmd = ['ogr2ogr', '-clipsrc', clip_src]

    if sql:
        cmd += ['-dialect', 'sqlite', '-sql', sql]

    cmd += [clipped_poly, in_poly,  '-skipfailures']

    subprocess.check_call(cmd)


def wbt_cd_preprocess(d, cost_ras):
    # create cost raster where 0 -> 100
    cost_raster_arr = write_numpy_to_raster.get_array(cost_ras)

    de_raster = write_numpy_to_raster.get_array(d.state_raster)

    # change 0's to 100
    new_a = np.where(cost_raster_arr == 0, 100, cost_raster_arr)

    # clip to raster that excludes the background
    new_a2 = np.where(de_raster == 1, new_a, 0)

    cost_raster_new = cost_ras.replace('.tif', '_cost_ras.tif')
    write_numpy_to_raster.array_to_rast(cost_ras, new_a2, cost_raster_new, -9999)

    return cost_raster_new


def hs_preprocess(d, ttime_landcover):

    # create cost raster where 0 -> 100
    cost_raster_arr = write_numpy_to_raster.get_array(ttime_landcover)

    de_raster = write_numpy_to_raster.get_array(d.state_raster)

    new_a = np.where(cost_raster_arr == 0, 100, cost_raster_arr)
    new_a2 = np.where(de_raster == 1, new_a, 0)

    cost_raster_new = os.path.join(d.out_dir, 'cost_distance', 'cost_raster_new.tif')
    write_numpy_to_raster.array_to_rast(ttime_landcover, new_a2, cost_raster_new, 0)

    return cost_raster_new, new_a2

def wbt_cd_postprocess(d, cd_ras, cost_ras, cd_out):
    # mask output cost distance using travel time raster
    travel_time_landcover_f_arr = write_numpy_to_raster.get_array(cost_ras)
    cost_dist_out_arr = write_numpy_to_raster.get_array(cd_ras)

    # write final cost distance raster
    new_a = np.where(travel_time_landcover_f_arr > 0, cost_dist_out_arr, -9999)
    write_numpy_to_raster.array_to_rast(cd_ras, new_a, cd_out, -9999, gdal.GDT_Float32)


def wbt_postproc(input_ras):
    # assign projection info to cd output
    assign_srs_102039(input_ras)


def buff_raster(d, dist, in_ras):
    dist = float(dist)

    # unset nodata for whitebox tools
    nodata_ras = in_ras.replace('.tif', '_nodata.tif')
    shutil.copy(in_ras, nodata_ras)

    cmd = ['gdal_edit.py', '-unsetnodata', nodata_ras]
    subprocess.check_call(cmd)

    euc_dist_temp = in_ras.replace('.tif', '_euc_temp.tif')
    whitebox.buffer_rast(nodata_ras, euc_dist_temp, dist)

    os.remove(nodata_ras)
    print('buffered raster: {}'.format(euc_dist_temp))
    # clip back to size of state
    euc_dist = in_ras.replace('.tif', '_euc.tif')

    cmd = ['gdal_calc.py', '-A', euc_dist_temp, '-B', d.state_raster, '--calc=A*(B==1)',
           '--outfile={}'.format(euc_dist), '--overwrite', '--co=COMPRESS=LZW', '--NoDataValue=0']
    subprocess.check_call(cmd)

    print('finished with buffering: {}'.format(euc_dist))

    return euc_dist
