'''utils.py by Lucia Gordon'''

# imports
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import yaml

# functions
def process_yaml(path):
    '''Extracts the data from a YAML file'''

    with open(path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    return yaml_data

def get_big_project_dir():
    '''Gets path to the project directory'''

    config = process_yaml('config.yaml')

    return config['big_project_dir']

def get_model_dir():
    '''Gets path to the project directory'''

    config = process_yaml('config.yaml')

    return config['model_dir']

def get_site():
    '''Gets name of the site being used'''

    config = process_yaml('config.yaml')

    return config['site']

def get_site_dir(site=None):
    '''Gets path to the site being used'''

    config = process_yaml('config.yaml')

    if site is not None:
        return f'{config["big_project_dir"]}/{site}'

    return f'{config["big_project_dir"]}/{config["site"]}'

def get_nonzero_bounds(array):
    '''Gets bounds corresponding to the nonzero section of an array'''

    if len(array.shape) == 2:
        rows, cols = array.nonzero()
        top = np.min(rows)
        bottom = np.max(rows)
        left = np.min(cols)
        right = np.max(cols)

    elif len(array.shape) == 3:
        tops, bottoms, lefts, rights = [], [], [], []

        for band in array.transpose(2, 0, 1):
            top, bottom, left, right = get_nonzero_bounds(band)
            tops.append(top)
            bottoms.append(bottom)
            lefts.append(left)
            rights.append(right)

        top = np.min(tops)
        bottom = np.max(bottoms)
        left = np.min(lefts)
        right = np.max(rights)

    return [top, bottom, left, right]

def pixels_to_meters(origin, pixels, res):
    return origin + pixels * res

def meters_to_pixels(origin, meters, res):
    return int((meters - origin) / res)

def get_bounds_meters(bounds_pixels, origin, res):
    top_pixels, bottom_pixels, left_pixels, right_pixels = bounds_pixels

    top_meters = pixels_to_meters(origin[1], top_pixels, res[1])
    bottom_meters = pixels_to_meters(origin[1], bottom_pixels, res[1])
    left_meters = pixels_to_meters(origin[0], left_pixels, res[0])
    right_meters = pixels_to_meters(origin[0], right_pixels, res[0])

    return [top_meters, bottom_meters, left_meters, right_meters]

def get_bounds_pixels(bounds_meters, origin, res):
    top_meters, bottom_meters, left_meters, right_meters = bounds_meters

    top_pixels = meters_to_pixels(origin[1], top_meters, res[1])
    bottom_pixels = meters_to_pixels(origin[1], bottom_meters, res[1])
    left_pixels = meters_to_pixels(origin[0], left_meters, res[0])
    right_pixels = meters_to_pixels(origin[0], right_meters, res[0])

    return [top_pixels, bottom_pixels, left_pixels, right_pixels]

def get_utm_zone(longitude):
    ''' Calculates the UTM zone number for a given longitude '''

    return int((longitude + 180) // 6) + 1

def latlon_to_utm(lat, lon):
    ''' Converts latitude and longitude to UTM coordinates '''

    utm_zone = get_utm_zone(lon)
    hemisphere_prefix = 327 if lat < 0 else 326 # if latitude is below zero, it's in the southern hemisphere
    epsg_code = f'epsg:{hemisphere_prefix}{utm_zone}'
    transformer = pyproj.Transformer.from_crs('epsg:4326', epsg_code, always_xy=True)
    utm_easting, utm_northing = transformer.transform(lon, lat)    

    return utm_easting, utm_northing

def get_tile(identifier):
    constants = process_yaml('constants.yaml')
    num_cols_in_tiling = constants['num_cols_in_tiling']

    row = int(np.floor(identifier / num_cols_in_tiling))
    col = int(identifier - num_cols_in_tiling*np.floor(identifier / num_cols_in_tiling))

    return row, col

def get_tile_center_pixels(identifier, modality):
    constants = process_yaml('constants.yaml')
    interval = constants[f'{modality}_interval']    
    row, col = get_tile(identifier)
    x = int(col*interval + interval/2)
    y = int(row*interval + interval/2)

    return y, x

def get_tile_center_meters(identifier):
    constants = process_yaml('constants.yaml')
    interval = constants['interval_meters']    
    row, col = get_tile(identifier)
    x = constants['tight_bounds_meters']['left'] + col*interval + interval/2
    y = constants['tight_bounds_meters']['top'] - row*interval - interval/2

    return y, x

def plot_array(array, png_path, modality):
    '''Plots an array as a PNG'''

    plt.figure(dpi=300)
    image = plt.imshow(array) # plot the array of pixel values as an image

    if modality == 'thermal' or modality == 'lidar':
        image.set_cmap('inferno')
    elif modality != 'rgb':
        image.set_cmap('gray')

    plt.axis('off') # remove axes        
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
    plt.close() # close the image to save memory

def plot_tiff(tiff_path, png_path):
    '''Converts a TIFF to a PNG'''

    with rasterio.open(tiff_path) as tiff:
        array = tiff.read(1) # use "1" as the argument in .read() for LiDAR and "4" for thermal

    plot_array(array, png_path)
