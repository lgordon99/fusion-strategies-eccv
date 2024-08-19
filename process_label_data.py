# imports
from shapely.geometry import Polygon, Point
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import shutil
import sys
import utils

class ProcessLabelData:
    def __init__(self, site):
        self.site_dir = utils.get_site_dir(site)
        self.constants = utils.process_yaml('constants.yaml')
        self.interval_meters = self.constants['interval_meters']
        self.classes = ['empty', 'midden', 'mound', 'water']

        self.generate_tiles_shapefile()

        for image_class in self.classes[1:]:
            self.coordinates_to_shapefile(image_class)

    def generate_tiles_shapefile(self):
        identifiers = np.load(f'{self.site_dir}/identifiers.npy')
        centers_in_meters = []

        for identifier in identifiers:
            y, x = utils.get_tile_center_meters(identifier)
            centers_in_meters.append([x, y])

        polygons = [Polygon([[center[0] - self.interval_meters/2, center[1] - self.interval_meters/2], [center[0] + self.interval_meters/2, center[1] - self.interval_meters/2], [center[0] + self.interval_meters/2, center[1] + self.interval_meters/2], [center[0] - self.interval_meters/2, center[1] + self.interval_meters/2]]) for center in centers_in_meters]
        gdf = gpd.GeoDataFrame(geometry=polygons)
        gdf['id'] = identifiers
        gdf['label'] = len(identifiers) * [0]
        os.makedirs(f'{self.site_dir}/label_data/tiles_shapefile', exist_ok=True)
        gdf.to_file(f'{self.site_dir}/label_data/tiles_shapefile/tiles_shapefile.shp')
        shutil.make_archive(f'{self.site_dir}/label_data/tiles_shapefile', 'zip', f'{self.site_dir}/label_data/tiles_shapefile')

    def coordinates_to_shapefile(self, image_class):
        coordinates_meters = pd.read_csv(f'{self.site_dir}/label_data/{image_class}_coordinates_meters.csv').to_numpy()
        points = [Point(coordinates_meters[row][0], coordinates_meters[row][1]) for row in range(len(coordinates_meters))]
        gdf = gpd.GeoDataFrame(geometry=points)
        gdf['noted'] = len(coordinates_meters) * ['']
        os.makedirs(f'{self.site_dir}/label_data/{image_class}_shapefile', exist_ok=True)
        gdf.to_file(f'{self.site_dir}/label_data/{image_class}_shapefile/{image_class}_shapefile.shp')
        shutil.make_archive(f'{self.site_dir}/label_data/{image_class}_shapefile', 'zip', f'{self.site_dir}/label_data/{image_class}_shapefile')

if __name__ == '__main__':
    ProcessLabelData(site=sys.argv[1])
