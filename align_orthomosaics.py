# imports
from osgeo import gdal
from PIL import Image
import numpy as np
import os
import rasterio
import shutil
import sys
import utils
import yaml

class AlignOrthomosaics:
    def __init__(self, site):
        self.site_dir = utils.get_site_dir(site)
        self.interval_meters = 20
        self.modalities = ['thermal', 'rgb', 'lidar']
        self.data = {}

        for modality in self.modalities:
            self.data[modality] = {}

            with rasterio.open(f'{self.site_dir}/tiffs/{modality}.tif') as tiff:
                self.data[modality]['res_x'], _, self.data[modality]['left_meters'], _, self.data[modality]['res_y'], self.data[modality]['top_meters'], _, _, _ = tiff.transform
                self.data[modality]['res_x'], self.data[modality]['res_y'] = round(self.data[modality]['res_x'], 2), round(self.data[modality]['res_y'], 2)
                self.data[modality]['metadata'] = tiff.meta
                self.data[modality]['orthomosaic'] = self.get_orthomosaic(modality, tiff)
                self.data[modality]['interval'] = self.get_interval(resolution=self.data[modality]['res_x'])
                print(f'Initial {modality} orthomosaic shape = {self.data[modality]["orthomosaic"].shape}')

        self.max_res = np.min([self.data[modality]['res_x'] for modality in self.modalities])
        self.tight_bounds_meters = self.get_tight_bounds_meters()
        self.crop_to_tight_bounds()
        self.pad_for_divisibility_by_interval()
        self.save_orthomosaics_as_arrays()
        self.save_orthomosaics_as_tiffs()
        self.upsample_tiffs()
        self.save_upsampled_orthomosaics_as_arrays()
        self.rgb_array_to_grayscale()
        self.save_constants()

    def get_orthomosaic(self, modality, tiff):
        if modality == 'thermal':
            orthomosaic = tiff.read(4) # 4th band holds the thermal data
            background_value = 2000 # the thermal data is above the background value
            orthomosaic[orthomosaic <= background_value] = 0 # sets all background pixels to 0
            min_non_background_value = np.min(orthomosaic[orthomosaic > 0])
            orthomosaic[orthomosaic > 0] -= min_non_background_value-1 # downshifts the thermal values such that their minimum is 1
        elif modality == 'rgb':
            orthomosaic = tiff.read().transpose(1, 2, 0)
        elif modality == 'lidar':
            orthomosaic = tiff.read(1)
            orthomosaic[orthomosaic < 0] = 0

        return orthomosaic

    def get_interval(self, resolution):
        return int(self.interval_meters / resolution)

    def get_tight_bounds_meters(self):
        nonzero_bounds = {'pixels': {}, 'meters': {}}

        for modality in self.modalities:
            nonzero_bounds['pixels'][modality] = {}
            nonzero_bounds['pixels'][modality]['top'], nonzero_bounds['pixels'][modality]['bottom'], nonzero_bounds['pixels'][modality]['left'], nonzero_bounds['pixels'][modality]['right'] = utils.get_nonzero_bounds(array=self.data[modality]['orthomosaic'])

            nonzero_bounds['meters'][modality] = {}
            nonzero_bounds['meters'][modality]['top'], nonzero_bounds['meters'][modality]['bottom'], nonzero_bounds['meters'][modality]['left'], nonzero_bounds['meters'][modality]['right'] = utils.get_bounds_meters(bounds_pixels=[nonzero_bounds['pixels'][modality]['top'],
                                                                                                                                                                                                                                      nonzero_bounds['pixels'][modality]['bottom'],
                                                                                                                                                                                                                                      nonzero_bounds['pixels'][modality]['left'],
                                                                                                                                                                                                                                      nonzero_bounds['pixels'][modality]['right']],
                                                                                                                                                                                                                       origin=[self.data[modality]['left_meters'], self.data[modality]['top_meters']],
                                                                                                                                                                                                                       res=[self.data[modality]['res_x'], self.data[modality]['res_y']])        
        
        tight_bounds_meters = {'top': np.min([nonzero_bounds['meters'][modality]['top'] for modality in self.modalities]),
                               'bottom': np.max([nonzero_bounds['meters'][modality]['bottom'] for modality in self.modalities]),
                               'left': np.max([nonzero_bounds['meters'][modality]['left'] for modality in self.modalities]),
                               'right': np.min([nonzero_bounds['meters'][modality]['right'] for modality in self.modalities])}

        print(f'Tight bounds in meters = {tight_bounds_meters}')

        return tight_bounds_meters

    def crop_to_tight_bounds(self):
        tight_bounds_pixels = {}

        for modality in self.modalities:
            tight_bounds_pixels[modality] = {}
            tight_bounds_pixels[modality]['top'], tight_bounds_pixels[modality]['bottom'], tight_bounds_pixels[modality]['left'], tight_bounds_pixels[modality]['right'] = utils.get_bounds_pixels(bounds_meters=[self.tight_bounds_meters['top'],
                                                                                                                                                                                                                  self.tight_bounds_meters['bottom'],
                                                                                                                                                                                                                  self.tight_bounds_meters['left'],
                                                                                                                                                                                                                  self.tight_bounds_meters['right']],
                                                                                                                                                                                                   origin=[self.data[modality]['left_meters'], self.data[modality]['top_meters']],
                                                                                                                                                                                                   res=[self.data[modality]['res_x'], self.data[modality]['res_y']])
            self.data[modality]['orthomosaic'] = self.data[modality]['orthomosaic'][tight_bounds_pixels[modality]['top'] : tight_bounds_pixels[modality]['bottom'], tight_bounds_pixels[modality]['left'] : tight_bounds_pixels[modality]['right']]
            print(f'Post-cropping {modality} orthomosaic shape = {self.data[modality]["orthomosaic"].shape}')

    def pad_for_divisibility_by_interval(self):
        for modality in self.modalities:
            if len(self.data[modality]['orthomosaic'].shape) == 2:
                pad_width = ((0, self.data[modality]['interval'] - self.data[modality]['orthomosaic'].shape[0]%self.data[modality]['interval']), (0, self.data[modality]['interval'] - self.data[modality]['orthomosaic'].shape[1]%self.data[modality]['interval']))
            else:
                pad_width = ((0, self.data[modality]['interval'] - self.data[modality]['orthomosaic'].shape[0]%self.data[modality]['interval']), (0, self.data[modality]['interval'] - self.data[modality]['orthomosaic'].shape[1]%self.data[modality]['interval']), (0,0))

            self.data[modality]['orthomosaic'] = np.pad(array=self.data[modality]['orthomosaic'], pad_width=pad_width)
            print(f'Post-padding {modality} orthomosaic shape = {self.data[modality]["orthomosaic"].shape}')

    def save_orthomosaics_as_arrays(self):
        os.makedirs(f'{self.site_dir}/orthomosaics', exist_ok=True)

        for modality in self.modalities:
            np.save(f'{self.site_dir}/orthomosaics/{modality}_orthomosaic', self.data[modality]['orthomosaic'])

        print(f'Orthomosaics saved as numpy arrays')

    def array_to_tiff(self, modality):
        metadata = self.data[modality]['metadata'].copy()
        array = self.data[modality]['orthomosaic']
        metadata['nodata'] = 0
        metadata['width'] = array.shape[1]
        metadata['height'] = array.shape[0]
        metadata['count'] = 1 if len(array.shape) == 2 else 3
        metadata['transform'] = rasterio.transform.Affine(self.data[modality]['res_x'], 0.0, self.tight_bounds_meters['left'], 0.0, self.data[modality]['res_y'], self.tight_bounds_meters['top'])

        with rasterio.open(f'{self.site_dir}/aligned_tiffs/{modality}.tif', 'w', **metadata) as tiff_file:
            if len(array.shape) == 2:
                tiff_file.write(array, 1)
            elif len(array.shape) == 3:
                for i, band in enumerate(array.transpose(2, 0, 1)):
                    tiff_file.write(band, i+1)

    def save_orthomosaics_as_tiffs(self):
        os.makedirs(f'{self.site_dir}/aligned_tiffs', exist_ok=True)

        for modality in self.modalities:
            self.array_to_tiff(modality)

        print(f'Orthomosaics saved as tiffs')

    def upsample_tiffs(self):
        os.makedirs(f'{self.site_dir}/upsampled_tiffs', exist_ok=True)

        for modality in self.modalities:
            if self.data[modality]['res_x'] == self.max_res:
                shutil.copy(f'{self.site_dir}/aligned_tiffs/{modality}.tif', f'{self.site_dir}/upsampled_tiffs/{modality}_upsampled.tif')
            else:
                gdal.Warp(destNameOrDestDS=f'{self.site_dir}/upsampled_tiffs/{modality}_upsampled.tif',
                          srcDSOrSrcDSTab=f'{self.site_dir}/aligned_tiffs/{modality}.tif',
                          xRes=self.max_res,
                          yRes=self.max_res,
                          resampleAlg='cubic',
                          srcNodata=0,
                          dstNodata=0)

            with rasterio.open(f'{self.site_dir}/upsampled_tiffs/{modality}_upsampled.tif') as tiff:
                print(f'Upsampled {modality} tiff shape = {tiff.read().shape}')
                print(f'Upsampled {modality} tiff resolution = {tiff.res}')

    def save_upsampled_orthomosaics_as_arrays(self):
        os.makedirs(f'{self.site_dir}/upsampled_orthomosaics', exist_ok=True)

        for modality in self.modalities:
            with rasterio.open(f'{self.site_dir}/upsampled_tiffs/{modality}_upsampled.tif') as tiff:
                upsampled_orthomosaic = tiff.read(1) if tiff.count == 1 else tiff.read().transpose(1, 2, 0)

                if np.min(upsampled_orthomosaic) < 0:
                    upsampled_orthomosaic[upsampled_orthomosaic != 0] -= np.min(upsampled_orthomosaic)

                np.save(f'{self.site_dir}/upsampled_orthomosaics/{modality}_upsampled_orthomosaic', upsampled_orthomosaic)

        print(f'Upsampled orthomosaics saved as numpy arrays')

    def rgb_array_to_grayscale(self):
        print((self.data['rgb']['orthomosaic'] == np.load(f'{self.site_dir}/upsampled_orthomosaics/rgb_upsampled_orthomosaic.npy')).all())

        grayscale_orthomosaic = np.array(Image.fromarray(self.data['rgb']['orthomosaic'], 'RGB').convert('L'))
        print(f'Grayscale orthomosaic shape = {grayscale_orthomosaic.shape}')
        np.save(f'{self.site_dir}/upsampled_orthomosaics/grayscale_upsampled_orthomosaic', grayscale_orthomosaic)

    def save_constants(self):
        with open('constants.yaml', 'w') as yaml_file:
            constants = {**{f'{modality}_res_x': self.data[modality]['res_x'] for modality in self.modalities},
                         **{f'{modality}_res_y': self.data[modality]['res_y'] for modality in self.modalities},
                         **{f'{modality}_interval': self.data[modality]['interval'] for modality in self.modalities},
                         **{'interval_meters': self.interval_meters,
                            'num_rows_in_tiling': int(self.data['thermal']['orthomosaic'].shape[0] / self.data['thermal']['interval']),
                            'num_cols_in_tiling': int(self.data['thermal']['orthomosaic'].shape[1] / self.data['thermal']['interval']),
                            'upsampled_interval': self.get_interval(resolution=self.max_res)},
                            'tight_bounds_meters': {key: float(value) for key, value in self.tight_bounds_meters.items()}}

            yaml.dump(constants, yaml_file, default_flow_style=False)

        print('Constants saved as YAML')

if __name__ == '__main__':
    AlignOrthomosaics(site=sys.argv[1])
