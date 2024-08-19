# imports
from cv2 import imread
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import utils

class TileOrthomosaics:
    def __init__(self, site):
        self.site_dir = utils.get_site_dir(site)
        self.modalities = ['thermal', 'rgb', 'lidar']
        self.constants = utils.process_yaml('constants.yaml')
        self.data = {}

        for modality in self.modalities:
            self.data[modality], self.data['grayscale'] = {}, {}
            self.data[modality]['tiles'] = self.tile(array_path=f'{self.site_dir}/orthomosaics/{modality}_orthomosaic.npy',
                                                     interval=self.constants[f'{modality}_interval'])
            print(f'Generated {modality} tiles')

            self.data[modality]['upsampled_tiles'] = self.tile(array_path=f'{self.site_dir}/upsampled_orthomosaics/{modality}_upsampled_orthomosaic.npy',
                                                               interval=self.constants['upsampled_interval'])
            print(f'Generated {modality} upsampled tiles')

        self.data['grayscale']['upsampled_tiles'] = self.tile(array_path=f'{self.site_dir}/upsampled_orthomosaics/grayscale_upsampled_orthomosaic.npy',
                                                              interval=self.constants['upsampled_interval'])

        self.identifiers = list(range(len(self.data['thermal']['tiles'])))
        self.identifier_matrix = np.zeros((self.constants['num_rows_in_tiling'], self.constants['num_cols_in_tiling']), dtype=np.int_)

        for row in range(self.constants['num_rows_in_tiling']):
            for col in range(self.constants['num_cols_in_tiling']):
                self.identifier_matrix[row][col] = self.identifiers[row*self.constants['num_cols_in_tiling'] + col]

        self.remove_empty_tiles()
        self.save_tiles()
        self.generate_png_tiles()
        self.generate_fused_tiles()

        np.save(f'{self.site_dir}/identifiers', self.identifiers)
        np.save(f'{self.site_dir}/identifier_matrix', self.identifier_matrix)

    def tile(self, array_path, interval):
        array = np.load(array_path)
        tiles = []

        for top in range(0, array.shape[0], interval):
            for left in range(0, array.shape[1], interval):
                tiles.append(array[top : top + interval, left : left + interval])

        return tiles

    def remove_empty_tiles(self):
        for i in reversed(range(len(self.identifiers))):
            if np.all(self.data['thermal']['tiles'][i] == 0) or np.all(self.data['rgb']['tiles'][i] == 0) or np.all(self.data['lidar']['tiles'][i] == 0):
                for modality in self.modalities:
                    del self.data[modality]['tiles'][i]
                    del self.data[modality]['upsampled_tiles'][i]

                del self.data['grayscale']['upsampled_tiles'][i]
                del self.identifiers[i]

        print('Removed empty tiles')

    def save_tiles(self):
        os.makedirs(f'{self.site_dir}/tiles', exist_ok=True)
        os.makedirs(f'{self.site_dir}/upsampled_tiles', exist_ok=True)

        for modality in self.modalities:
            np.save(f'{self.site_dir}/tiles/{modality}_tiles', self.data[modality]['tiles'])
            np.save(f'{self.site_dir}/upsampled_tiles/{modality}_upsampled_tiles', self.data[modality]['upsampled_tiles'])

        np.save(f'{self.site_dir}/upsampled_tiles/grayscale_upsampled_tiles', self.data['grayscale']['upsampled_tiles'])

        print('Tiles and upsampled tiles saved')

    def generate_png_tiles(self):
        for modality in self.modalities:
            os.makedirs(f'{self.site_dir}/png_images/{modality}', exist_ok=True)

            png_tiles = []

            for i in range(len(self.data[modality]['tiles'])):
                plt.figure(dpi=60.7) # dpi=60.7 to get resultant arrays of (224,224,3)
                image = plt.imshow(self.data[modality]['tiles'][i]) # plot the array of pixel values as an image

                if modality == 'thermal' or modality == 'lidar':
                    image.set_cmap('inferno')

                plt.axis('off') # remove axes
                plt.savefig(f'{self.site_dir}/png_images/{modality}/{modality}_png_image_{self.identifiers[i]}.png', bbox_inches='tight', pad_inches=0) # save the tile as a PNG
                plt.close() # close the image to save memory
                png_tiles.append(imread(f'{self.site_dir}/png_images/{modality}/{modality}_png_image_{self.identifiers[i]}.png')) # convert the PNG image to a 3D array

            np.save(f'{self.site_dir}/png_tiles/{modality}_png_tiles', png_tiles)

            print(f'Generated {modality} PNG tiles')

    def generate_fused_tiles(self):
        os.makedirs(f'{self.site_dir}/fused_images', exist_ok=True)

        fused_tiles = []

        for i in range(len(self.data['thermal']['tiles'])):
            thermal_image = Image.open(f'{self.site_dir}/png_images/thermal/thermal_png_image_{self.identifiers[i]}.png')
            rgb_image = Image.open(f'{self.site_dir}/png_images/rgb/rgb_png_image_{self.identifiers[i]}.png')
            thermal_rgb_fused_image = Image.blend(thermal_image, rgb_image, 1/2)
            lidar_image = Image.open(f'{self.site_dir}/png_images/lidar/lidar_png_image_{self.identifiers[i]}.png')
            fused_image = Image.blend(thermal_rgb_fused_image, lidar_image, 1/3)
            fused_image.save(f'{self.site_dir}/fused_images/fused_image_{self.identifiers[i]}.png', 'PNG')
            fused_tiles.append(imread(f'{self.site_dir}/fused_images/fused_image_{self.identifiers[i]}.png'))

        np.save(f'{self.site_dir}/fused_tiles', fused_tiles)

        print(f'Generated fused tiles')

if __name__ == '__main__':
    TileOrthomosaics(site=sys.argv[1])
