# imports
import numpy as np
import pandas as pd
import sys
import utils

def assign_tile_labels(site):
    site_dir = utils.get_site_dir(site)
    constants = utils.process_yaml('constants.yaml')
    image_classes = ['empty', 'midden', 'mound', 'water']
    identifiers = np.load(f'{site_dir}/identifiers.npy')
    identifier_matrix = np.load(f'{site_dir}/identifier_matrix.npy')
    labels = pd.read_csv(f'{site_dir}/label_data/tile_labels.csv').to_numpy().T[1]

    print('Class breakdown')
    for image_class in image_classes:
        print(f'{image_class.capitalize()}: {len(labels[labels == image_classes.index(image_class)])}')
    
    label_matrix = np.zeros((constants['num_rows_in_tiling'], constants['num_cols_in_tiling']), dtype=np.int_)

    for row in range(constants['num_rows_in_tiling']):
        for col in range(constants['num_cols_in_tiling']):
            label_matrix[row][col] = labels[np.argmax(identifiers == identifier_matrix[row][col])]

    np.save(f'{site_dir}/labels', labels)
    np.save(f'{site_dir}/label_matrix', label_matrix)

if __name__ == '__main__':
    assign_tile_labels(site=sys.argv[1])
