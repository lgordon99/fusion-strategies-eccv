# imports
import json
import numpy as np
import torch
import utils

trials = 50
gating_weights_by_class = {class_label: {trial: {'gating_weights': [], 'mean': np.nan, 'std': np.nan} for trial in range(1, trials+1)} for class_label in range(4)}

for trial in range(1, trials+1):
    with open(f'moe_gating_weights/trial_{trial}.json', 'r') as gating_weights_json:
        data = json.load(gating_weights_json)

        for i in range(len(data['labels'])):
            gating_weights_by_class[data['labels'][i]][trial]['gating_weights'].append(data['gating_weights'][i])

for class_label, data in gating_weights_by_class.items():
    for trial in range(1, trials+1):
        gating_weights_by_class[class_label][trial]['mean'] = np.mean(data[trial]['gating_weights'], axis=0)
        gating_weights_by_class[class_label][trial]['std'] = np.std(data[trial]['gating_weights'], axis=0)

for class_label in range(4):
    print(class_label)
    pooled_mean = np.round(np.mean([gating_weights_by_class[class_label][trial]['mean'] for trial in range(1, trials+1)], axis=0), 2)
    print(pooled_mean)
    pooled_se = np.round(2 * np.sqrt(np.sum([gating_weights_by_class[class_label][trial]['std']**2 for trial in range(1, trials+1)], axis=0) / trials) / np.sqrt(trials), 2)
    print(pooled_se)
