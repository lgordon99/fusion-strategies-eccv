# imports
import numpy as np
import os

settings = ['early_fusion', 'late_fusion', 'mixture_of_experts']
lr_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]
setting_lr = {setting: {lr: np.mean([np.load(f'hp_tuning/{setting}/lr={lr}/trial_{trial}.npy').tolist() for trial in range(1, 11)]) for lr in lr_values if len(os.listdir(f'hp_tuning/{setting}/lr={lr}')) == 10} for setting in settings}
print(setting_lr)
print('Tuned learning rates')

for setting in settings:
    print(f'{setting}: {max(setting_lr[setting], key=setting_lr[setting].get)}')
