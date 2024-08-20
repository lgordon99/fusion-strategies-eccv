# imports
import numpy as np
import os

settings = ['early_fusion', 'late_fusion', 'mixture_of_experts']
lr_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]
setting_lr = {setting: {lr: np.load(f'hp_tuning/{setting}/lr={lr}.npy').tolist() for lr in lr_values if os.path.exists(f'hp_tuning/{setting}/lr={lr}.npy')} for setting in settings}

print('Tuned learning rates')

for setting in settings:
    print(f'{setting}: {max(setting_lr[setting], key=setting_lr[setting].get)}')
