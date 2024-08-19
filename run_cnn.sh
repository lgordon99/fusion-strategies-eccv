#!/bin/bash

date
source ~/.bashrc
nvidia-smi
python cnn.py $@