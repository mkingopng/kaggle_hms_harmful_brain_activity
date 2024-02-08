"""
configuration module for the project
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gc
from glob import glob
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import GroupKFold
import time
import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List


# configuration
class CFG:
    """configuration class holding mainly hyperparameters"""
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VALID = 32
    DROP_RATE = 0.1
    DROP_PATH_RATE = 0.2
    EPOCHS = 4
    FINAL_DIV_FACTOR = 100
    FOLDS = 5
    FREEZE = False
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1e-7
    MODEL = "tf_efficientnet_b0"
    NUM_FROZEN_LAYERS = 39
    NUM_WORKERS = multiprocessing.cpu_count()
    OPTIMIZER_LR = 0.1
    PCT_START = 0.05
    PRINT_FREQ = 20
    SCHEDULER_MAX_LR = 1e-3
    SEED = 20
    VERSION = 1
    WEIGHT_DECAY = 0.01
    TARGET_PREDS = [x + "_pred" for x in [
        'seizure_vote',
        'lpd_vote',
        'gpd_vote',
        'lrda_vote',
        'grda_vote',
        'other_vote'
    ]]

    LABEL_TO_NUM = {
        'Seizure': 0,
        'LPD': 1,
        'GPD': 2,
        'LRDA': 3,
        'GRDA': 4,
        'Other': 5
    }


class BOOLS:
    """boolean constants for the project"""
    AMP = True
    READ_EEG_SPEC_FILES = False
    READ_SPEC_FILES = False  #
    TRAIN_FULL_DATA = False  #
    VISUALIZE = False


class PATHS:
    """file paths used in the project"""
    OUTPUT_DIR = "./torch_efficientnet_b0_checkpoints/"
    PRE_LOADED_EEGS = './brain_eeg_spectrograms/eeg_specs.npy'
    PRE_LOADED_SPECTOGRAMS = './brain_spectrograms/specs.npy'
    TRAIN_CSV = "./data/train.csv"
    TRAIN_EEGS = "./brain_eeg_spectrograms/EEG_Spectrograms/"
    TRAIN_SPECTOGRAMS = "./data/train_spectrograms/"


# set device to gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')