
import os, random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def get_K_fold_with_test_generator(data, n_fold=5, seed=42, train_ratio=0.875):
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(data):
        train_size = int(len(train_index) * train_ratio)
        train_idx = np.random.choice(train_index, train_size, replace=False)
        val_idx = np.array([idx for idx in train_index if idx not in train_idx])

        yield train_idx, val_idx, test_index


def get_K_fold_generator(data, n_fold=10, seed=42):
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(data):

        yield train_index, test_index