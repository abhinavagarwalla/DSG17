from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from utilities import make_folder
from configs import *
import pickle as pkl
import pandas as pd

paths = [FEATURES_PATH, DATASET_PATH, RESULTS_PATH]

for i in range(configs['n_folds']):
    paths.append(FOLD_PATH + 'fold' + str(i + 1))

for path in paths:
    make_folder(path)