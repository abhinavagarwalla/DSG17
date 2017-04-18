from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from configs import *
import pickle as pkl
import numpy as np

def simple_split():
	y_train = np.load(DATASET_PATH + "Y_train.npy").ravel()
	folds = list(StratifiedKFold(y_train, n_folds=configs['n_folds'], shuffle=True))
	pkl.dump(folds, open(DATASET_PATH + 'folds.pkl', 'wb'))

if __name__=="__main__":
	simple_split()