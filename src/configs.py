INPUT_PATH = "../data/"

FEATURES_PATH = "features/"

DATASET_PATH = 'datasets/'

RESULTS_PATH = 'results/'

configs = {
    'n_folds': 5,
    'seed': 2017,
    'silent': False
}

LEN_TRAIN = 6761179
LEN_VALID = 1522358

TRAIN_FILE = INPUT_PATH + 'train.csv'
VALID_FILE = INPUT_PATH + 'valid.csv'
TEST_FILE = INPUT_PATH + 'test.csv'

FOLD_PATH = DATASET_PATH + str(configs['n_folds']) + 'folds/'

float_formatter = lambda x: "%.3f" % x

# sklearn
skl_n_estimators_min = 100
skl_n_estimators_max = 1000
skl_n_estimators_step = 10