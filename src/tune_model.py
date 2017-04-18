from sklearn import cross_validation
import time
import xgboost as xgb

start_time = time.time()
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from configs import *
from utilities import change_to_int
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, make_scorer
from sklearn.cross_validation import StratifiedKFold

model_name = 'xgb'
dataset_name = 'as_it_is'

# Choosing parameters on subsets.
X_train = np.load('%s%s_train.npy' % (DATASET_PATH, dataset_name))
X_valid = np.load('%s%s_valid.npy' % (DATASET_PATH, dataset_name))
y_train = np.load('%sY_train.npy' % DATASET_PATH).ravel()
y_valid = np.load('%sY_valid.npy' % DATASET_PATH).ravel()

if model_name == 'rfr':
    # {'max_features': 13, 'min_samples_split': 6, 'n_estimators': 310}
    level0 = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=2016, verbose=1)
    param_grid = {'max_features': [25], 'max_depth': [25]}
    h_param_grid = {'n_estimators': hp.quniform('n_estimators', 100, 500, 1),
                    'max_features': hp.quniform('max_features', 5, 50, 1),
                    'min_samples_split':hp.quniform('min_samples_split', 2, 100, 1),
                    }
elif model_name == 'xgb':
    # {'colsample_bytree': 0.55, 'n_estimators': 1920.0, 'subsample': 0.8500000000000001, 'learning_rate': 0.325, 'max_depth': 6.0, 'gamma': 0.8500000000000001}
    level0 = xgb.XGBClassifier(learning_rate=0.325,
                               silent=True,
                               objective="binary:logistic",
                               nthread=-1,
                               gamma=0.85,
                               min_child_weight=5,
                               max_delta_step=1,
                               subsample=0.85,
                               colsample_bytree=0.55,
                               colsample_bylevel=1,
                               reg_alpha=0.5,
                               reg_lambda=1,
                               scale_pos_weight=1,
                               base_score=0.5,
                               seed=0,
                               missing=None,
                               n_estimators=1920, max_depth=6)
    h_param_grid = {'max_depth': hp.quniform('max_depth', 1, 13, 1),
                    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                    'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
                    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
                    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                    'n_estimators': hp.quniform('n_estimators', 100, 1000, 5),
                    }
elif model_name == 'tree':
    # {'max_depth': 29.0}
    level0 = DecisionTreeClassifier(random_state=2016)
    h_param_grid = {  # 'max_features': hp.quniform('max_features', 50, 100, 1),
                      'max_depth': hp.quniform('max_depth', 5, 50, 1)}

# Hyperopt Implementatation
def score(params):
    change_to_int(params, ['max_depth', 'n_estimators'])
    level0.set_params(**params)

    print(params)
    score = cross_validation.cross_val_score(level0, X_train, y_train, cv=2, scoring='neg_log_loss', n_jobs=-1)
    print('%s ------ Score Mean:%f, Std:%f' % (params, score.mean(), score.std()))
    return {'loss': score.mean(), 'status': STATUS_OK}

def optimize(trials):
    print('Tuning Parameters')
    best = fmin(score, h_param_grid, algo=tpe.suggest, trials=trials, max_evals=100)
    # best = {'max_features': 13, 'min_samples_split': 6, 'n_estimators': 310}
    # best = {'colsample_bytree': 0.55, 'n_estimators': 1920.0, 'subsample': 0.8500000000000001, 'learning_rate': 0.325, 'max_depth': 6.0, 'gamma': 0.8500000000000001}
    print('\n\nBest Scoring Value')
    print(best)

    change_to_int(best, ['max_depth', 'n_estimators'])
    level0.set_params(**best)
    level0.fit(X_train, y_train)

    print('Validation Score: %f' % scoring(level0, X_valid, y_valid))
    return best

trials = Trials()
optimize(trials)


#print('Validation Training')
#level0.fit(X_train, y_train)

#print(scoring(level0, X_train, y_train))
#print(scoring(level0, X_valid, y_valid))