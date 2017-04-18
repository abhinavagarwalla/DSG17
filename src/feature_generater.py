from load_data import *
from generate_dataset import save_feature
import numpy as np

print('- Data and Modules Loaded')

def as_it_is():
    save_feature(data_all.values, 'as_it_is')
    train_target.values.dump('%s%s.npy' % (DATASET_PATH, 'Y_train'))

if __name__ == '__main__':
    # data_all = np.log10(1 + data_all)
    # train = np.log10(1 + train)

    # data_all = (data_all - train.mean()) / (data_all.max() - data_all.min())
    # data_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    # data_all.dropna(1, inplace=True)

    as_it_is()


# Interaction Features
# if __name__ == "__main__":
#     print('- Data Loaded')
#     top = list(data_all.columns.values)

#     for feat1, feat2 in itertools.combinations(top, 2):
#         col1 = data_all[feat1].values
#         col2 = data_all[feat2].values

#         feat = col1 * col2
#         print('%s_x_%s Generated' % (feat1, feat2))
#         save_feature(feat, '%s_x_%s' % (feat1, feat2))

#         feat = col1 / col2
#         print('%s_by_%s Generated' % (feat1, feat2))
#         save_feature(feat, '%s_by_%s' % (feat1, feat2))

#     print('- Features Generated')
