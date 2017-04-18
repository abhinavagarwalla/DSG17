from utilities import SimpleTransform
from generate_dataset import generate_dataset
import numpy as np

def log_transform(x):
    return np.log(1 + x)


if __name__ == '__main__':
    features = [
        ('as_it_is', SimpleTransform()),
        # ('keras_feature', SimpleTransform()),

    ]

    generate_dataset(features, 'as_it_is')