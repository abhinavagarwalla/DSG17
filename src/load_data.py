import pandas as pd
from configs import *

train = pd.read_csv(INPUT_PATH + "train.csv", encoding="ISO-8859-1")
test = pd.read_csv(INPUT_PATH + "test.csv", encoding="ISO-8859-1")

data_all = pd.concat([train, test])
train_target = train.is_listened
