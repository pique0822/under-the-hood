import pickle
import sklearn.linear_model as sk

import numpy as np

location = '../trained-classifiers/train/hx_l0.pickle'
model = pickle.load(open(location,'rb'))
weights, bias = model
weights = weights[0]

regr
