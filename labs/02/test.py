
import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

import sklearn.preprocessing

# 1 2 3
# 2 2 2


x = np.array([[0], [-1], [0], [30]])


xx = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

print(x.shape)


one = sklearn.preprocessing.OneHotEncoder(sparse=False, categories='auto')

scaler = sklearn.preprocessing.StandardScaler(1)

pol = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)


y = one.fit_transform(x)

y = scaler.fit_transform(xx)

y = pol.fit_transform(xx)


d = {}

d[4] = 1
d[2] = 2

print(len(d))
exit()

#enc = sklearn.preprocessing.OneHotEncoder(sparse=False)


print(y)

print(y.shape)