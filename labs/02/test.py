
import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

import sklearn.preprocessing

# 1 2 3
# 2 2 2


x = np.array([[0], [-1], [0], [3]])


print(x.shape)


one = sklearn.preprocessing.OneHotEncoder(sparse=False, categories='auto')

pipeline=  sklearn.pipeline.Pipeline([('one', one)])

y = pipeline['one'].fit_transform(x)


#enc = sklearn.preprocessing.OneHotEncoder(sparse=False)


print(y)

print(y.shape)