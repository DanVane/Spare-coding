from sklearn import svm
import csv as csv
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from time import time


t0 = time()
param_grid = {'C': [5, 10, 15, 20, 25],
              'gamma': [0.005, 0.001, 0.05, 0.01],
              'kernel': ['rbf']
              }
clf = GridSearchCV(svm.SVC(class_weight='auto'), param_grid, cv = 3)
# clf = svm.SVC(kernel='sigmoid', class_weight='auto', gamma = 0.004, C = 0.5)
print "Fitting data..."
clf = clf.fit(train_data2[::,1::], train_data2[::,0])
print "done"
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
