from sklearn import svm
import csv as csv
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from time import time
from sklearn import preprocessing


df = pd.read_csv('./MNIST/train.csv')
img_count = df.shape[0]
train_count = np.ceil(img_count*0.7)
np.random.seed(1234)
random_idx = np.random.permutation(range(img_count))

train_idx = random_idx[:train_count]
test_idx = random_idx[train_count:]

df_value = df.values.astype(float)

# df_value[:,0] = df_value[:,0].astype(int)
df_value[:,1:] = preprocessing.scale(df_value[:,1:])
train_data = df_value[train_idx, :]
test_data = df_value[test_idx, :]



DOWN_SAMPLE_RATE = 10



def train_predict_raw():
    t0 = time()

    clf = svm.SVC()
    print "Fitting data..."
    clf = clf.fit(train_data[::DOWN_SAMPLE_RATE,1:], train_data[::DOWN_SAMPLE_RATE,0])
    print "done"
    print("done in %0.3fs" % (time() - t0))

    print "Predicting..."
    outcome = clf.predict(test_data[::DOWN_SAMPLE_RATE,1:]).astype(int)

    print "Prediction accuracy:",sum(outcome==test_data[::DOWN_SAMPLE_RATE,0])/float(len(outcome))
    return outcome

def train_predict_sift():
    #TODO: How to choose SIFT features from result? each image has different # of features
    pass


# output_file = open('MNIST_prediction_pixel_downsampled_trial_1.csv', 'wb')
# open_file_object = csv.writer(output_file)
# open_file_object.writerow(["Prediction_Idx", "Predicted_label"])
# open_file_object.writerows(zip(train_idx,outcome))
# output_file.close()








