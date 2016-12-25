from sklearn.datasets import load_iris
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import coverage_error
import numpy as np
import csv
import collections
import math
import operator
from operator import itemgetter
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import random
import time
import sys

# data structures to store the training and testing instances

final_data = {}
final_data['data'] = [] # list of all the data vectors accross the whole dataset
final_data['targets'] = [] # list of all the target labels accross the whole dataset

# read input from user regarding the kind of output that is required, and the desired k value. If the user does not enter anything, use
# default values.

if(len(sys.argv)==1):
    file_path = '/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsedocfiles/sparse_doc_file_22.csv'
    k = 5
if(len(sys.argv)>=5):
    file_type = sys.argv[1]
    file_index = sys.argv[2]
    file_path = '/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparse'+file_type+'files/sparse_'+file_type+'_file_'+str(file_index)+'.csv'
    k = int(sys.argv[3])
if(len(sys.argv)==7 and sys.argv[5]=='-s'): # -s indicates that the user intends to give a custom split ratio
    test_fraction = float(sys.argv[6])
else : test_fraction = 0.3

print "                         **********Reading data from the file**********"

with open(file_path,'r') as doc_fp:
    doc_reader = csv.reader(doc_fp,delimiter=',')
    doc_reader.next()
    for line in doc_reader :
        bar_index = line.index('|')
        topics = line[bar_index+1:]
        final_data['data'].append([float(l) for l in line[1:bar_index]])
        final_data['targets'].append(topics)

# binarize the multi-label problem to convert it into a single label problem, and the fit the model on the dataset

binarizer = MultiLabelBinarizer()
binary_labels = binarizer.fit_transform(final_data['targets'])

# split the dataset into training and testing sets according to the ration specified by the user.

print "splitting data into training and testing sets"

train_vectors, test_vectors, train_labels, test_labels = cross_validation.train_test_split(final_data['data'],
                                                                                               binary_labels,
                                                                                               test_size=test_fraction,
                                                                                         random_state=1)

print "size of training set : ",len(train_vectors)
print "size of testing set : ",len(test_vectors)

# initialize the K nearest Neighbors classifier

classifier = KNeighborsClassifier(n_neighbors=k)

print "                         **********Begin Classification**********"

classification_start_time = time.time()
classifier.fit(train_vectors,train_labels)
print "time taken to fit the model to the training data : ",time.time()-classification_start_time," secs"

print "                         **********Computing performance measures**********"

prediction_start = time.time()
acc = True
coverror = False

# analyse the user input to determine the kind of output that the user desires

if(len(sys.argv)>=5):
    if(sys.argv[4]=='-c'):
        acc = False
        coverror = True
    if(sys.argv[4]=='-ac'):
        coverror = True
if(acc):
    prediction_time_begin = time.time()
    # in-built method to predict accuracy of the predictions. The score method has an in-built predictor method in it, so predict() method need not be run again.
    accuracy = classifier.score(test_vectors,test_labels)
    print "obtained accuracy : ",accuracy*100,"%"
    print "time taken to classify (online cost) : ",time.time()-prediction_time_begin
if(coverror):
    # predict the test vector labels and compute the coverage error for the predictions
    predictions = classifier.predict(test_vectors)
    cov_err = coverage_error(test_labels,predictions)
    print "obtained coverage error : ",cov_err
    print "time taken to compute performance metrics : ", time.time() - prediction_start," secs"
    print "writing predictions to predictions_file.csv file "
    with open('/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/predictions_file.csv','wb') as pred_fp :
        pred_writer = csv.writer(pred_fp,delimiter=',')
        pred_writer.writerows(predictions)

print "processing successfully completed"