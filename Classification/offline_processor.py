'''script to read the input data file and find out the k nearest neighbors for each of the instances in the training set. Subsequrntly, this script
also initialzes the training and testing datasets, from which the script knn.py can read the data and perform probability computations to facilitate multi
label predictions.'''

from sklearn.datasets import load_iris
from sklearn import cross_validation
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
# start the timer for the program
start = time.time()

# data structures to store the training and testing instances

final_data = {}
final_data['data'] = [] # list of all the data vectors accross the whole dataset
final_data['targets'] = [] # list of all the target labels accross the whole dataset
unique_label_map = {}  # this will later be converted to a list for iteration purposes

if(len(sys.argv)==1):
    file_path = '/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsedocfiles/sparse_doc_file_1.csv'
elif(len(sys.argv)>=3) :
    file_type = sys.argv[1]
    file_index = sys.argv[2]
    if(len(sys.argv)==4):
        test_frac = float(sys.argv[3])
    else : test_frac = 0.3

with open(file_path,'r') as doc_fp:
    doc_reader = csv.reader(doc_fp,delimiter=',')
    doc_reader.next()
    label_index = 0
    for line in doc_reader :
        bar_index = line.index('|')
        topics = line[bar_index+1:]
        final_data['data'].append(line[1:bar_index])
        final_data['targets'].append(topics)
        for t in topics :
            if t not in unique_label_map :
                unique_label_map[t] = label_index
                label_index += 1
# create the category vector data for every instance
    category_vector = []
    N = len(unique_label_map)

    for target_list in final_data['targets'] :
        target_set = set(target_list)
        row = [0] * N
        for l in unique_label_map :
            if l in target_set :
                row[unique_label_map[l]] = 1
        category_vector.append(row)


# split the data into training and testing sets. Here, the user is not given choice dut the high cost of computation. 70-30 split is used instead.

train_vectors, test_vectors, train_labels, test_labels = cross_validation.train_test_split(final_data['data'],
                                                                                               category_vector,
                                                                                               test_size=test_frac,
                                                                                              random_state=1)

training_set = (zip(train_vectors,train_labels))
testing_set = (zip(test_vectors, test_labels))

print "training set size : ",len(training_set)
print "testing set size : ",len(testing_set)

# write the training and the testing sets to appropriate files.

with open('/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/new_offline_data/training_data_file.csv','wb') as train_fp :
    train_writer = csv.writer(train_fp,delimiter=',')
    for t in training_set :
        train_writer.writerow(t[0]+['|']+t[1])
with open('/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/new_offline_data/testing_data_file.csv','wb') as test_fp :
    test_writer = csv.writer(test_fp,delimiter=',')
    for t in testing_set :
        test_writer.writerow(t[0]+['|']+t[1])

# calculate all the neighbours of the training set and put them in a file

# function to compute the euclidean distance between two data vector instances

def get_euclidean_distance(training_instance,testing_instance) :
    train_vector = training_instance[0]
    test_vector = testing_instance[0]
    points = zip(train_vector,test_vector)
    euclidean_dist = math.sqrt(sum([pow(int(a) - int(b), 2) for (a, b) in points]))
    return euclidean_dist

# function to return the k nearest neighbors of the given data vector which is passed as a parameter. The sorting of the vectors is done based on euclidean distance

def nearest_neighbours(test_instance,training_set,k) :

    tuple_distance_list = []
    for train_instance in training_set :
        dist = get_euclidean_distance(train_instance,test_instance)
        tuple_distance_list.append([train_instance, dist])
    sorted_tuple_distance_list = sorted(tuple_distance_list,key=operator.itemgetter(1))

    k_nearest_neighbours = []
    for i in range(0,k):
        k_nearest_neighbours.append(sorted_tuple_distance_list[i][0])
    return k_nearest_neighbours

# write the k-nearest neighbors of every training instance into the 'neighbors.csv' file

with open('/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/new_offline_data/neighbours.csv','wb') as nb_fp :
    nb_writer = csv.writer(nb_fp,delimiter=',')
    k = 5
    for i in range(0,len(training_set)):
        nbrs = nearest_neighbours(training_set[i],training_set,k)
        for n in nbrs :
            nb_writer.writerow(n[0]+['|']+n[1])
    print "neighbours of training instances initialized"

end = time.time()

print "time taken for complete offline processing : ",end-start
