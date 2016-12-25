#from __future__ import division
import sys
import numpy as np
import csv
import time
import gc
import math
import operator

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from scipy import spatial

''' This function will split the data set into training and testing set depending on the value of split ratio which is given as
an input to this function. The function aslo takes an input "i" to determine the sample on which it has to work.
Basically the entire data set is divided into different samples to get more accuracy.
The output of the function is traning data, training class labels, testing data and testing class labels respectively.
'''
def splitData(filename,split_ratio, i):
    
    dataMatrix = [] # This is a list which will store all the valid rows of the feature vectors i.e all the rows which has a label 
    train_data = [] # This is a sub part of the dataMatrix list which will store all the valid feature values of the feature vectors of the training data 
    test_data = [] # This is a sub part of the dataMatrix list which will store all the valid feature values of the feature vectors of the testing data 
    train_labels = [] # This is a sub part of the dataMatrix list which will store all the valid labels of the feature vectors of the training data 
    test_labels = [] # This is a sub part of the dataMatrix list which will store all the valid labels of the feature vectors of the testing data 

    with open(filename, 'rb') as feature_file_22:
        
        feature_file_22_reader = csv.reader(feature_file_22, delimiter=',') # This will open the feature vector file
        feature_file_22_reader.next()

        for row in feature_file_22_reader:
            doc_id = int(row[0]) # retrieving the document id of each document
            if doc_id > 2720 * i and doc_id < 2720*(i+1):
                index = row.index('|')
                topic = row[index+1]
                if topic not in (None, ""): # Checking if the label is null or not
                    dataMatrix.append(row)    

    train, test = train_test_split(dataMatrix, test_size=split_ratio, random_state=42)
    del dataMatrix[:] # As the work of data matrix is over, hence clearing its memory

# Creating the testing data set and testing label set    
    for each in test:
        index = each.index('|')
        topic = each[index+1]
        
        test_labels.append(each[index+1])
        test_data.append(each[1:index-1])
    
# Creating the training data set and training label set
    for each in train:
        index = each.index('|')
        topic = each[index+1]
        
        train_labels.append(each[index+1])
        train_data.append(each[1:index-1])

    return np.array(train_data).astype(np.float),np.array(test_data).astype(np.float),np.array(train_labels),np.array(test_labels)

# Naive Bayes Classification
def NBClassifier(  filename, split_ratio, samples): 
    
    print "----------------Naive Bayes Classfier----------------"
    print "Data Split (training-testing):", (1-split_ratio)*100, "-", split_ratio*100
    Offline_Cost = 0.0 # Total cost for training of all samples
    Online_Cost = 0.0 # Total cost for testing of all samples
    accuracy = 0.0 # Total accuracy of all samples

    for i in range(0,samples):
        
        if ((samples) < 9):
          print "----------Sample ",i,"-----------"
          begin_time = time.time()
          TrainingSet, TestingSet, Training_labels, Testing_labels = splitData(filename,split_ratio, i)     
		  
          nb_model = GaussianNB()
          nb_model.fit(TrainingSet, Training_labels) # Traning model
          Of_cost = (time.time() - begin_time) # Offline cost for each sample
          Offline_Cost = Offline_Cost + Of_cost
          print " Time taken (Offline cost): " + str(Of_cost)
                 
          begin_time = time.time()
          predicted = nb_model.predict(TestingSet) # Prediction of class labels
          accuracy_val = nb_model.score(TestingSet,Testing_labels,sample_weight=None)*100 # Calculating accuracy
          accuracy = accuracy + accuracy_val
          print " Accuracy =", accuracy_val
          On_cost = (time.time() - begin_time) # Online cost for each sample
          Online_Cost = Online_Cost + On_cost
          print " Time taken (Online cost): " + str(On_cost)
          print "---------------------------------------------------------------"
          
          report = metrics.classification_report(Testing_labels, predicted)
          confusionMatrix = metrics.confusion_matrix(Testing_labels, predicted)
          reportname = "/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/NB_DT/NB_Report_" + str(split_ratio) +"_split_" + filename[63:69] + str(i+1) +"_file.txt"
          confusionMatrixname = "/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/NB_DT/NB_CM_" + str(split_ratio) +"_split_" + filename[63:69] + str(i+1) + "_file.csv"
          write_report(report, reportname)
          write_matrix(confusionMatrix, confusionMatrixname)
        else:
            print 'More number of samples inputed than expected'


    print " Total time taken (Offline Cost): ",Offline_Cost
    print " Total time taken (Online Cost): ", Online_Cost
    print " Total accuracy: ",(accuracy/samples)

def decisionTrees(filename, split_ratio, samples):
    print "----------------Decision Tree Classfier----------------"
    print "Data Split (training-testing):", (1-split_ratio)*100, "-", split_ratio*100
    Offline_Cost = 0.0 # Total cost for training of all samples
    Online_Cost = 0.0 # Total cost for testing of all samples
    accuracy = 0.0 # Total accuracy of all samples
    for i in range(0,samples):
        if ((samples) < 9):
            print "----------Sample ",i,"-----------"
            begin_time = time.time()
            TrainingSet, TestingSet, Training_labels, Testing_labels = splitData(filename,split_ratio, i)
            DT_model = DecisionTreeClassifier()	
            DT_model.fit(TrainingSet, Training_labels) # Traning model
            Of_cost = (time.time() - begin_time) # Offline cost for each sample
            Offline_Cost = Offline_Cost + Of_cost
            print " Time taken (Offline cost): " + str(Of_cost)
            
            begin_time = time.time()
            predicted = DT_model.predict(TestingSet) # Prediction of class labels

            accuracy_val = DT_model.score(TestingSet,Testing_labels,sample_weight=None)*100
            accuracy = accuracy + accuracy_val 
            print " Accuracy = ", accuracy_val
            On_cost = (time.time() - begin_time) # Online cost for each sample
            Online_Cost = Online_Cost + On_cost
            print " Time taken (Online cost): " + str(On_cost)
            print "---------------------------------------------------------------"
            
            report = metrics.classification_report(Testing_labels, predicted)
            confusionMatrix = metrics.confusion_matrix(Testing_labels, predicted)
            
            reportname = "/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/NB_DT/DT_Report_" + str(split_ratio) +"_split_" + filename[63:69] +"_file.txt"
            confusionMatrixname = "/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/NB_DT/DT_CM_" + str(split_ratio) +"_split_" + filename [63:69]+"_file.csv"
            
            write_report(report, reportname)
            write_matrix(confusionMatrix, confusionMatrixname)
        else:
            print 'More number of samples inputed than expected'


    print " Total time taken (Offline Cost): ",Offline_Cost
    print " Total time taken (Online Cost): ", Online_Cost
    print " Total accuracy: ",(accuracy/samples)


                
def write_report(report, file_name):
        text_file_22 = open(file_name, "w")
        text_file_22.write(report)
        
def write_matrix(matrix, file_name):
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar=',', quoting=csv.QUOTE_MINIMAL)   
        for rows in matrix:
            writer.writerow([data for data in rows])
 


    
if len(sys.argv) == 2:

    print "-"*20,"Classification on: Document Word Frequency Feature Vector","-"*20
    NBClassifier("/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsedocfiles/sparse_doc_file_22.csv",0.20, int(sys.argv[1]))
    NBClassifier("/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsedocfiles/sparse_doc_file_22.csv",0.25, int(sys.argv[1]))
    NBClassifier("/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsedocfiles/sparse_doc_file_22.csv",0.30, int(sys.argv[1]))

    decisionTrees("/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsedocfiles/sparse_doc_file_22.csv",0.20, int(sys.argv[1]))
    decisionTrees("/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsedocfiles/sparse_doc_file_22.csv",0.25, int(sys.argv[1]))
    decisionTrees("/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsedocfiles/sparse_doc_file_22.csv",0.30, int(sys.argv[1]))

    
    print "-"*20,"Classification on: Document TF-IDF Feature Vector","-"*20
    NBClassifier("/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsetfidffiles/sparse_tfidf_file_22.csv",0.20, int(sys.argv[1]))
    NBClassifier("/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsetfidffiles/sparse_tfidf_file_22.csv",0.25, int(sys.argv[1]))
    NBClassifier("/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsetfidffiles/sparse_tfidf_file_22.csv",0.30, int(sys.argv[1]))
    
    decisionTrees("/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsetfidffiles/sparse_tfidf_file_22.csv",0.20, int(sys.argv[1]))
    decisionTrees("/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsetfidffiles/sparse_tfidf_file_22.csv",0.25, int(sys.argv[1]))
    decisionTrees("/home/8/athmakuri.1/athmakuri.1_DM_Lab_2/sparsetfidffiles/sparse_tfidf_file_22.csv",0.30, int(sys.argv[1]))

else:
    
    print 'Please give the right format to run the files'


