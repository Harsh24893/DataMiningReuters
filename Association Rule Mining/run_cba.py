import apriori_h
from data_splitter import split_data
from apriori_h import classfiy_based_association
import sys
import time


test_frac = 0.4
support = 5
confidence = 20

if(len(sys.argv)==4):

    test_frac = int(sys.argv[1])
    support = int(sys.argv[2])
    confidence = int(sys.argv[3])

split_start = time.time();
print "splitting data into ("+str((1-test_frac)*100)+"%)training and ("+str((test_frac)*100)+"%)testing sets - "
split_data(test_frac)
print "time taken to split data : ",time.time()-split_start," seconds"
print "Beginning Classification Based on Association (CBA)"
classfiy_based_association(support,confidence)


