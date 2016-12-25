import random
import string
from math import sqrt
from time import time
from os import listdir
from bs4 import BeautifulSoup
from nltk import PorterStemmer
# from __future__ import division
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import jaccard_similarity_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import re
import sys
import csv
import math
import nltk
import urllib2
import operator
import collections
import numpy as np

global ngram # stores the number of n grams, by default the value is 3
ngram = 3; 
global hashFunctionCount # stores the nummber of hash functions to run the documents on
hashFunctionCount = 16;
global parsed_documents
parsed_documents = []  # document in the reuter dataset which we are parsing to create document term matrix
global document_id
document_id = [];
global documentCount
documentCount = 0;
global shingleCount
shingleCount = 0;
global global_signature_matrix
global_signature_matrix = []
global X
X = []


#------------------------------- function to generate hash function -----------------------------------
def generate_hash_functions(k):
    randList = random.sample(xrange(0, 10000), k);
    return randList
#------------------------------------------------------------------------------------------------------

#------------------------- function to generate a random prime number ---------------------------------
def compute_largest_prime(no_of_shingles):
    if(no_of_shingles==62201):
        return 62201
    elif (no_of_shingles == 117097):
        return 117101
    elif (no_of_shingles==167125):
        return 167149
    elif (no_of_shingles==222200):
        return 222247
    elif (no_of_shingles == 545013):
        return 545023
    elif (no_of_shingles == 1078020):
        return 1078027
    elif(no_of_shingles==65915):
        return 65921
    elif(no_of_shingles==310645) :
        return 310663
    elif (no_of_shingles==50000):
        return 50021
    return 1374589
#------------------------------------------------------------------------------------------------------

#------------------------- function to compute Jaccard Similarity -------------------------------------
def jaccard_similarity(U,V):
    x = U.nonzero()[1]
    y = V.nonzero()[1]
    numerator = len(set(x).intersection(set(y)))
    denominator = len(x)+len(y)-numerator
    if (denominator == 0):
        jac_sim = 0;
    else:
        jac_sim = numerator / float(denominator);
    return jac_sim;
#------------------------------------------------------------------------------------------------------

#------------- function to compute Estimated Jaccard Similarity (MinHash Approach) --------------------
def estimateSimilarity(U,V):
    common = 0
    for index in range(hashFunctionCount):
        if U[index] == V[index]:
            common += 1
    hash_value = common/float(hashFunctionCount);
    return hash_value
#------------------------------------------------------------------------------------------------------


#------------------------- function to Parse the documents --------------------------------------------
def parseDoc(doc):
    global tokenizer
    tokenizer = RegexpTokenizer(
        r'[A-Za-z]+')  # Used to extract only the english words from a string, as specified by the regex pattern
    global token_pattern
    token_pattern = re.compile(
        '<d>([\w]+)</d>')  # Pattern used to extract the alpha-numeric words from the sub-tags of a larger tags. Specifically, used to extract 'topics' and 'places'.
    global stop_words
    stop_words = set(stopwords.words(
        'english'))  # We delete all the words which occuur very frequently,but are not actually useful for classificaiton.
    global stemmer
    stemmer = SnowballStemmer('english')  # used to stem and treat as one, the words which share a common root
    doc_words = tokenizer.tokenize(doc)
    word_list = [stemmer.stem(word.lower()) for word in doc_words if word.lower() not in stop_words and len(word) >= 3]
    text = ' '.join(word_list)
	
    return str(text.encode('utf-8'))
#------------------------------------------------------------------------------------------------------


def main():
    global ngram
    global hashFunctionCount
    global parsed_documents
    global document_id
    global documentCount
    global shingleCount
    global X

    start = time();
    file_count = 0;
    grams = 0;
    if(len(sys.argv)==1):
        ngram = 3;
        file_count = 1;
        hashFunctionCount = 16;
    elif(len(sys.argv)==4):
        ngram = int(sys.argv[1])
        file_count = int(sys.argv[2])
        hashFunctionCount = int(sys.argv[3])
    else :
        print "please enter valid command line arguments"
        sys.exit()

# ==================================================Part 1 (Parsing)=================================================

    dir_path = "C:\Python27\DMSubmission\SampleDataFile.txt"

    # parse and clean up data
    print "Parsing Documents ..."
    start_time = time()

    file_index = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                  '014', '015', '016', '017', '018', '019', '020', '021']

    documentCollection = []
    for i in range(0,file_count):
        print "Parsing File ", i
        f = urllib2.urlopen("http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-" + file_index[i] + ".sgm")
        #f = open(dir_path,'rb');
        raw_data = BeautifulSoup(f, "html.parser")
        for document in raw_data.findAll("reuters"):
            # print 'Hellooo'
            if document.find("body") and document.find('body').getText():
                # print 'Hellooo'
                document_id.append(int(document['newid']));
                documentCollection.append(parseDoc(document.find("body").getText()))


    #parsed_documents = random.sample(documentCollection,2000);
    parsed_documents = documentCollection
    print("%s : %.2fs" % ("Total Time taken for parsing documents ", time() - start_time))
    print "No. of documents : ", parsed_documents.__len__()

    # create feature vector using tf and binary document term frequency
    print "Creating document term matrix ..."
    start_time = time()
    vectorizer = CountVectorizer(ngram_range=(ngram, ngram),binary=True,dtype=np.bool,min_df=1,max_df=0.9)
    X = vectorizer.fit_transform(parsed_documents)

    print("%s : %.2fs" % ("Total Time taken for creating document term matrix ", time() - start_time))
	
# ===================================================================================================================


# ==========================================Part 2 (Creating Signature Matrix)=======================================
    documentCount = X.get_shape()[0];
    shingleCount = X.get_shape()[1];
    print "shingle count : ",shingleCount

    coeffA = generate_hash_functions(hashFunctionCount);
    coeffB = generate_hash_functions(hashFunctionCount);
    C = compute_largest_prime(shingleCount);

    print "creating signature matrix : "
    sigStart = time()
    for id in range(0,documentCount):
        signature = []
        indices = X[id].nonzero()[1];
        for i in range(0,hashFunctionCount):
            min_hash_value = documentCount+1;
            for x in indices :
                hash_value = ((long(coeffA[i])*long(x) + long(coeffB[i]))%C)%documentCount;
                min_hash_value = min(hash_value,min_hash_value);
            signature.append(min_hash_value)
        global_signature_matrix.append(signature)

    print "time taken to create signature matrix : ",time()-sigStart
# ====================================================================================================================

# =====================================Part 3 (Baseline Jaccard Similarity)===========================================
	
    print "Calculating Jaccard Similarity Matrix : "
    jaccard_start = time()
    true_sim = [x[:] for x in [[0]*documentCount]]*documentCount
    for i in range(0,documentCount):
        print i
        for j in range(i+1,documentCount):
            true_sim[i][j]  = jaccard_similarity(X[i],X[j])
    print "time to calculate jaccard : ",time()-jaccard_start
# ====================================================================================================================


# ==============================Part 4 & 5 (MinHash Jacard Similarity and Efficacy Metric)============================
    print "calculating min hash values and RMSE : "
    RMSE = 0
    minstart = time()
    for i in range(0,documentCount):
        print i
        for j in range(i+1,documentCount):
            #est_sim = estimateSimilarity(global_signature_matrix[i],global_signature_matrix[j],true_sim[i][j])
            est_sim = estimateSimilarity(global_signature_matrix[i],global_signature_matrix[j])
            RMSE = RMSE + (true_sim[i][j] - est_sim)**2


    print "time taken to create min hash values : ",time()-minstart
    pairs = (documentCount*(documentCount))/2;
    RMSE = sqrt(RMSE/pairs)
    print "Root Mean Squared Error Obtained : ",RMSE
    print 'Time taken :', time() - start
# ====================================================================================================================

main()

