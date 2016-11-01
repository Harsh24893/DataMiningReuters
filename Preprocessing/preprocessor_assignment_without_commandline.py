'''
This program will take files from reuters database and preprocess it which includes stop word removal, unwanted numeric and non numeric character removal and finally stemming.
After preprocessing the program will create multiple files among which there is a matrix which has a document word frequence and a TF - IDF docuemnt matrix which acts as our
feature vector. 
'''
import re
import urllib2
import nltk
import sys
import csv
import time
import operator
import math
import collections
import sys

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer


'''
This is a class which does the cleaning and parsing of documents
'''
class Parser:

    document_id_topic_places_title = [] #This is a list which will store the id, topic, place and title value of all the documents which are considered VALID in the reuters database, in a dictionary format
    document_id_body = [] #This is a list which will store the id and body text value of all the documents which are considered VALID in the reuters database, in a dictionary format
    unprocessed_document_id_topic_places_title = [] #This is a list which will store the id, topic, place and title value of all the documents which are NOT considered VALID in the reuters database, in a dictionary format
    unprocessed_document_id_body = [] #This is a list which will store the id and body text value of all the documents which are NOT considered VALID in the reuters database, in a dictionary format
    document_count_words = 0 #This variable will store the number of valid documents which were preprocessed
    
    document_count_titles = 0 #The number of documents that are used to generate the titles
    global_column_index = 0


#This function will take a document as a parameter and will remove all unwanted characters in the text and then will remove stop words along with stemming the document text.

    def clean_stopword_stem(self, doc):
        doc_words = tokenizer.tokenize(doc)
        word_list = [stemmer.stem(word.lower()) for word in doc_words if word.lower() not in stop_words and len(word) >=3]
        text = ' '.join(word_list)

        return str(text.encode('utf-8'))
        

#This function will take each document in the 21K document database and will parse it to extract useful information like its topic, place, newId, title and body text from it.

    def parse(self, unprocessed_files):

            for doc in unprocessed_files.findAll('reuters'):

                file_id = doc['newid']#This variable stores the id of each document which is parsed
                word_count = dict()
                word_set = set()
                doc_words = []
                doc_topics = []
                doc_places = []
                doc_id_topic_places_title = {}
                doc_id_body = {}

                doc_id_topic_places_title['ID'] = file_id
                doc_id_body['ID'] = file_id


                topics = doc.find("topics")
                places = doc.find("places")
                titles = doc.find("title")
                body = doc.find("body")
                if (topics is None or places is None or titles is None or body is None):
                    doc_id_topic_places_title['ID'] = file_id
                    if topics is not None:
                        doc_id_topic_places_title['TOPICS'] = topics.getText()
                        doc_id_body['TOPICS'] = topics.getText()
                    else:
                        doc_id_topic_places_title['TOPICS'] = ["NOT AVAIABLE"]
                    if places is not None:
                        doc_id_topic_places_title['PLACES'] = places.text
                    else:
                        doc_id_topic_places_title['PLACES'] = ["NOT AVAIABLE"]
                    if titles is not None:
                        doc_id_topic_places_title['TITLE'] = self.clean_stopword_stem(titles.getText())
                    else:
                        doc_id_topic_places_title['TITLE'] = ["NOT AVAIABLE"]
                    if body is not None:
                        doc_id_body['BODY'] = self.clean_stopword_stem(body.getText())
                    else:
                        doc_id_topic_places_title['BODY'] = ["NOT AVAIABLE"]
                       
                    self.unprocessed_document_id_topic_places_title.append(doc_id_topic_places_title)
                    self.unprocessed_document_id_body.append(doc_id_body)
                    
                else:
                    doc_id_topic_places_title['ID'] = file_id
                    doc_id_topic_places_title['TOPICS'] = topics.getText()
                    doc_id_topic_places_title['PLACES'] = places.text
                    doc_id_topic_places_title['TITLE'] = self.clean_stopword_stem(titles.getText())
                    doc_id_body['BODY'] = self.clean_stopword_stem(body.getText())
                    doc_id_body['TOPICS'] = topics.getText()
                    self.document_count_words+=1
                   
                    self.document_id_topic_places_title.append(doc_id_topic_places_title)
                    self.document_id_body.append(doc_id_body)

class PreProcessor:
    
    global tokenizer
    tokenizer = RegexpTokenizer(r'[A-Za-z]+') #Used to extract only the english words from a string, as specified by the regex pattern
    global token_pattern
    token_pattern = re.compile('<d>([\w]+)</d>')#Pattern used to extract the alpha-numeric words from the sub-tags of a larger tags. Specifically, used to extract 'topics' and 'places'.
    global stop_words
    stop_words = set(stopwords.words('english'))#We delete all the words which occuur very frequently,but are not actually useful for classificaiton.
    global stemmer
    stemmer = SnowballStemmer('english') # used to stem and treat as one, the words which share a common root

	#Initialize all the variable 
    def __init__(self):
        
        self.title_word_count = dict() # a map to store the number of times, a particular word occurs in the title of the document
        self.title_column_map = collections.OrderedDict()# maps every word (in the document title) to the column in which the title word should go in the final feature matrix
        self.global_column_title_index = 0 # used to incrementally assign a new column (in the feature vector matrix) to every new title-word that is encountered
        self.parser = Parser()
        self.global_word_set = set()
        

        self.word_column_map = collections.OrderedDict() # maps every word (in the document body) to the column to which the word should go, in the final feature vector matrix
        self.global_column_word_index = 0 # used to incrementally assign a new column (in the feature vector matrix) to every new body-word that is encountered
        self.word_cumulative_freq = dict()  # a map which tells the total number of occurrences of a word across all the articles across all the files
        self.word_document_count = dict()  # a map which tells us the number of documents in which a particular word occurs
        self.unique_word_list = []
        
    
#This function will write each document's id and each title word and its count along with topic labels in a file. Note that the file id and the attributes are separated by a ',' but the attributes and the class labels are separated by a special character '|'. This is done to handle the cases of multiple class labels.          
    def each_document_title_word_count(self):

        with open('title_file.csv', 'w') as title_file:
            title_file_writer = csv.writer(title_file, delimiter=',')
            title_file_writer.writerow(["DOCUMENT ID"] + ["TITLE WORDS"] + ["TOPICS (CLASS LABELS)"])

            
            for doc in self.parser.document_id_topic_places_title:

                self.title_word_count =dict()
                for w in str(doc['TITLE']).split() :    
                    if w not in self.title_word_count :
                        self.title_word_count[w] = 1
                    else : self.title_word_count[w] += 1
                    if w not in self.title_column_map :
                        self.title_column_map[w] = self.global_column_title_index
                        self.global_column_title_index += 1
                title_file_row = []
                for t in self.title_word_count :
                    title_file_row.append(t + ":" + str(self.title_word_count[t]))
                title_file_writer.writerow([doc['ID']] + title_file_row + ['|'] + [doc['TOPICS']])

    
##This function will write each document's id and each body word and its count along with topic labels in a file. Note that the file id and the attributes are separated by a ',' but the attributes and the class labels are separated by a special character '|'. This is done to handle the cases of multiple class labels.

    def each_document_body_word_count(self):

        with open('document_file.csv', 'w') as doc_file:
            doc_file_writer = csv.writer(doc_file, delimiter=',')
            doc_file_writer.writerow(["DOCUMENT ID"] + ["DOCUMENT WORDS"] + ["TOPICS (CLASS LABELS)"])

            for doc in self.parser.document_id_body:

                self.word_count = dict()
                for w in str(doc['BODY']).split():
                    if w not in self.word_count:
                        self.word_count[w] = 1
                    else:
                        self.word_count[w] += 1
                    if w not in self.word_column_map:
                        self.word_column_map[w] = self.global_column_word_index
                        self.global_column_word_index += 1

                row = []
                for w in self.word_count:
                    if w not in self.word_cumulative_freq:
                        self.word_cumulative_freq[w] = 1
                    else:
                        self.word_cumulative_freq[w] += self.word_count[w]
                    if w not in self.word_document_count:
                        self.word_document_count[w] = 1
                    else:
                        self.word_document_count[w] += 1
                    row.append(w + ":" + str(self.word_count[w]))
                doc_file_writer.writerow([doc['ID']] + row + ['|'] + [doc['TOPICS']])
 
#We will now create the feature vector matrice for the title-words. To do this, the program iterates over the 'title_file.csv', created earlier, which have the informaiton about the local frequency of each title,and create the feature vector matrix from those frequencies. Note that the file id and the attributes are separated by a ',' but the attributes and theclass labels are separated by a special character '|'. This is done to handle the cases of multiple class labels.
    def sparse_title(self):
        
        N = len(self.title_column_map)

        
        
        with open('title_file.csv', 'rb') as title_file, open('sparse_title_file.csv', 'w') as sparse_title_fp:
            sparse_title_writer = csv.writer(sparse_title_fp, delimiter=',')
            title_file_reader = csv.reader(title_file, delimiter=',')
		
            title_file_reader.next()

            row_count = 1

            unique_title_list = []
            for t in self.title_column_map :
                unique_title_list.append(t)
            sparse_title_writer.writerow(['DOCUMENT ID']+unique_title_list)

            for row in title_file_reader:

                doc_id = row[0]
                #print row_count
                index = row.index('|')

                new_sparse_row = ['0'] * N
                for i in range(1, index):
                    row_tokens = (row[i]).split(":")
                    wrd = row_tokens[0]
                    cnt = row_tokens[1]
                    column = self.title_column_map[wrd]
                    new_sparse_row[column] = str(int(new_sparse_row[column]) + int(cnt))
                sparse_title_writer.writerow([doc_id] + new_sparse_row)
                row_count += 1

				
 
 #We will now create the feature vector matrice for the body-words. To do this, the program iterates over the 'document_file.csv', created earlier, which have the informaiton about the local frequency of each word,
 #and create the feature vector matrix from those frequencies. Note that the file id and the attributes are separated by a ',' but the attributes and the
 #class labels are separated by a special character '|'. This is done to handle the cases of multiple class labels.

    def sparse_bag_of_words(self):
        
        N = len(self.word_column_map)

        with open('document_file.csv', 'rb') as doc_file, open('sparse_doc_file.csv', 'wb') as sparse_doc_fp:
            doc_file_reader = csv.reader(doc_file, delimiter=',')
            sparse_doc_writer = csv.writer(sparse_doc_fp, delimiter=',')
			
            doc_file_reader.next()
            row_count = 1

            
            for w in self.word_column_map:
                self.unique_word_list.append(w)
            sparse_doc_writer.writerow(['DOCUMENT ID'] + self.unique_word_list)

            for row in doc_file_reader:

                doc_id = row[0]
                #print row_count
                index = row.index('|')

                new_sparse_row = ['0'] * N
                for i in range(1, index):
                    row_tokens = (row[i]).split(":")
                    wrd = row_tokens[0]
                    cnt = row_tokens[1]
                    column = self.word_column_map[wrd]
                    new_sparse_row[column] = str(int(new_sparse_row[column]) + int(cnt))
                sparse_doc_writer.writerow([doc_id] + new_sparse_row)
                row_count += 1



# We will now create the feature vector matrice for the tf-idf matrix for the bag of words. To do this,
#the program iterates over the 'document_file.csv created earlier, which have the informaiton about the local frequency of each word,
# and create the feature vector matrix from those frequencies. Note that the file id and the attributes are separated by a ',' but the attributes and the
# class labels are separated by a special character '|'. This is done to handle the cases of multiple class labels.			
				
    def tf_idf_matrix(self):
        
        N = len(self.word_column_map)

        word_idf = dict()  # a map which tells us the idf of every word across all the entire corpus

        # calculate the idf measure for every word

        for word in self.word_document_count:
            word_idf[self.word_column_map[word]] = math.log(self.parser.document_count_words / self.word_document_count[word])

        # once you get the word idfs generate the feature vector matrix for the tf/idf measure

        row_count = 1

        with open('document_file.csv','r') as word_file, open('sparse_tfidf_file.csv','wb') as sparse_tfidf_file :
            doc_file_reader = csv.reader(word_file,delimiter=',')
            sparse_tfidf_writer = csv.writer(sparse_tfidf_file,delimiter=',')
			
            doc_file_reader.next()

            sparse_tfidf_writer.writerow(['DOCUMENT ID'] + self.unique_word_list)

            for row in doc_file_reader :
                doc_id = row[0]
                index = row.index('|')
                #print row_count
                new_row = ['0']*N

                for i in range(1, index):
                    row_tokens = (row[i]).split(":")
                    wrd = row_tokens[0]
                    cnt = row_tokens[1]
                    column = self.word_column_map[wrd]
                    new_row[column] = str(float(cnt) * word_idf[self.word_column_map[wrd]])
                sparse_tfidf_writer.writerow([doc_id]+new_row)
                row_count += 1


    def preprocess(self):

        if len(sys.argv) == 1:
            

            begin_total_preproces_time = time.time()
            print "==================== Begin Preprocessing ===================="
            print "================== Parsing files =================="

            start= time.time()

            #============ Index to store total number of file ============"
            file_index = ['000','001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021']
            start = time.time()
            for i in range(0,22) :
                print " Parsing file: ",i
                f = urllib2.urlopen("http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-"+file_index[i]+".sgm")
                unprocessed_files = BeautifulSoup(f,"html.parser")
                self.parser.parse(unprocessed_files)

            print "Time taken to parse and clean up all documents : ", float("{0:.2f}".format(time.time() - start)), "s"

            print "No. of documents correctly parsed : ", self.parser.document_id_topic_places_title.__len__()
            print "No. of documents skipped : ", self.parser.unprocessed_document_id_body.__len__()

            print "================== Writing each document body word count files =================="
            start = time.time()
            self.each_document_body_word_count()
            print "Time taken to create document body word count file : ", float("{0:.2f}".format(time.time() - start)), "s"

            print "================== Writing each document title word count files =================="
            start = time.time()
            self.each_document_title_word_count()
            print "Time taken to create document title word count file : ", float("{0:.2f}".format(time.time() - start)), "s"

                   
            print "================== Writing spare title file =================="
            start = time.time()
            self.sparse_title()
            print "Time taken to create document sparse title file : ", float("{0:.2f}".format(time.time() - start)), "s"

   
            print "================== Writing sparse bag of words file =================="
            start = time.time()
            self.sparse_bag_of_words()
            print "Time taken to create sparse bag of words file : ", float("{0:.2f}".format(time.time() - start)), "s"

    
            print "================== Writing TF-IDF matrix =================="
            start = time.time()
            self.tf_idf_matrix()
            print "Time taken to create TF IDF file : ", float("{0:.2f}".format(time.time() - start)), "s"
                        
            print "Total time taken for processing : ", time.time() - begin_total_preproces_time
        else:
            begin_total_preproces_time = time.time()
            print "==================== Begin Preprocessing ===================="
            print "================== Parsing files =================="

            start= time.time()

            #============ Index to store total number of file ============"
            file_index = ['000','001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021']
            start = time.time()
            for i in range(0,22) :
                print " Parsing file: ",i
                f = urllib2.urlopen("http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-"+file_index[i]+".sgm")
                unprocessed_files = BeautifulSoup(f,"html.parser")
                self.parser.parse(unprocessed_files)

            print "Time taken to parse and clean up all documents : ", float("{0:.2f}".format(time.time() - start)), "s"

            print "No. of documents correctly parsed : ", self.parser.document_id_topic_places_title.__len__()
            print "No. of documents skipped : ", self.parser.unprocessed_document_id_body.__len__()

            print "================== Writing each document body word count files =================="
            start = time.time()
            self.each_document_body_word_count()
            print "Time taken to create document body word count file : ", float("{0:.2f}".format(time.time() - start)), "s"

            print "================== Writing each document title word count files =================="
            start = time.time()
            self.each_document_title_word_count()
            print "Time taken to create document title word count file : ", float("{0:.2f}".format(time.time() - start)), "s"

            print "================== Writing spare title file =================="
            start = time.time()
            self.sparse_title()
            print "Time taken to create document sparse title file : ", float("{0:.2f}".format(time.time() - start)), "s"

    
            print "================== Writing sparse bag of words file =================="
            start = time.time()
            self.sparse_bag_of_words()
            print "Time taken to create sparse bag of words file : ", float("{0:.2f}".format(time.time() - start)), "s"

    
            print "================== Writing TF-IDF matrix =================="
            start = time.time()
            self.tf_idf_matrix()
            print "Time taken to create TF IDF file : ", float("{0:.2f}".format(time.time() - start)), "s"
                        
            print "Total time taken for processing : ", time.time() - begin_total_preproces_time
            


preProcessor = PreProcessor()
preProcessor.preprocess()
