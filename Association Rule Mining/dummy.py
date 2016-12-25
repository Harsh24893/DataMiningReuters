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

                if doc.topics and doc.topics.getText() :

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

                    doc_topics_line = topics.findChildren()
                    for t in doc_topics_line:
                        m = token_pattern.search(str(t))
                        if m:
                            doc_topics.append(m.group(1))
                            # doc_id_body['TOPICS'] = topics.getText()
                    if (topics is None or places is None or titles is None or body is None):
                        doc_id_topic_places_title['ID'] = file_id
                        if topics is not None:
                            #doc_id_topic_places_title['TOPICS'] = topics.getText()
                            doc_id_body['TOPICS'] = (doc_topics)
                            doc_id_topic_places_title['TOPICS'] = (doc_topics)
                        else:
                            doc_id_topic_places_title['TOPICS'] = (["NOT AVAILABLE"])
                        if places is not None:
                            doc_id_topic_places_title['PLACES'] = places.text
                        else:
                            doc_id_topic_places_title['PLACES'] = ["NOT AVAILABLE"]
                        if titles is not None:
                            doc_id_topic_places_title['TITLE'] = self.clean_stopword_stem(titles.getText())
                        else:
                            doc_id_topic_places_title['TITLE'] = ["NOT AVAILABLE"]
                        if body is not None:
                            doc_id_body['BODY'] = self.clean_stopword_stem(body.getText())
                        else:
                            doc_id_topic_places_title['BODY'] = ["NOT AVAILABLE"]

                        self.unprocessed_document_id_topic_places_title.append(doc_id_topic_places_title)
                        self.unprocessed_document_id_body.append(doc_id_body)

                    else:
                        doc_id_topic_places_title['ID'] = file_id
                        doc_id_topic_places_title['TOPICS'] = (doc_topics)
                        #doc_id_topic_places_title['TOPICS'] = topics.getText()
                        doc_id_topic_places_title['PLACES'] = places.text
                        doc_id_topic_places_title['TITLE'] = self.clean_stopword_stem(titles.getText())
                        doc_id_body['BODY'] = self.clean_stopword_stem(body.getText())
                        doc_id_body['TOPICS'] = (doc_topics)
                        #doc_id_body['TOPICS'] = topics.getText()
                        self.document_count_words+=1

                        self.document_id_topic_places_title.append(doc_id_topic_places_title)
                        self.document_id_body.append(doc_id_body)

class PreProcessor:

    global tokenizer
    tokenizer = RegexpTokenizer(r'[A-Za-z]+') #Used to extract only the english words from a string, as specified by the regex pattern
    global token_pattern
    token_pattern = re.compile('<d>([\w\-?\w]+)</d>')#Pattern used to extract the alpha-numeric words from the sub-tags of a larger tags. Specifically, used to extract 'topics' and 'places'.
    global stop_words
    stop_words = set(stopwords.words('english'))#We delete all the words which occuur very frequently,but are not actually useful for classificaiton.
    stop_words.add('reuter')
    stop_words.add('said')
    stop_words.add('would')
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


##This function will write each document's id and each body word and its count along with topic labels in a file. Note that the file id and the attributes are separated by a ',' but the attributes and the class labels are separated by a special character '|'. This is done to handle the cases of multiple class labels.

    def each_document_body_word_count(self):

        with open('document_file.csv', 'w') as doc_file:
            doc_file_writer = csv.writer(doc_file, delimiter=' ')
            #doc_file_writer.writerow(["DOCUMENT ID"] + ["DOCUMENT WORDS"] + ["TOPICS (CLASS LABELS)"])

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
                    #row.append(w + ":" + str(self.word_count[w]))
                    row.append(w)
                #doc_file_writer.writerow([doc['ID']] + row + ['|'] + list(doc['TOPICS']))
                l = []
                for words in (list(doc['TOPICS'])):
                    s='C:'
                    s = s + words
                    l.append(s)
                doc_file_writer.writerow(row + l)


    def preprocess(self):

        if len(sys.argv) == 1:


            begin_total_preproces_time = time.time()
            print "==================== Begin Preprocessing ===================="
            print "================== Parsing files =================="

            start= time.time()

            #============ Index to store total number of file ============"
            file_index = ['000','001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021']
            start = time.time()
            '''change here'''
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


            print "Total time taken for processing : ", time.time() - begin_total_preproces_time
        elif(len(sys.argv)==2):
            begin_total_preproces_time = time.time()
            print "==================== Begin Preprocessing ===================="
            print "================== Parsing files =================="

            start= time.time()

            #============ Index to store total number of file ============"
            file_index = ['000','001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021']
            start = time.time()
            for i in range(0,int(sys.argv[2])) :
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

            print "Total time taken for processing : ", time.time() - begin_total_preproces_time

	else : print "Please enter valid command arguments"



preProcessor = PreProcessor()
preProcessor.preprocess()
