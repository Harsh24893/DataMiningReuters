import csv
from math import ceil
import random

def split_data(test_frac):

        with open('document_file.csv', 'rb') as f:
                data_file_reader = csv.reader(f, delimiter=',')
                data_test_train = []
                for row in data_file_reader :
                        #print row
                        data_test_train.append(row)

        random.shuffle(data_test_train)

        test_data = data_test_train[:int(ceil(len(data_test_train)*test_frac))]
        train_data = data_test_train[int(ceil(len(data_test_train)*test_frac)):]

        with open('training_data.csv', 'w') as train_file:
                train_file_writer = csv.writer(train_file, delimiter=' ')

                for row in train_data:

                        x = []

                        for words in row[0].split(' '):
                                x.append(words)
                        train_file_writer.writerow(x)

        with open('testing_data.csv', 'w') as test_file, open('test_data_class.csv','w') as test_label_file:
                test_label_file_writer = csv.writer(test_label_file, delimiter=',')
                test_file_writer = csv.writer(test_file, delimiter=' ')
                for row in test_data:
                        w = []
                        class_test = []
                        index = row[0].index('_')
                        #print index
                        test_words = row[0][:index-1]
                        label_test = row[0][index-1:]
                        for words in test_words.split(' '):
                                w.append(words)
                        for words in label_test.split(' '):
                                class_test.append(words)

                        test_file_writer.writerow(w)
                        test_label_file_writer.writerow(class_test)


