import csv

unique_topic_list = []
with open('document_file.csv', 'rb') as doc_file, open('appearences.csv', 'wb') as topic_file:
    doc_file_reader = csv.reader(doc_file, delimiter=',')
    topic_file_writer = csv.writer(topic_file, delimiter='\n')

    for line in doc_file_reader:

        index = line[0].index('_')

        label_train = line[0][index - 1:]

        for topic in label_train.split():
            # print topic
            for words in topic.split():
                unique_topic_list.append(words)
    unique_topic_list = set(unique_topic_list)
    topic_file_writer.writerow(['antecedent'])
    for t in unique_topic_list:
        topic_string = t + " " + "consequent"
        topic_file_writer.writerow([topic_string])
