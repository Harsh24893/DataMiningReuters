import sys, string, os
import csv
from subprocess import call
import time
from operator import itemgetter, attrgetter, methodcaller
import operator
import time


# def accuracy(predicted,test_labels):
# 	acc = 0
# 	#print len(predicted),len(test_labels)
#
# 	for i,val in enumerate(predicted):
# 		if set(predicted).intersection(set(list(test_labels[i]))) == set(list(val.split())):
# 			acc = acc + 1
# 	print (float(acc)/float(len(predicted)))*100

def accuracy(predicted,test_labels):
    acc = 0
    #print "check : ",len(predicted),len(test_labels)
    i = 0
    for pred in predicted:
        if(set(pred).issubset(set(test_labels[i]))):
            acc = acc + 1
        i += 1
    print "Computed Accuracy : ",(float(acc)/float(len(predicted)))*100,"%"

def classfiy_based_association(S,C):

    apriori_start = time.time()
    print "Beginning Association Rule Mining using Apriori Algorithm"
    supportArg = "-s"+str(S)
    confArg = "-c"+str(C)
    call(["apriori", "-tr",supportArg,confArg,"-R","appearences.csv","training_data.csv","assoc_rules.csv"])
    print "Time Taken to Mine the Rules : ",time.time()-apriori_start," seconds"
    class_start = time.time()
    print "Beginning Classification of Test Set"
    ## store the rules with their id and support and confidence values
    rules = [];
    ruleIndex = 0;
    ruleMap = {}
    association_rule_ant = {}
    association_rule_class = {}
    association_rule_confidence = {}
    association_rule_support = {}
    with open("assoc_rules.csv","rb") as rule_file :
        ruleFileReader = csv.reader(rule_file,delimiter='\n')
        for line in ruleFileReader :
            rule = line[0]
            ruleMap[ruleIndex] = rule;
            angleBracket = rule.index('<')
            openBracket = rule.index('(')
            closedBracket = rule.index(')')
            ruleclass = rule[:angleBracket-1]
            ruleClassList = ruleclass.split(",");
            #print ruleclass
            ant = rule[angleBracket+3:openBracket-1]
            #print ant
            ruleConsequent = rule[:angleBracket]
            ruleAntecedent = rule[angleBracket+2:openBracket]
            supportConfidencePair = rule[openBracket+1:closedBracket]
            support = float(supportConfidencePair.split(",")[0])
            confidence = float(supportConfidencePair.split(",")[1])
            association_rule_class[ruleIndex] = ruleClassList;
            association_rule_ant[ruleIndex] = ant
            # association_rule_confidence[rule] = confidence
            # association_rule_support[rule] = support
            rules.append((ruleIndex,support,confidence));
            ruleIndex+=1
    #sorted_association_rule_confidence = sorted(association_rule_confidence.items(), key=operator.itemgetter(1),reverse=True)
    rules.sort(key=lambda tup:(tup[2],tup[1],tup[0]),reverse=True)
    

    with open('testing_data.csv', 'rb') as test_data_file, open('test_data_class.csv', 'rb') as test_class_file:
            test_data_file_reader = csv.reader(test_data_file, delimiter=',')
            test_class_file_reader = csv.reader(test_class_file, delimiter=',')

            testset = []
            for test_set in test_data_file_reader:
                testset.append(test_set)

            test_labels = []
            for test_label in test_class_file_reader:
                test_labels.append(test_label)
        
    predicted = []
    #
    # for datapoint in testset:
    #
    #     for rule in sorted_association_rule_confidence:
    #             if set(datapoint).intersection(set(list(association_rule_ant[rule[0]].split()))) == set(list(association_rule_ant[rule[0]].split())):
    #                     prediction = association_rule_class[rule[0]]
    #                     break
    #             else:
    #                     prediction = "Default"
    #     predicted.append(prediction)

    for datapoint in testset :
        dataPointWords = set(datapoint[0].split(" "))
        for r in rules :
            antecedent = association_rule_ant[r[0]];
            antset = set(antecedent.split(" "));
            if(antset.issubset(dataPointWords)):
                prediction = association_rule_class[r[0]]
                break;
            else :
                #print "entered else at ",r[0],antset,dataPointWords
                prediction = ["C:earn"]

        predicted.append(prediction)

    # print "predicted set ka size : ",len(predicted)
    # print "testlabels ka size : ", len(test_labels)
    print "Time Take to Classify Test Instances : ",time.time()-class_start," seconds"
    accuracy(predicted,test_labels)


#classfiy_based_association(5,10)


