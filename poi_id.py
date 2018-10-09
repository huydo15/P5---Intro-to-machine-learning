#!/usr/bin/python

import sys
import pickle
import numpy
import pandas as pd
sys.path.append("../")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','bonus','deferral_payments', 'deferred_income', 'director_fees', 'exercised_stock_options', 
#                'expenses', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances', 
#                'long_term_incentive','other', 'restricted_stock', 'restricted_stock_deferred', 'salary', 
#                'shared_receipt_with_poi','total_payments', 'total_stock_value', 'emailwithPOI', 
#                'short_term_interest', 'long_term_interest']

#final feature list
features_list = ['poi','bonus', 'exercised_stock_options', 'salary', 'total_stock_value', 'short_term_interest']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

#remove TOTAL from dataset
data_dict.pop('TOTAL',0)

#remove Lockhart from dataset since any data he has is NaN
data_dict.pop('LOCKHART EUGENE E',0)

#remove THE TRAVEL AGENCY IN THE PARK from dataset because this is not a person. According to enron61702insiderpay pdf document,
#this is a travel agency that books business travels for Enron employees
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

#remove NaN value
import math

NaNlist = ['bonus','deferred_income','from_this_person_to_poi', 'from_messages', 'from_poi_to_this_person', 'to_messages',
          'salary', 'other', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock']

for person in data_dict:
    for feature in NaNlist:
        if math.isnan(float(data_dict[person][feature])) == True:
            data_dict[person][feature] = 0

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#create a new feature: emailwithPOI to measure the % of email a person sends to or receives from a POI over the total emails 
#that person sends or receives. The idea is that a person with more communication with a POI is more likely to collude 
#in the scandal
for person, datapoints in my_dataset.iteritems():
    if datapoints['from_messages'] == 0 or datapoints['to_messages'] == 0:
        datapoints['emailwithPOI'] = 0
    else:
        datapoints['emailwithPOI'] = float(((datapoints['from_this_person_to_poi'] * 100 / datapoints['from_messages']) + 
                                    (datapoints['from_poi_to_this_person'] *100 / datapoints['to_messages'])))

# short_term_interest is the new metric that combine all payments that an Enron employee can receive within a year. These 
# payments tend to inventivize the employee to focus on short-term success of the company to maximize payout
    datapoints['short_term_interest'] = (datapoints['salary'] + datapoints['bonus'] + datapoints['other']
                        + datapoints['expenses'] + datapoints['exercised_stock_options'])
# long_term_interest forces the employee to think about long-term success of the company as these incentives
# cannot be received right away
    datapoints['long_term_interest'] = (datapoints['long_term_incentive'] + datapoints['restricted_stock'])

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Naive_Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]    
    
# Use SlectKBest ad f_classif to chose the best features
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)
selector.fit(features_train, labels_train)
new_features_train = selector.transform(features_train)
new_features_test = selector.transform(features_test)

# Print out the chosen features. Go back to feature_list and include only the chosen features
chosen_features = selector.get_support(indices=False)
testing_features = features_list[1:]
print pd.DataFrame(testing_features, chosen_features)

print selector.pvalues_
scores = selector.scores_
print scores
print pd.DataFrame(testing_features, scores)

import matplotlib.pyplot as plt
plt.bar(range(len(testing_features)), scores)
plt.xticks(range(len(testing_features)), testing_features, rotation='vertical')
plt.show()

from time import time
t0 = time()
clf.fit(new_features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(new_features_test)
print "predicting time:", round(time()-t1, 3), "s"

accuracy = clf.score(new_features_test, labels_test)
print "accuracy is:", accuracy

from sklearn import metrics
print "Precision score is:", metrics.precision_score(labels_test, pred)
print "Recall score is:", metrics.recall_score(labels_test, pred)

from sklearn.metrics import confusion_matrix
print "Confusion Matrix"
confusion_matrix(labels_test, pred, labels=[1,0])

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
