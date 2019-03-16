#!/home/ahernandez/anaconda2/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB

### create classifier
classifier = GaussianNB()

### train the classifier on the training features and the training labels
t0 = time()
classifier.fit(features_train, labels_train)
print "training time: ", round(time()-t0, 3), "s"

### use the trained classifier to predict labels for the test features 
t0 = time()
pred = classifier.predict(features_test)
print "predict time: ", round(time()-t0, 3), "s"

### compute the accuracy of our classifier
### accuracy = no. of points clasified correclty / all points (in test set)
### (hint: sklearn.metrics and import from there accuracy_score)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(pred, labels_test)

print "Hola!, the accuracy of your classifier, using the given training data is: ", accuracy

#########################################################


