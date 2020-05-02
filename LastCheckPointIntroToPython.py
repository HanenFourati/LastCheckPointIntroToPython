#!/usr/bin/env python
# coding: utf-8

# In[28]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
  
# load the iris datasets
dataset = datasets.load_iris()
# Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target)
print("X_train: "+str(X_train))
print("X_test: "+str(X_test))
print("y_train: "+str(y_train))
print("y_test: "+str(y_test))
"""
Result:
X_train: [[6.7 3.3 5.7 2.1]
 [5.8 2.7 5.1 1.9]
 [5.1 3.8 1.5 0.3]
 [7.2 3.  5.8 1.6]
 [7.2 3.2 6.  1.8]
 [5.7 2.6 3.5 1. ]
 [6.2 2.2 4.5 1.5]
 [5.3 3.7 1.5 0.2]
 [5.6 2.5 3.9 1.1]
 [4.7 3.2 1.3 0.2]
 [6.5 3.2 5.1 2. ]
 [4.9 3.6 1.4 0.1]
 [4.3 3.  1.1 0.1]
 [5.5 2.6 4.4 1.2]
 [5.  3.6 1.4 0.2]
 [6.3 2.7 4.9 1.8]
 [6.1 3.  4.6 1.4]
 [4.6 3.1 1.5 0.2]
 [5.6 3.  4.5 1.5]
 [5.4 3.4 1.5 0.4]
 [7.7 2.8 6.7 2. ]
 [6.8 3.  5.5 2.1]
 [6.  3.  4.8 1.8]
 [5.8 2.7 3.9 1.2]
 [4.7 3.2 1.6 0.2]
 [5.9 3.  5.1 1.8]
 [7.7 2.6 6.9 2.3]
 [6.7 3.1 4.4 1.4]
 [5.7 2.8 4.1 1.3]
 [4.4 3.2 1.3 0.2]
 [6.9 3.2 5.7 2.3]
 [6.3 3.3 4.7 1.6]
 [5.4 3.7 1.5 0.2]
 [5.  3.2 1.2 0.2]
 [4.6 3.4 1.4 0.3]
 [6.2 3.4 5.4 2.3]
 [5.5 2.4 3.7 1. ]
 [6.7 3.  5.2 2.3]
 [6.6 2.9 4.6 1.3]
 [6.9 3.1 4.9 1.5]
 [5.  3.3 1.4 0.2]
 [6.3 2.5 4.9 1.5]
 [6.4 3.2 5.3 2.3]
 [4.4 2.9 1.4 0.2]
 [5.2 4.1 1.5 0.1]
 [5.  3.  1.6 0.2]
 [6.5 3.  5.5 1.8]
 [7.1 3.  5.9 2.1]
 [7.4 2.8 6.1 1.9]
 [6.4 3.1 5.5 1.8]
 [4.8 3.4 1.9 0.2]
 [6.5 3.  5.2 2. ]
 [5.1 3.8 1.6 0.2]
 [5.8 2.7 5.1 1.9]
 [5.5 2.5 4.  1.3]
 [4.8 3.4 1.6 0.2]
 [5.9 3.  4.2 1.5]
 [5.  3.5 1.3 0.3]
 [5.6 2.7 4.2 1.3]
 [7.7 3.  6.1 2.3]
 [6.2 2.8 4.8 1.8]
 [6.  2.9 4.5 1.5]
 [6.3 3.4 5.6 2.4]
 [5.8 2.7 4.1 1. ]
 [6.3 2.5 5.  1.9]
 [5.2 3.5 1.5 0.2]
 [6.4 2.8 5.6 2.2]
 [6.1 3.  4.9 1.8]
 [6.1 2.8 4.  1.3]
 [6.8 3.2 5.9 2.3]
 [6.4 2.7 5.3 1.9]
 [6.5 2.8 4.6 1.5]
 [5.5 3.5 1.3 0.2]
 [5.1 3.5 1.4 0.3]
 [6.  2.2 5.  1.5]
 [7.6 3.  6.6 2.1]
 [5.1 3.5 1.4 0.2]
 [5.7 2.8 4.5 1.3]
 [6.5 3.  5.8 2.2]
 [6.3 3.3 6.  2.5]
 [6.4 2.9 4.3 1.3]
 [7.2 3.6 6.1 2.5]
 [5.6 2.8 4.9 2. ]
 [5.7 2.5 5.  2. ]
 [6.8 2.8 4.8 1.4]
 [6.1 2.6 5.6 1.4]
 [4.9 3.1 1.5 0.2]
 [6.2 2.9 4.3 1.3]
 [5.  2.  3.5 1. ]
 [4.6 3.6 1.  0.2]
 [5.2 2.7 3.9 1.4]
 [6.4 2.8 5.6 2.1]
 [4.9 3.1 1.5 0.1]
 [5.  3.4 1.6 0.4]
 [7.3 2.9 6.3 1.8]
 [4.9 2.5 4.5 1.7]
 [5.7 3.8 1.7 0.3]
 [5.1 3.7 1.5 0.4]
 [6.3 2.8 5.1 1.5]
 [5.5 2.4 3.8 1.1]
 [5.5 4.2 1.4 0.2]
 [5.6 2.9 3.6 1.3]
 [5.4 3.9 1.3 0.4]
 [5.7 3.  4.2 1.2]
 [5.8 2.8 5.1 2.4]
 [4.6 3.2 1.4 0.2]
 [5.4 3.  4.5 1.5]
 [5.  2.3 3.3 1. ]
 [6.7 3.  5.  1.7]
 [6.9 3.1 5.1 2.3]
 [6.  2.2 4.  1. ]
 [6.  2.7 5.1 1.6]]
X_test: [[5.2 3.4 1.4 0.2]
 [7.9 3.8 6.4 2. ]
 [6.1 2.8 4.7 1.2]
 [4.8 3.1 1.6 0.2]
 [6.4 3.2 4.5 1.5]
 [5.8 2.6 4.  1.2]
 [6.7 3.1 4.7 1.5]
 [4.9 3.  1.4 0.2]
 [6.7 3.1 5.6 2.4]
 [7.  3.2 4.7 1.4]
 [6.3 2.9 5.6 1.8]
 [4.8 3.  1.4 0.1]
 [6.6 3.  4.4 1.4]
 [4.9 2.4 3.3 1. ]
 [4.5 2.3 1.3 0.3]
 [5.4 3.9 1.7 0.4]
 [5.1 3.4 1.5 0.2]
 [4.8 3.  1.4 0.3]
 [6.7 3.3 5.7 2.5]
 [6.1 2.9 4.7 1.4]
 [6.7 2.5 5.8 1.8]
 [4.4 3.  1.3 0.2]
 [7.7 3.8 6.7 2.2]
 [5.8 4.  1.2 0.2]
 [6.9 3.1 5.4 2.1]
 [6.  3.4 4.5 1.6]
 [5.1 3.8 1.9 0.4]
 [5.6 3.  4.1 1.3]
 [5.7 2.9 4.2 1.3]
 [5.  3.4 1.5 0.2]
 [5.7 4.4 1.5 0.4]
 [5.1 3.3 1.7 0.5]
 [5.9 3.2 4.8 1.8]
 [6.3 2.3 4.4 1.3]
 [5.  3.5 1.6 0.6]
 [5.5 2.3 4.  1.3]
 [5.4 3.4 1.7 0.2]
 [5.1 2.5 3.  1.1]]
y_train: [2 2 0 2 2 1 1 0 1 0 2 0 0 1 0 2 1 0 1 0 2 2 2 1 0 2 2 1 1 0 2 1 0 0 0 2 1
 2 1 1 0 1 2 0 0 0 2 2 2 2 0 2 0 2 1 0 1 0 1 2 2 1 2 1 2 0 2 2 1 2 2 1 0 0
 2 2 0 1 2 2 1 2 2 2 1 2 0 1 1 0 1 2 0 0 2 2 0 0 2 1 0 1 0 1 2 0 1 1 1 2 1
 1]
y_test: [0 2 1 0 1 1 1 0 2 1 2 0 1 1 0 0 0 0 2 1 2 0 2 0 2 1 0 1 1 0 0 0 1 1 0 1 0
 1] 
"""
# fit a SVC model to the data
model = LinearSVC() # perform a multi-class classification on a dataset.
model.fit(X_train, y_train) # train the data set with train subsets
print(model.fit(X_train, y_train))

# predict new values
expected = y_test # the expected values that are stored in the test subset
predicted = model.predict(X_test) # the predicted values
print("expected values"+str(expected))
print("Predected values"+str(predicted))
# As seen in the result expected and predicted give the same values
# Result: expected values[0 2 1 0 1 1 1 0 2 1 2 0 1 1 0 0 0 0 2 1 2 0 2 0 2 1 0 1 1 0 0 0 1 1 0 1 0 1]
# Predected values[0 2 1 0 1 1 1 0 2 1 2 0 1 1 0 0 0 0 2 1 2 0 2 0 2 2 0 1 1 0 0 0 2 1 0 1 0 1]
# the confusion matrix shows a high recall, precision, and accuracy rates which implies that our classification algorithm performs well.
res = confusion_matrix(expected, predicted) 
print ('Confusion Matrix :')
print (res) 
print ('Accuracy Score :',accuracy_score(expected, predicted) )
print ('Report : ')
print (classification_report(expected, predicted) )


# In[20]:


# What does a confusion matrix tell you?
"""
A confusion matrix is a summary of prediction results on a classification problem. 
The number of correct and incorrect predictions are summarized with count values and broken down by each class.
The confusion matrix shows the ways in which your classification model is confused when it makes predictions.
It gives us insight not only into the errors being made by a classifier but more importantly the types of errors 
that are being made
""""
# Why do we need Confusion Matrix?
"""
A confusion matrix tells you how good a classification algorithm is. In particular it tells you about both the 
false negatives,true negatives, false positives and true positives.
This is useful because the results of classification algorithms cannot generally be expressed well in one number.
It can be used to evaluate the performance of a classification model through the calculation of performance metrics
like accuracy, precision, recall, and F1-score
"""
# How do you create a confusion matrix in python? 
"""
Using these two functions:
metrics.confusion_matrix(): takes in the list of actual labels, the list of predicted labels, and an optional 
argument to specify the order of the labels. It calculates the confusion matrix for the given inputs.
metrics.classification_report(): takes in the list of actual labels, the list of predicted labels, and an optional argument
to specify the order of the labels. It calculates performance metrics like precision, recall, and support.
"""


# In[24]:


# confusion matrix example 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
  
actual = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0] 
predicted = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0] 
results = confusion_matrix(actual, predicted) 
  
print ('Confusion Matrix :')
print (results) 
print ('Accuracy Score :',accuracy_score(actual, predicted) )
print ('Report : ')
print (classification_report(actual, predicted) )
"""
Confusion Matrix :
[[4 2]
 [1 3]]
Accuracy Score : 0.7
Report : 
              precision    recall  f1-score   support

           0       0.80      0.67      0.73         6
           1       0.60      0.75      0.67         4

    accuracy                           0.70        10
   macro avg       0.70      0.71      0.70        10
weighted avg       0.72      0.70      0.70        10
"""


# In[ ]:


# Set of classification algorithms: Logistic Regression, Naïve Bayes, Stochastic Gradient Descent, K-Nearest Neighbours, Decision Tree, Random Forest, and Support Vector Machine


# In[36]:


# K-Nearest Neighbours algorithm
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 
# The k-nearest neighbor algorithm is imported from the scikit-learn package.  
# Loading data 
irisData = load_iris() 
  
# create feature and target variables.
X = irisData.data 
y = irisData.target 
# Split into training and test set 
X_train, X_test, y_train, y_test = train_test_split( 
             X, y, test_size = 0.2, random_state=42) 

# Generate a k-NN model using neighbors value.  
knn = KNeighborsClassifier(n_neighbors=7) 
# Train or fit the data into the model.
knn.fit(X_train, y_train) 
print(knn.fit(X_train, y_train) )  
"""
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=7, p=2,
                     weights='uniform')
"""
# Predict on dataset which model has not seen before 
print(knn.predict(X_test)) 
# result [1 0 2 1 1 0 1 2 2 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]
# Calculate the accuracy of the model 
print(knn.score(X_test, y_test)) 
# result: 0.9666666666666667

