#!/usr/bin/env python
# coding: utf-8

# In[24]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# load the breast cancer dataset
dataset = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target)

# fit a SVC model to the data
model = LinearSVC()
model.fit(X_train, y_train)

# make predictions
expected = y_test
predicted = model.predict(X_test)


# In[ ]:




