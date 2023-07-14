#!/usr/bin/env python
# coding: utf-8

# What is a TfidfVectorizer?
# TF (Term Frequency): The number of times a word appears in a document is its Term Frequency. A higher value means a term appears more often than others, and so, the document is a good match when the term is part of the search terms.
# 
# IDF (Inverse Document Frequency): Words that occur many times a document, but also occur many times in many others, may be irrelevant. IDF is a measure of how significant a term is in the entire corpus.
# 
# The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features.

# What is a PassiveAggressiveClassifier?
# Passive Aggressive algorithms are online learning algorithms. Such an algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector

# In[1]:


pip install numpy pandas sklearn


# In[3]:


import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[5]:


#Read the data
df = pd.read_csv("news.csv")


# In[6]:


#Get shape and head
df.shape


# In[7]:


#Get head
df.head()


# In[8]:


#get the labels
labels = df.label
labels.head()


# In[14]:


#split dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(df['text'],labels, test_size= 0.2, random_state = 7)


# Let’s initialize a TfidfVectorizer with stop words from the English language and a maximum document frequency of 0.7 (terms with a higher document frequency will be discarded). Stop words are the most common words in a language that are to be filtered out before processing the natural language data. And a TfidfVectorizer turns a collection of raw documents into a matrix of TF-IDF features.
# 
# Now, fit and transform the vectorizer on the train set, and transform the vectorizer on the test set.

# In[15]:


#initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)


# In[16]:


#fit and transfrm train set , transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


# In[17]:


#initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


# Next, we’ll initialize a PassiveAggressiveClassifier. This is. We’ll fit this on tfidf_train and y_train.
# 
# Then, we’ll predict on the test set from the TfidfVectorizer and calculate the accuracy with accuracy_score() from sklearn.metrics.

# In[21]:


#predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')


#  We got an accuracy of 92.98% with this model. Finally, let’s print out a confusion matrix to gain insight into the number of false and true negatives and positives.

# In[23]:


# Build confusion matrix
confusion_matrix(y_test,y_pred, labels = ['FAKE', "REAL"])


# So with this model, we have 590 true positives, 588 true negatives, 41 false positives, and 48
# false negatives.
# 
# Summary
# Today, we learned to detect fake news with Python. We took a political dataset, implemented a TfidfVectorizer, initialized a PassiveAggressiveClassifier, and fit our model. We ended up obtaining an accuracy of 92.82% in magnitude.

# In[ ]:




