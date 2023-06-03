#!/usr/bin/env python
# coding: utf-8

# In[13]:


#Import Libraries

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[5]:


#Load Data

df =df = pd.read_csv("C:/Users/ELCOT/Desktop/DATA SCIENCE/DataSets/news.csv")
df.head()


# In[7]:


df.shape


# In[8]:


#Get Labels

labels = df.label
labels.head()


# In[9]:


#Split data into Train and Test

x_train,x_test,y_train,y_test = train_test_split(df['text'], labels, test_size=0.2,random_state=7)


# In[17]:


#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[19]:


#PassiveAggressiveClassifier

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)


# In[23]:


#Calculate Accuracy 

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy :{round(score*100,2)}%')


# In[24]:


#Confusion Matrix

confusion_matrix(y_test,y_pred, labels=["FAKE","REAL"])


# In[ ]:


#We have 590 true positives(TP), 588 true negatives(TN), 41 false positives(FP), and 48 false negatives(FN).

