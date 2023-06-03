# Detecting-Fake-News-with-Python-and-Machine-Learning

To build a model to accurately classify a piece of news as REAL or FAKE

### Overview

This advanced python project of detecting fake news deals with fake and real news.Using sklearn, we build a TfidfVectorizer on our dataset. Then, we initialize a PassiveAggressive Classifier and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares.

### Steps 

* Make necessary imports
    * import numpy as np
    * import pandas as pd
    * import itertools
    * from sklearn.model_selection import train_test_split
    * from sklearn.feature_extraction.text import TfidfVectorizer
    * from sklearn.linear_model import PassiveAggressiveClassifier
    * from sklearn.metrics import accuracy_score, confusion_matrix
* Read the data into a DataFrame
* Get the labels from the DataFrame
* Split the dataset into training and testing sets
* Initialize a TfidfVectorizer 
* Initialize a PassiveAggressiveClassifier
* Find Accuracy and Confusion matrix

We took a dataset, implemented a TfidfVectorizer, initialized a PassiveAggressiveClassifier, and fit our model. We ended up obtaining an accuracy of 92.98% in magnitude.
