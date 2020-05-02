# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:23:48 2020

@author: Kristina Cheng
"""

import requests

import pandas as pd
import csv

import re

from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

from sklearn.metrics import classification_report

urls = ['https://raw.githubusercontent.com/KristinaM703/BIA_660_group9/master/Clean%20Data/Cleaned-Amazon.csv', 
        'https://raw.githubusercontent.com/KristinaM703/BIA_660_group9/master/Clean%20Data/Cleaned-Books.csv',
        'https://raw.githubusercontent.com/KristinaM703/BIA_660_group9/master/Clean%20Data/Cleaned-Hotels.csv',
        'https://raw.githubusercontent.com/KristinaM703/BIA_660_group9/master/Clean%20Data/Cleaned-Movies60K.csv',
        'https://raw.githubusercontent.com/KristinaM703/BIA_660_group9/master/Clean%20Data/Cleaned-WomensClothing.csv',
        'https://raw.githubusercontent.com/KristinaM703/BIA_660_group9/master/Clean%20Data/Cleaned-Twitter.csv',
        'https://raw.githubusercontent.com/KristinaM703/BIA_660_group9/master/Clean%20Data/Cleaned-Restaurant.csv',
        'https://raw.githubusercontent.com/KristinaM703/BIA_660_group9/master/Clean%20Data/Cleaned-Reddit.csv']

df0 = pd.read_csv(urls[0], header = None)
df1 = pd.read_csv(urls[1], header = None)
df2 = pd.read_csv(urls[2], header = None)
df3 = pd.read_csv(urls[3], header = None)
df4 = pd.read_csv(urls[4], header = None)
df5 = pd.read_csv(urls[5], header = None)
df6 = pd.read_csv(urls[6], header = None)
df7 = pd.read_csv(urls[7], header = None)

df = pd.concat([df0,df1, df2, df3, df4, df5, df6, df7], axis=0)


df.columns = ['review', 'rating']

df['review'] = df['review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

stop = stopwords.words('english')

df['review'] = df['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

reviews = df['review']
ratings = df['rating']

x_train, x_test, y_train, y_test = train_test_split(reviews, ratings, test_size=0.33, random_state=42)

tuned_parameters = {
    'vect__ngram_range': [(1,2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}

clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring='accuracy')
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test), digits=4))