# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:23:48 2020

@author: Kristina Cheng
"""

import requests

import pandas as pd
import numpy as np

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

urls = ['https://raw.githubusercontent.com/KristinaM703/BIA_660_group9/master/MasterTrainData.txt']

trn_txt = []
for url in urls:
    page = requests.get(url)
    text = page.text.split('\r')
    trn_txt.append(text)
trn_txt = [item for sublist in trn_txt for item in sublist]

trn_txt = [tuple(x.split('\t')) for x in trn_txt]

df = pd.DataFrame(trn_txt, columns =['review', 'rating'])

df['review'] = df['review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
df['review'] = df['review'].apply((lambda x: re.sub('\n','',x)))

stop = stopwords.words('english')

df['review'] = df['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

reviews = df['review']
ratings = df['rating']

x_train, x_test, y_train, y_test = train_test_split(reviews, ratings, test_size=0.33, random_state=42)

tuned_parameters = {
    'vect__ngram_range': [(1, 2), (2, 2),(2,3),(3,3)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}



clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring='accuracy')
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test), digits=4))