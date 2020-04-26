# ===================================================
#
#    BIA660 Web Mining Final Porject
#    By: Tyler Bryk & Kristina Cheng
#
# ===================================================

import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def loadData(fname):
    reviews = []
    labels = []
    stop_words = set(stopwords.words('english')) 
    f = open(fname)
    for line in f:
        review,rating = line.strip().split('\t')
        word_tokens = word_tokenize(review)   
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        filtered_sentence2 = []
        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(w)
        for w in filtered_sentence: 
            if w not in string.punctuation: 
                filtered_sentence2.append(w)
        stopremoved = ' '.join([str(elem) for elem in filtered_sentence2])
        reviews.append(stopremoved)    
        labels.append(int(rating))
    f.close()
    return reviews,labels

# Load in Training / Testing Data
trData, trLabel = loadData('TrainData.txt')
teData, teLabel = loadData('TestData.txt')

# Transform Data and Fit Neural Network
counter = CountVectorizer()
counter.fit(trData)
trCount = counter.transform(trData)
teCount = counter.transform(teData)
clf = MLPClassifier(hidden_layer_sizes=(1000, ), activation='logistic', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.005, power_t=0.5, max_iter=200, shuffle=False, random_state=8, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10 )

# Code used to Tune Model Hyperparameters
#print("[INFO] Tuning Hyperparameters via GridSearchCV")
#params = {"alpha": [0.00006, 0.0001, 0.0006], "learning_rate_init": [0.003, 0.004, 0.005, 0.006, 0.007], 'beta_1': [0.85, 0.9, 0.95], 'beta_2':[0.995, 0.997, 0.999] }
#grid = GridSearchCV(clf, params)
#grid.fit(counts_train, labels_train)
#acc = grid.score(counts_test, labels_test)
#print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
#print("[INFO] grid search best parameters: {}".format(grid.best_params_))

# Test Model Accuracy
clf.fit(trCount, trLabel)
pred = clf.predict(teCount)
print(accuracy_score(pred, teLabel))