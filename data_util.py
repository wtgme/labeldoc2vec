# -*- coding: utf-8 -*-
"""
Created on 15:25, 06/10/16

@author: wt
"""

import multiprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import SVC
from collections import namedtuple

# def classifier_cv(X, y, K=5):
#     skf = StratifiedKFold(n_splits=K)
#     accuracys = []
#     for train_index, test_index in skf.split(X, y):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', n_jobs=multiprocessing.cpu_count())
#         # svc_lin = SVC(kernel='linear', class_weight='balanced')
#         y_lin = logistic.fit(X_train, y_train).predict(X_test)
#         score = accuracy_score(y_lin, y_test)
#         print "Fold Accuracy: %0.4f" % (score)
#         accuracys.append(score)
#     print("Overall Accuracy: %0.4f (+/- %0.4f)" % (np.mean(accuracys), np.std(accuracys)))


def logit(X_train, y_train, X_test, y_test):
    logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', n_jobs=multiprocessing.cpu_count())
    # svc_lin = SVC(kernel='linear', class_weight='balanced')
    y_lin = logistic.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_lin, y_test)
    print "Logit Accuracy: %0.4f" % (score)
    return score


def svm(X_train, y_train, X_test, y_test):
    # logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', n_jobs=multiprocessing.cpu_count())
    svc_lin = SVC(kernel='linear', class_weight='balanced')
    y_lin = svc_lin.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_lin, y_test)
    print "SVM Accuracy: %0.4f" % (score)
    return score


def pre_classify_text(X_train, y_train, X_test, y_test=None):
    corpus = np.append(X_train, X_test)
    from sklearn.feature_extraction.text import HashingVectorizer
    vectorizer = HashingVectorizer()
    X = vectorizer.fit_transform(corpus)
    """SVM classifier: too slow"""
    # svc_lin = SVC(kernel='linear', class_weight='balanced')
    # y_lin = svc_lin.fit(X[:len(X_train), :], y_train).predict(X[len(X_train):, :])
    """Parallel KNN: more fast"""
    # from sklearn.neighbors import KNeighborsClassifier
    # neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=multiprocessing.cpu_count())
    '''Batched Logit regression'''
    from sklearn import linear_model
    clf = linear_model.SGDClassifier(loss='log', n_jobs=multiprocessing.cpu_count(), )
    y_lin = clf.fit(X[:len(X_train), :], y_train).predict(X[len(X_train):, :])
    if y_test:
        print "Pre-classification accuracy: %0.4f" % accuracy_score(y_lin, y_test)
    return y_lin

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    return norm_text


def get_imdb_data():
    SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
    alldocs = []  # will hold all docs in original order
    with open('aclImdb/alldata-id.txt', 'r') as alldata:
        for line_no, line in enumerate(alldata):
            tokens = line.split()
            words = tokens[1:]
            tags = [line_no]  # `tags = [tokens[0]]` would also work at extra memory cost
            split = ['train', 'test', 'extra', 'extra'][line_no // 25000]  # 25k train, 25k test, 25k extra
            sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][
                line_no // 12500]  # [12.5K pos, 12.5K neg]*2 then unknown
            alldocs.append(SentimentDocument(words, tags, split, sentiment))
    return alldocs


def get_ng_data():
    from sklearn.datasets import fetch_20newsgroups
    # remove = ('headers', 'footers', 'quotes')
    data_train = fetch_20newsgroups(subset='train')
    data_test = fetch_20newsgroups(subset='test')
    y_train, y_test = data_train.target, data_test.target # Label ID from 0 to 19
    # names = (list(data_train.target_names))
    SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
    alldocs = []
    for line_no, line in enumerate(data_train.data):
            words = normalize_text(line).split()
            tags = [line_no]  # `tags = [tokens[0]]` would also work at extra memory cost
            split = 'train'
            sentiment = y_train[line_no]  # [12.5K pos, 12.5K neg]*2 then unknown
            alldocs.append(SentimentDocument(words, tags, split, sentiment))
    train_len = len(data_train.data)
    for line_no, line in enumerate(data_test.data):
            words = normalize_text(line).split()
            tags = [line_no+train_len]  # `tags = [tokens[0]]` would also work at extra memory cost
            split = 'test'
            sentiment = y_test[line_no]  # [12.5K pos, 12.5K neg]*2 then unknown
            alldocs.append(SentimentDocument(words, tags, split, sentiment))
    return alldocs

def get_rcv():
    from sklearn.datasets import fetch_rcv1
    rcv1 = fetch_rcv1()
    for doc in rcv1.data:
        print doc