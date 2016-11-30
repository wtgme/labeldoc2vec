# -*- coding: utf-8 -*-
"""
Created on 16:15, 30/10/16

@author: wt
"""
import data_util
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def classification(data):
    train_docs = [doc for doc in data if doc.split == 'train']
    test_docs = [doc for doc in data if doc.split == 'test']
    train, y_train = [], []
    for doc in train_docs:
        train.append(' '.join(doc.words))
        y_train.append(doc.label)
    test, y_test = [], []
    for doc in test_docs:
        test.append(' '.join(doc.words))
        y_test.append(doc.label)

    vectorizer = TfidfVectorizer(min_df=0)
    print len(train)
    X_train = vectorizer.fit_transform(train)
    X_test = vectorizer.transform(test)
    data_util.svm_class(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    data = data_util.get_ng_data()
    classification(data)



