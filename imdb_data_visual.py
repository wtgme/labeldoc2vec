# -*- coding: utf-8 -*-
"""
Created on 13:47, 19/10/16

@author: wt
"""

from collections import OrderedDict
from gensim.models.doc2vec import Doc2Vec
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from gensim.models.labeldoc2vec import LabelDoc2Vec, LabeledTaggedDocument
import data_util
import logging
import visualize
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

window = 5
size = 100
iter = 20

def doc_vect(alldocs):
    print 'Doc2Vec with lineNO as ID'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    unlable_docs = [doc for doc in alldocs if doc.split == 'extra']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs:
        sentence = TaggedDocument(doc.words, doc.tags)
        documents.append(sentence)
    for doc in unlable_docs:
        sentence = TaggedDocument(doc.words, doc.tags)
        documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, min_count=1, workers=cores)
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)

    for name, model in models_by_name.items():
        print name
        pws = model.most_similar(positive=['good'], topn=5)
        nws = model.most_similar(positive=['bad'], topn=5)
        words = ['good', 'bad']
        for ws in pws:
            words.append(ws[0])
        for ws in nws:
            words.append(ws[0])
        vectors = [model[word] for word in words]
        visualize.draw_words(vectors, words, True, False, r'Doc2Vec')





def class_vect(alldocs):
    print 'Doc2Vec with ClassID as ID'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    unlable_docs = [doc for doc in alldocs if doc.split == 'extra']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs:
        sentence = TaggedDocument(doc.words, ['l'+str(doc.sentiment)])
        documents.append(sentence)
    for doc in unlable_docs:
        sentence = TaggedDocument(doc.words, [])
        documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, min_count=1, workers=cores)
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)

    for name, model in models_by_name.items():
        print name
        pws = model.most_similar(positive=['good'], topn=5)
        nws = model.most_similar(positive=['bad'], topn=5)
        words = ['good', 'bad']
        for ws in pws:
            words.append(ws[0])
        for ws in nws:
            words.append(ws[0])
        vectors = [model[word] for word in words]
        visualize.draw_words(vectors, words, True, False, r'Class2Vec')


def labeldoc_vect(alldocs):
    print 'LabelDoc2Vec with lineNO as ID'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    unlable_docs = [doc for doc in alldocs if doc.split == 'extra']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs:
        slable = []
        if doc.split == 'train':
            slable = ['s'+str(doc.sentiment)]
        sentence = LabeledTaggedDocument(doc.words, doc.tags, slable)
        documents.append(sentence)
    for doc in unlable_docs:
        slable = []
        if doc.split == 'train':
            slable = ['s'+str(doc.sentiment)]
        sentence = LabeledTaggedDocument(doc.words, doc.tags, slable)
        documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, min_count=1, workers=cores)
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)

    for name, model in models_by_name.items():
        print name
        pws = model.most_similar(positive=['good'], topn=5)
        nws = model.most_similar(positive=['bad'], topn=5)
        words = ['good', 'bad']
        for ws in pws:
            words.append(ws[0])
        for ws in nws:
            words.append(ws[0])
        vectors = [model[word] for word in words]
        visualize.draw_words(vectors, words, True, False, r'Label2Vec')

if __name__ == '__main__':
    data = data_util.get_imdb_data()
    doc_vect(data)
    class_vect(data)
    labeldoc_vect(data)

