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
size = 400
iter = 5


def doc_vect(alldocs):
    print 'Doc2Vec with lineNO as ID'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    print('%d docs: %d train-label, %d test-label' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs:
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
        targets, regressors = zip(*[(doc.label, model.docvecs[doc.tags[0]]) for doc in train_docs])
        visualize.draw_words(regressors, targets, True, False, r'Doc2VecNG')
        # data_util.sim_ratio(regressors, targets)


def class_vect(alldocs):
    print 'Doc2Vec with ClassID as ID'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    print('%d docs: %d train-label, %d test-label' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs:
        sentence = TaggedDocument(doc.words, [doc.label])
        documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, min_count=1, workers=cores),
                # Doc2Vec(documents, dm=1, dm_concat=1, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, min_count=1, workers=cores)
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)

    for name, model in models_by_name.items():
        print name
        targets, regressors = zip(*[(doc.label, model.infer_vector(doc.words)) for doc in train_docs])
        # visualize.draw_words(regressors, targets, True, False, r'Class2VecNG')
        data_util.sim_ratio(regressors, targets)


def labeldoc_vect(alldocs):
    print 'LabelDoc2Vec with lineNO as ID'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    print('%d docs: %d train-label, %d test-label' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs+test_docs:
        slable = []
        if doc.split == 'train':
            slable = [doc.label]
        sentence = LabeledTaggedDocument(doc.words, doc.tags, slable)
        documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DBOW
                LabelDoc2Vec(documents, dm=0, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, min_count=1, workers=cores)
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)

    for name, model in models_by_name.items():
        print name
        targets, regressors = zip(*[(doc.label, model.docvecs[doc.tags[0]]) for doc in train_docs+test_docs])
        visualize.draw_words(regressors, targets, True, False, r'Label2VecNG')
        data_util.sim_ratio(regressors, targets)

if __name__ == '__main__':
    data = data_util.get_ng_data()
    # doc_vect(data)
    class_vect(data)
    # labeldoc_vect(data)

