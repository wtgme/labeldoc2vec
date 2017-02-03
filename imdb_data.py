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
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

window = 5
size = 400
iter = 5


def doc_vect(alldocs):
    print 'Doc2Vec with lineNO as ID'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    # unlable_docs = [doc for doc in alldocs if doc.split == 'extra']
    print('%d docs: %d train-label, %d test-label' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs:
        sentence = TaggedDocument(doc.words, doc.tags)
        documents.append(sentence)
    # for doc in unlable_docs:
    #     sentence = TaggedDocument(doc.words, doc.tags)
    #     documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                Doc2Vec(documents, dm=1, dm_concat=1, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, min_count=1, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, min_count=1, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, min_count=1, workers=cores),
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    for name, model in models_by_name.items():
        print name
        train_targets, train_regressors = zip(*[(doc.label, model.docvecs[doc.tags[0]]) for doc in train_docs])
        test_targets, test_regressors = zip(*[(doc.label, model.infer_vector(doc.words)) for doc in test_docs])
        data_util.logit(train_regressors, train_targets, test_regressors, test_targets)


def class_vect(alldocs):
    print 'Doc2Vec with ClassID as ID'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    # unlable_docs = [doc for doc in alldocs if doc.split == 'extra']
    print('%d docs: %d train-label, %d test-label' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs:
        sentence = TaggedDocument(doc.words, [doc.label])
        documents.append(sentence)
    # for doc in unlable_docs:
    #     sentence = TaggedDocument(doc.words, [])
    #     documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                Doc2Vec(documents, dm=1, dm_concat=1, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, workers=cores),
                # PV-DBOW Learing words
                # Doc2Vec(documents, dm=0, dbow_words=1, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, workers=cores),

                #  # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                # Doc2Vec(documents, dm=1, dm_concat=1, size=size, window=window, negative=5, hs=0, sample=1e-3, iter=iter, workers=cores),
                # # PV-DBOW
                # Doc2Vec(documents, dm=0, size=size, window=window, negative=5, hs=0, sample=1e-3, iter=iter, workers=cores),
                # # PV-DBOW Learing words
                # Doc2Vec(documents, dm=0, dbow_words=1, size=size, window=window, negative=5, hs=0, sample=1e-3, iter=iter, workers=cores),
                # # PV-DM w/average
                # Doc2Vec(documents, dm=1, dm_mean=1, size=size, window=window, negative=5, hs=0, sample=1e-3, iter=iter, workers=cores),

                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    # models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    # models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    for name, model in models_by_name.items():
        print name
        train_targets, train_regressors = zip(*[(doc.label, model.infer_vector(doc.words)) for doc in train_docs])
        test_targets, test_regressors = zip(*[(doc.label, model.infer_vector(doc.words)) for doc in test_docs])
        data_util.svm_class(train_regressors, train_targets, test_regressors, test_targets)
        data_util.model_similar(model, train_regressors, train_targets, test_regressors, test_targets)


def labeldoc_vect(alldocs):
    print 'LabelDoc2Vec with lineNO as ID'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    # unlable_docs = [doc for doc in alldocs if doc.split == 'extra']
    print('%d docs: %d train, %d test' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs:
        slable = []
        if doc.split == 'train':
            slable = ['s'+str(doc.label)]
        sentence = LabeledTaggedDocument(doc.words, doc.tags, slable)
        documents.append(sentence)
    # for doc in unlable_docs:
    #     slable = []
    #     if doc.split == 'train':
    #         slable = ['s'+str(doc.label)]
    #     sentence = LabeledTaggedDocument(doc.words, doc.tags, slable)
    #     documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                LabelDoc2Vec(documents, dm=1, dm_concat=1, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, min_count=1, workers=cores),
                # PV-DBOW
                LabelDoc2Vec(documents, dm=0, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, min_count=1, workers=cores),
                # PV-DM w/average
                LabelDoc2Vec(documents, dm=1, dm_mean=1, size=size, window=window, negative=5, hs=1, sample=1e-3, iter=iter, min_count=1, workers=cores),
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    for name, model in models_by_name.items():
        print name
        train_targets, train_regressors = zip(*[(doc.label, model.docvecs[doc.tags[0]]) for doc in train_docs])
        test_targets, test_regressors = zip(*[(doc.label, model.infer_vector_label(doc.words)) for doc in test_docs])
        data_util.logit(train_regressors, train_targets, test_regressors, test_targets)

if __name__ == '__main__':
    data = data_util.get_reuters_data()
    # doc_vect(data)
    class_vect(data)
    # labeldoc_vect(data)

