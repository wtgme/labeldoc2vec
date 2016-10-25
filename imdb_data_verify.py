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

window = 10
size = 100
iter = 5

def labeldoc_vect(alldocs):
    print 'LabelDoc2Vec with lineNO as ID'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    # unlable_docs = [doc for doc in alldocs if doc.split == 'extra']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs:
        slable = ['s'+str(doc.sentiment)]
        sentence = LabeledTaggedDocument(doc.words, doc.tags, slable)
        documents.append(sentence)
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
        train_targets, train_regressors = zip(*[(doc.sentiment, model.docvecs[doc.tags[0]]) for doc in train_docs])
        test_targets, test_regressors = zip(*[(doc.sentiment, model.infer_vector_label(doc.words)) for doc in test_docs])
        data_util.logit(train_regressors, train_targets, test_regressors, test_targets)

if __name__ == '__main__':
    data = data_util.get_ng_data()
    # doc_vect(data)
    # class_vect(data)
    labeldoc_vect(data)

