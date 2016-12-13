# -*- coding: utf-8 -*-
"""
Created on 19:44, 18/10/16

@author: wt
"""
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models.labeldoc2vec import LabelDoc2Vec
from gensim.models.labeldoc2vec import LabeledTaggedDocument
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import visualize

def test():
    documents = []
    documents.append(LabeledTaggedDocument('this book is good , I like it'.split(), ['1'], ['g']))
    documents.append(LabeledTaggedDocument('this book is bad , I hate it'.split(), ['2'], ['b']))
    documents.append(LabeledTaggedDocument('this room is good , I like it'.split(), ['3'], ['g']))
    documents.append(LabeledTaggedDocument('this room is bad , I hate it'.split(), ['4'], ['b']))
    # model =Doc2Vec(documents, dm=0, size=100, window=3, negative=5, hs=1, sample=1e-4, iter=100, min_count=1, workers=1)

    model = LabelDoc2Vec(documents, dm=1, dm_concat=1, size=100, window=3, negative=3, hs=1, iter=20, min_count=1, workers=1)
    print('%s:\n %s' % (model, model.docvecs.most_similar('1')))
    # print model.docvecs['1']
    # v1 = model.infer_vector('this book is good'.split())
    # v2 = model.infer_vector('this book is bad'.split())
    # print (1 - spatial.distance.cosine(v1, v2))
    print 'Similiarity of Bad and Good: ', model.similarity('bad', 'good')
    print 'Similiarity of Book and Room: ', model.similarity('book', 'room')
    # # print model.similarity('book', 'room') - model.similarity('bad', 'good')
    # # print model.docvecs.similarity('1', '0')


def test2():
    documents = []
    documents.append(TaggedDocument('this book is good , I like it'.split(), ['1']))
    documents.append(TaggedDocument('this book is bad , I hate it'.split(), ['2']))
    documents.append(TaggedDocument('this room is good , I like it'.split(), ['3']))
    documents.append(TaggedDocument('this room is bad , I hate it'.split(), ['4']))
    # model =Doc2Vec(documents, dm=0, size=100, window=3, negative=5, hs=1, sample=1e-4, iter=100, min_count=1, workers=1)

    model = Doc2Vec(documents, dm=1, dm_concat=1, size=100, window=3, negative=5, hs=1, sample=1e-3, iter=10, min_count=0, workers=1)
    # model = Doc2Vec(documents, dm=0, dbow_words=1, size=50, window=5, negative=0, hs=1, iter=2000, min_count=0, workers=1)
    print('%s:\n %s' % (model, model.docvecs.most_similar('1')))
    # print model.docvecs['1']
    # v1 = model.infer_vector('this book is good'.split())
    # v2 = model.infer_vector('this book is bad'.split())
    # # print (1 - spatial.distance.cosine(v1, v2))
    print 'Similiarity of Bad and Good: ', model.similarity('bad', 'good')
    print 'Similiarity of Book and Room: ', model.similarity('book', 'room')
    # # print model.similarity('book', 'room') - model.similarity('bad', 'good')
    # # print model.docvecs.similarity('1', '0')
    words = ['book', 'room', 'good', 'bad', 'like', 'hate']
    vectors = []
    for word in words:
        vectors.append(model.infer_vector(word))
    visualize.draw_words_ano(vectors, words, True, False, r'wdoc2Vec')


def test3():
    documents = []
    documents.append(TaggedDocument('this book is good , I like it'.split(), ['1']))
    documents.append(TaggedDocument('this book is bad , I hate it'.split(), ['0']))
    documents.append(TaggedDocument('this room is good , I like it'.split(), ['1']))
    documents.append(TaggedDocument('this room is bad , I hate it'.split(), ['0']))
    # model =Doc2Vec(documents, dm=0, size=100, window=3, negative=5, hs=1, sample=1e-4, iter=100, min_count=1, workers=1)
    model = Doc2Vec(documents, dm=1, dm_concat=1, size=100, window=3, negative=3, hs=1, sample=1e-3, iter=20, min_count=0, workers=1)
    # model = Doc2Vec(documents, dm=0, dbow_words=1, size=100, window=3, negative=3, hs=1, iter=10, min_count=0, workers=1)
    print('%s:\n %s' % (model, model.docvecs.most_similar('1')))
    # print model.docvecs['1']
    # v1 = model.infer_vector('this book is good'.split())
    # v2 = model.infer_vector('this book is bad'.split())
    # # print (1 - spatial.distance.cosine(v1, v2))
    print 'Similiarity of Bad and Good: ', model.similarity('bad', 'good')
    print 'Similiarity of Book and Room: ', model.similarity('book', 'room')
    # # print model.similarity('book', 'room') - model.similarity('bad', 'good')
    # # print model.docvecs.similarity('1', '0')
    words = ['book', 'room', 'good', 'bad', 'like', 'hate']
    vectors = []
    for word in words:
        vectors.append(model[word])
    visualize.draw_words_ano(vectors, words, True, False, r'wclass2Vec')


def test4():
    documents = []
    documents.append(('this book is good , I like it'.split()))
    documents.append(('this book is bad , I hate it'.split()))
    documents.append(('this room is good , I like it'.split()))
    documents.append(('this room is bad , I hate it'.split()))
    model = Word2Vec(documents, size=100, window=3, negative=3, hs=1, sample=1e-3, iter=20, min_count=0, workers=1)
    # model =Doc2Vec(documents, dm=0, size=100, window=3, negative=5, hs=1, sample=1e-4, iter=100, min_count=1, workers=1)
    # model = Doc2Vec(documents, dm=1, dm_concat=1, size=50, window=3, negative=5, hs=1, sample=1e-3, iter=10, min_count=0, workers=1)
    # model = Doc2Vec(documents, dm=0, dbow_words=1, size=50, window=5, negative=0, hs=1, iter=2000, min_count=0, workers=1)
    # print('%s:\n %s' % (model, model.docvecs.most_similar('1')))
    # print model.docvecs['1']
    # v1 = model.infer_vector('this book is good'.split())
    # v2 = model.infer_vector('this book is bad'.split())
    # # print (1 - spatial.distance.cosine(v1, v2))
    print 'Similiarity of Bad and Good: ', model.similarity('bad', 'good')
    print 'Similiarity of Book and Room: ', model.similarity('book', 'room')
    # # print model.similarity('book', 'room') - model.similarity('bad', 'good')
    # # print model.docvecs.similarity('1', '0')
    words = ['book', 'room', 'good', 'bad', 'like', 'hate']
    vectors = []
    for word in words:
        vectors.append(model[word])
    visualize.draw_words_ano(vectors, words, True, False, r'word2Vec')


# test2()
test3()
test4()