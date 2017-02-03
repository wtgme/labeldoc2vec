# -*- coding: utf-8 -*-
"""
Created on 15:25, 06/10/16

@author: wt
"""

import multiprocessing
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from sklearn import svm
from collections import namedtuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import re
import sys
import unicodedata
unicode_punc_tbl = dict.fromkeys( i for i in xrange(128, sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P') )


def get_scores( true_classes, pred_classes, average):
    precision = metrics.precision_score( true_classes, pred_classes, average=average )
    recall = metrics.recall_score( true_classes, pred_classes, average=average )
    f1 = metrics.f1_score( true_classes, pred_classes, average=average )
    accuracy = metrics.accuracy_score( true_classes, pred_classes )
    return precision, recall, f1, accuracy


def print_scores(pred_train_classes, true_train_classes, pred_test_classes, true_test_classes):
    print metrics.classification_report(true_train_classes, pred_train_classes, digits=3)
    print metrics.classification_report(true_test_classes, pred_test_classes, digits=3)

    for average in ['micro', 'macro']:
        train_precision, train_recall, train_f1, train_acc = get_scores( true_train_classes, pred_train_classes, average )
        print "Train Prec (%s average): %.3f, recall: %.3f, F1: %.3f, Acc: %.3f" %( average,
                            train_precision, train_recall, train_f1, train_acc )

        test_precision, test_recall, test_f1, test_acc = get_scores( true_test_classes, pred_test_classes, average )
        print "Test Prec (%s average): %.3f, recall: %.3f, F1: %.3f, Acc: %.3f" %(  average,
                            test_precision, test_recall, test_f1, test_acc )


def logit(X_train, y_train, X_test, y_test):
    logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', n_jobs=multiprocessing.cpu_count())
    # svc_lin = SVC(kernel='linear', class_weight='balanced')
    logistic.fit(X_train, y_train)
    y_lin = logistic.predict(X_test)
    y_tlin = logistic.predict(X_train)
    print "Logit Performance"
    print_scores(y_tlin, y_train, y_lin, y_test)


def model_similar(model, X_train, y_train, X_test, y_test):
    y_lin = []
    y_tlin = []
    for doc in X_train:
        y_tlin.append(model.docvecs.most_similar(positive=[doc], topn=1)[0][0])
    for doc in X_test:
        y_lin.append(model.docvecs.most_similar(positive=[doc], topn=1)[0][0])
    print "Similarity Performance"
    print_scores(y_tlin, y_train, y_lin, y_test)


def svm_class(X_train, y_train, X_test, y_test):
    # # logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', n_jobs=multiprocessing.cpu_count())
    # svc_lin = SVC(kernel='linear', class_weight='balanced')
    # svc_lin.fit(X_train, y_train)
    # y_lin = svc_lin.predict(X_test)
    # y_tlin = svc_lin.predict(X_train)
    # print "SVM Performance"
    # print_scores(y_tlin, y_train, y_lin, y_test)
    model = svm.LinearSVC(penalty='l1', dual=False)
    model.fit( X_train, y_train )
    pred_train_classes = model.predict( X_train )
    pred_test_classes = model.predict( X_test )
    print_scores(pred_train_classes, y_train, pred_test_classes, y_test)


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
        print "Pre-classification Performance:"
        print_scores(y_lin, y_test)
    return y_lin


def extract_words(doc, remove_url=True, remove_punc="utf-8", min_length=1):
    doc = doc.encode("utf-8").lower()
    doc = doc.replace('<br />', ' ')
    if remove_punc:
        # ensure doc_u is in unicode
        if not isinstance(doc, unicode):
            encoding = remove_punc
            doc_u = doc.decode(encoding)
        else:
            doc_u = doc
        # remove unicode punctuation marks, keep ascii punctuation marks
        doc_u = doc_u.translate(unicode_punc_tbl)
        if not isinstance(doc, unicode):
            doc = doc_u.encode(encoding)
        else:
            doc = doc_u
    if remove_url:
        re_url = r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        doc = re.sub( re_url, "", doc )
    sentences = re.split( r"\s*[,;:`\"()?!{}]\s*|--+|\s*-\s+|''|\.\s|\.$|\.\.+|¡°|¡±", doc ) #"
    wc = 0
    wordsInSentences = []
    for sentence in sentences:
        if sentence == "":
            continue
        if not re.search( "[A-Za-z0-9]", sentence ):
            continue
        words = re.split( r"\s+\+|^\+|\+?[\-*\/&%=<>\[\]~\|\@\$]+\+?|\'\s+|\'s\s+|\'s$|\s+\'|^\'|\'$|\$|\\|\s+", sentence )
        words = filter( lambda w: w, words )
        if len(words) >= min_length:
            for word in words:
                wordsInSentences.append(word)
            wc += len(words)
    return wordsInSentences


# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.encode("utf-8").lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    return norm_text


def get_imdb_data():
    # try:
    #     alldocs = pickle.load(open('imdb.data', 'r'))
    # except Exception:
    LabelDocument = namedtuple('LabelDocument', 'words tags split label')
    alldocs = []  # will hold all docs in original order
    with open('aclImdb/alldata-id.txt', 'r') as alldata:
        for line_no, line in enumerate(alldata):
            tokens = extract_words(line)
            words = tokens[1:]
            tags = [line_no]  # `tags = [tokens[0]]` would also work at extra memory cost
            split = ['train', 'test', 'extra', 'extra'][line_no // 25000]  # 25k train, 25k test, 25k extra
            label = ['P', 'N', 'P', 'N', None, None, None, None][
                line_no // 12500]  # [12.5K pos, 12.5K neg]*2 then unknown
            alldocs.append(LabelDocument(words, tags, split, label))
        # pickle.dump(alldocs, open('imdb.data', 'w'))
    return alldocs


def ng_data():
    from sklearn.datasets import fetch_20newsgroups
    # remove = ('headers', 'footers', 'quotes')
    data_train = fetch_20newsgroups(subset='train')
    data_test = fetch_20newsgroups(subset='test')
    y_train, y_test = data_train.target, data_test.target # Label ID from 0 to 19
    names = data_train.target_names
    docs_dict = {}

    for line_no, line in enumerate(data_train.data):
        words = extract_words(line)
        label = y_train[line_no]  # [12.5K pos, 12.5K neg]*2 then unknown
        docs = docs_dict.get(label, [])
        docs.append(words)
        docs_dict[label] = docs

    for key in docs_dict:
        print '---------------%s------------------' %key
        docs = docs_dict[key]
        print 'numbers of documents: %d' %len(docs)
        avenum = np.mean([len(doc) for doc in docs])
        print 'average number of words in doc: %.3f' % avenum
        uniqword = [w for w in doc for doc in docs]
        print 'Number of word: %d' %len(uniqword)
        print 'Number of unique words: %d' %len(set(uniqword))
        print 'Average number of frequences of word: %.3f' %(len(uniqword)/len(set(uniqword)))


def remove_duplic(data):
    from collections import Counter
    names = []
    for name in data.filenames:
        names.append(name.split('/')[-1])
    dup_docs = [item for item, count in Counter(names).iteritems() if count > 1]

    datal, filenamesl, targetl = [], [], []
    for ids, name in enumerate(names):
        if name not in dup_docs:
            datal.append(data.data[ids])
            filenamesl.append(data.filenames[ids])
            targetl.append(data.target[ids])
    data.data = datal
    data.filenames = filenamesl
    data.target = targetl
    return data

def get_ohsumed_dat(filespath):
    import sklearn.datasets as data_loader
    train = data_loader.load_files(filespath+'/training/')
    test = data_loader.load_files(filespath+'/test/')
    print len(train.data)
    print len(test.data)

    train = remove_duplic(train)
    test = remove_duplic(test)
    print len(train.data)
    print len(test.data)

    Target = True
    for i in xrange(len(train.target_names)):
        if train.target_names[i] != test.target_names[i]:
            Target = False
            print '-------------------------------------'
            break
    if Target:
        import copy
        all = copy.deepcopy(train)
        all.data = np.concatenate((train.data, test.data))
        all.filenames = np.concatenate((train.filenames, test.filenames))
        all.target = np.concatenate((train.target, test.target))
        print len(all.data)
        print len(train.data)
        print len(test.data)

        from collections import Counter
        names = []
        all = remove_duplic(all)
        for name in all.filenames:
            names.append(name.split('/')[-1])
        dup_docs = [item for item, count in Counter(names).iteritems() if count > 1]
        print len(dup_docs)
        print len(set(dup_docs))
        names = np.array(names)
        for item in dup_docs:
            docs = all.filenames[np.where(names==item)]
            print docs
            # print docs[0].split('/')[-3]
            sets = set([sn.split('/')[-3] for sn in docs])
            if len(sets) > 2:
                print docs


def get_ng_data():
    # try:
    #     alldocs = pickle.load(open('20ng.data', 'r'))
    # except Exception:
    from sklearn.datasets import fetch_20newsgroups
    # remove = ('headers', 'footers', 'quotes')
    data_train = fetch_20newsgroups(subset='train')
    data_test = fetch_20newsgroups(subset='test')
    y_train, y_test = data_train.target, data_test.target # Label ID from 0 to 19
    names = data_train.target_names

    LabelDocument = namedtuple('LabelDocument', 'words tags split label')
    alldocs = []
    for line_no, line in enumerate(data_train.data):
            words = extract_words(line)
            tags = [line_no]  # `tags = [tokens[0]]` would also work at extra memory cost
            split = 'train'
            label = names[y_train[line_no]]  # [12.5K pos, 12.5K neg]*2 then unknown
            alldocs.append(LabelDocument(words, tags, split, label))
    train_len = len(data_train.data)
    for line_no, line in enumerate(data_test.data):
            words = extract_words(line)
            tags = [line_no+train_len]  # `tags = [tokens[0]]` would also work at extra memory cost
            split = 'test'
            label = names[y_test[line_no]]  # [12.5K pos, 12.5K neg]*2 then unknown
            alldocs.append(LabelDocument(words, tags, split, label))
        # pickle.dump(alldocs, open('20ng.data', 'w'))
    return alldocs


def get_reuters_data():
    # try:
    #     alldocs = pickle.load(open('10re.data', 'r'))
    # except Exception:
    from nltk.corpus import reuters
    import HTMLParser
    html = HTMLParser.HTMLParser()
    doc_ids = reuters.fileids()
    '''Select top 10 largest categories'''
    cat2all_num = {}
    for doc_id in doc_ids:
        # only choose docs belonging in one category
        if len( reuters.categories(doc_id) ) == 1:
            cat = reuters.categories(doc_id)[0]
            if cat in cat2all_num:
                cat2all_num[cat] += 1
            else:
                cat2all_num[cat] = 1

    sorted_cats = sorted(cat2all_num.keys(), key=lambda cat: cat2all_num[cat],
                            reverse=True )
    catNum = 10
    topN_cats = sorted_cats[:catNum]
    print "Top 10 categories:"
    for cat in topN_cats:
        print "%s: %d" %(cat, cat2all_num[cat])
    '''Prepare Data'''
    LabelDocument = namedtuple('LabelDocument', 'words tags split label')
    alldocs = []  # will hold all docs in original order
    line_no = 0
    for doc_id in doc_ids:
        cat = reuters.categories(doc_id)[0]
        if len(reuters.categories(doc_id)) == 1 and (cat in topN_cats):
            line = html.unescape(reuters.raw(doc_id))
            words = extract_words(line)
            tags = [line_no]
            if doc_id.startswith("train"):
                split = 'train'
            else:
                split = 'test'
            label = cat
            alldocs.append(LabelDocument(words, tags, split, label))
            line_no += 1
        # pickle.dump(alldocs, open('10re.data', 'w'))
    return alldocs


def get_rcv():
    from sklearn.datasets import fetch_rcv1
    rcv1 = fetch_rcv1()
    for doc in rcv1.data:
        print doc


def sim_ratio(vectors, labels):
    sims = cosine_similarity(vectors)
    size = len(vectors)
    sim_inter, sim_intra, count_inter, count_intra = 0.0, 0.0, 0.0, 0.0
    for i in xrange(size):
        for j in xrange(i+1, size):
            if labels[i] == labels[j]:
                sim_intra += sims[i][j]
                count_intra += 1
            else:
                sim_inter += sims[i][j]
                count_inter += 1
    sim_inter_avg = sim_inter/count_inter
    sim_intra_avg = sim_intra/count_intra
    ratio = sim_intra_avg/sim_inter_avg
    print 'Intra_similarity: %.3f, \t Inter_simiarlty: %.3f, \t Ratio: %.3f' %(sim_intra_avg, sim_inter_avg, ratio)


if __name__ == '__main__':
    # ng_data()
    get_ohsumed_dat('/home/wt/grive/baselines/topicvec-master/ohsumed/')

