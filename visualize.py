# -*- coding: utf-8 -*-
"""
Created on 9:33 PM, 10/14/16

@author: tw
"""

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import distance

def draw_words(vectors, words, alternate=True, arrows=True, title=''):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    vectors2d = tsne.fit_transform(vectors)

    first = True # color alternation to divide given groups
    for point, word in zip(vectors2d, words):
        # plot points
        plt.scatter(point[0], point[1], c='r' if first else 'g')
        # plot word annotations
        plt.annotate(
            word,
            xy = (point[0], point[1]),
            xytext = (-10, -10) if first else (10, -10),
            textcoords = 'offset points',
            ha = 'right' if first else 'left',
            va = 'bottom',
            size = "x-large",
            # arrowprops=dict(arrowstyle="->")
        )
        first = not first if alternate else first

    # draw arrows
    if arrows:
        for i in xrange(0, len(words)-1, 2):
            a = vectors2d[i][0]
            b = vectors2d[i][1]
            c = vectors2d[i+1][0]
            d = vectors2d[i+1][1]
            # print distance.euclidean((a,b), (c,d))
            plt.arrow(a, b, c-a, d-b,
                # shape='full',
                # lw=0.1,
                edgecolor='k',
                facecolor='k',
                # length_includes_head=True,
                # head_width=0.03,
                # width=0.01
                shape='full', lw=1, length_includes_head=True, head_width=15
            )

    # draw diagram title
    if title:
        plt.title(title)

    plt.savefig(title+'.pdf')
    plt.clf()

'''# get trained model
model = gensim.models.Word2Vec.load_word2vec_format("model/SG-300-5-NS10-R50.model", binary=True)
# draw pca plots
draw_words(model, currency, True, True, True, -3, 3, -2, 2, r'$PCA\ Visualisierung:\ W\ddot{a}hrung$')
draw_words(model, capital, True, True, True, -3, 3, -2, 2.2, r'$PCA\ Visualisierung:\ Hauptstadt$')
draw_words(model, language, True, True, True, -3, 3, -2, 1.7, r'$PCA\ Visualisierung:\ Sprache$')'''