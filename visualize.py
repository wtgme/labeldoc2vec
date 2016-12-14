# -*- coding: utf-8 -*-
"""
Created on 9:33 PM, 10/14/16

@author: tw

Ref: http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
http://matplotlib.org/users/legend_guide.html
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# from scipy.spatial import distance


def draw_words(vectors, words, alternate=True, arrows=True, title=''):
    # sims = cosine_similarity(vectors)
    # # print sims.shape
    # sims[sims>1.0] = 1.0
    # sims = np.matrix(sims)
    #
    # print sims
    # dis = 1. - sims
    # # print dis
    # tsne = TSNE(random_state=0, metric='precomputed')
    # vectors2d = tsne.fit_transform(dis)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    np.set_printoptions(suppress=True)
    vectors2d = tsne.fit_transform(vectors)
    word_list = list(set(words))
    nCols = len(word_list)
    colors = cm.rainbow(np.linspace(0, 1, nCols))

    first = True # color alternation to divide given groups

    for point, word in zip(vectors2d, words):
        # plot points
        plt.scatter(point[0], point[1], c=colors[word_list.index(word)])
        # plot word annotations
        # plt.annotate(
        #     word,
        #     xy = (point[0], point[1]),
        #     xytext = (-10, -10) if first else (10, -10),
        #     textcoords = 'offset points',
        #     ha = 'right' if first else 'left',
        #     va = 'bottom',
        #     size = "x-large",
        #     # arrowprops=dict(arrowstyle="->")
        # )
        # first = not first if alternate else first

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
    handles_list = []
    for c, word in zip(colors, word_list):
        handles_list.append(mpatches.Patch(color=c, label=word))

    lgd = plt.legend(handles=handles_list, loc='center left', bbox_to_anchor=(1, 0.5))
    # draw diagram title
    # if title:
    #     plt.title(title)

    plt.savefig(title+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight', rasterized = True)
    plt.clf()

'''# get trained model
model = gensim.models.Word2Vec.load_word2vec_format("model/SG-300-5-NS10-R50.model", binary=True)
# draw pca plots
draw_words(model, currency, True, True, True, -3, 3, -2, 2, r'$PCA\ Visualisierung:\ W\ddot{a}hrung$')
draw_words(model, capital, True, True, True, -3, 3, -2, 2.2, r'$PCA\ Visualisierung:\ Hauptstadt$')
draw_words(model, language, True, True, True, -3, 3, -2, 1.7, r'$PCA\ Visualisierung:\ Sprache$')'''

def draw_words_ano(vectors, words, alternate=True, arrows=True, title=''):
    # sims = cosine_similarity(vectors)
    # # print sims.shape
    # sims[sims>1.0] = 1.0
    # sims = np.matrix(sims)
    #
    # print sims
    # dis = 1. - sims
    # # print dis
    # tsne = TSNE(random_state=0, metric='precomputed')
    # vectors2d = tsne.fit_transform(dis)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    np.set_printoptions(suppress=True)
    vectors2d = tsne.fit_transform(vectors)
    word_list = list(set(words))
    nCols = len(word_list)
    colors = cm.rainbow(np.linspace(0, 1, nCols))

    first = True # color alternation to divide given groups
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    for point, word in zip(vectors2d, words):
        # plot points
        plt.scatter(point[0], point[1], c=colors[word_list.index(word)])
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
    # handles_list = []
    # for c, word in zip(colors, word_list):
    #     handles_list.append(mpatches.Patch(color=c, label=word))
    #
    # lgd = plt.legend(handles=handles_list, loc='center left', bbox_to_anchor=(1, 0.5))
    # draw diagram title
    # if title:
    #     plt.title(title)

    plt.savefig(title+'.pdf')
    plt.clf()
if __name__ == '__main__':
    word_list = range(10)
    print word_list
    nCols = len(word_list)
    colors = cm.rainbow(np.linspace(0, 1, nCols))
    print colors
    handles_list = []
    for c, word in zip(colors, word_list):
        handles_list.append(mpatches.Patch(color=c, label=str(word)))

    plt.legend(handles=handles_list)
    plt.show()

    # red_patch = mpatches.Patch(color='red', label='The red data')
    # plt.legend(handles=[red_patch])
    #
    # plt.show()