"""
choose multiple categories based on similarities.
category x is chosen when: score(x) / score(top) > alpha (alpha is a constant between 0 and 1, top is class with the highest score)
score are calculated by looking at k-nearest neighbors
decide alpha by minimizing: error = average_num_of_categories_predicted - average_num_of_categories

hyper parameters: alpha, k
"""
from collections import defaultdict
import csv
from scipy.sparse import csr_matrix
import sys

__author__ = 'kensk8er'

from utils.util import *
import numpy as np


def classify(size):
    print 'load data...'
    starting_id = 0  #20000
    similarities = unpickle('data/similarities_best.pkl')
    Y_train = unpickle('data/train_label.pkl')
    Y_test = unpickle('data/test_label.pkl')
    test_len = len(similarities)

    # hyper parameters
    alpha = 0.7
    k = 5
    predicted = {}

    # iterate over every test document
    progress = 0
    print 'classify categories for each test document...'
    for test_doc_id in xrange(test_len):  # test_doc_id starts from 0, thus you need to add 1 to this at the end
        # print progress
        if (float(test_doc_id) / test_len) * 100 > progress:
            print '\r', progress, '%',
            progress += 1

        # sort by similarity
        #train_doc_ids, similarity_vals = similarities[test_doc_id]
        #train_doc_ids = train_doc_ids[0]
        #similarity_vals = similarity_vals[0]
        #pairs = zip(similarity_vals, train_doc_ids)  # list[tuple(similarity, train_doc_id), ...]
        pairs = similarities[test_doc_id]

        if max(pairs)[0] > 0:  # following procedures are only meaningful when there's non-zero similarity train document
            pairs.sort()
            pairs.reverse()
            if len(pairs) > k:
                pairs = pairs[0: k]  # keep only k-nearest neighbors

            # calculate category scores
            scores = defaultdict(int)
            for pair in pairs:
                similarity = pair[0]
                train_doc_id = pair[1]
                categories = Y_train[train_doc_id]

                for category in categories:
                    category = int(category)
                    scores[category] += similarity  # this algorithm might have room to improve

            # sort by descending order
            scores = scores.items()
            scores.sort(key=lambda entry: entry[1])
            scores.reverse()
            max_score = scores[0][1]

            # choose top-x categories based on alpha value
            categories = []
            previous_score = 0
            for score_tuple in scores:
                category = score_tuple[0]
                score = score_tuple[1]

                #if max_score > 0:  # some test data don't have any feature so max_score could be zero
                #    if score / max_score > alpha:
                #        categories.append(category)
                if max_score > 0:
                    if len(categories) < 3:
                        categories.append(category)
                    #elif score == previous_score:
                    #    categories.append(category)
                    else:
                        break
                else:
                    categories.append(314523)

                previous_score = score

        else:  # procedures when there isn't any non-zero similarity train document
            categories = [314523]  # TBF: room to improve

        # insert categories into predictions
        document_id = int(Y_test[test_doc_id][1])  # test_doc_id starts from zero so you can't directly use it
        predicted[document_id] = categories
    print '\r100 % done!'

    # write to csv file
    print 'save the result into csv file...'
    filename = 'result/result_best10.csv'
    writecsv = csv.writer(file(filename, 'wb'), lineterminator='\n')
    writecsv.writerow(['Id', 'Predicted'])

    predicted = predicted.items()
    predicted.sort()

    for pred in predicted:
        document_id = pred[0] + starting_id
        categories = " ".join(map(str, pred[1]))
        writecsv.writerow([document_id, categories])

    print 'save into pickle...'
    enpickle(predicted, 'result/predicted.pkl')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        size = 'full'
        classify(size)
    elif sys.argv[1] == 'min':
        size = 'min'
        classify(size)

