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


def classify(similarity_file, output_file, start_id):
    print 'load data...'
    similarities = unpickle(similarity_file)
    Y_train = unpickle('data/train_label.pkl')
    Y_test = unpickle('data/test_label.pkl')
    test_len = len(similarities)

    print 'load relatives...'
    #relatives = unpickle('data/relatives.pkl')

    # hyper parameters
    alpha = 0.25
    k = 10
    threshold = 0.35
    default_category = 24177  # this is the most frequent category in the train data
    predicted = {}
    #relative_coef = 0.1

    # iterate over every test document
    print 'classify categories for each test document...'
    for test_doc_id in xrange(test_len):  # test_doc_id starts from 0, thus you need to add 1 to this at the end
        # print progress
        if test_doc_id % 1000 == 0:
            print '\r', float(test_doc_id) / test_len * 100, '%',

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

                    # take the hierarchical information into account
                    #if relatives.has_key(category):
                    #    relative_categories = relatives[category]
                    #    for relative_category in relative_categories:
                    #        scores[relative_category] += similarity * relative_coef

            # sort by descending order
            scores = scores.items()
            scores.sort(key=lambda entry: entry[1])
            scores.reverse()
            max_score = scores[0][1]

            # choose top-x categories based on alpha value
            categories = []
            #previous_score = 0
            for score_tuple in scores:
                category = score_tuple[0]
                score = score_tuple[1]

                if max_score > 0:  # some test data don't have any feature so max_score could be zero
                    if score > threshold:
                        if score / max_score > alpha:
                            categories.append(category)
                    elif len(categories) == 0:
                        categories.append(default_category)
                        break
                else:
                    categories.append(default_category)

        else:  # procedures when there isn't any non-zero similarity train document
            categories = [default_category]  # TBF: room to improve

        # insert categories into predictions
        document_id = int(Y_test[test_doc_id + start_id][1])  # test_doc_id starts from zero so you can't directly use it
        predicted[document_id] = categories
    print '\r100 % done!'

    # write to csv file
    print 'save the result into csv file...'
    writecsv = csv.writer(file(output_file, 'a+b'), lineterminator='\n')

    predicted = predicted.items()
    predicted.sort()

    for pred in predicted:
        document_id = pred[0]
        categories = " ".join(map(str, pred[1]))
        writecsv.writerow([document_id, categories])

    print 'save into pickle...'
    enpickle(predicted, 'result/predicted.pkl')

    return test_len

if __name__ == '__main__':
    output_file_name = 'result/result_total.csv'
    similarity_file_name = 'similarity/similarities_test'

    # initial write
    writecsv = csv.writer(file(output_file_name, 'wb'), lineterminator='\n')
    writecsv.writerow(['Id', 'Predicted'])

    start_id = 0
    for i in range(1, 16):
        end_id = classify(similarity_file=similarity_file_name+str(i)+'.pkl', output_file=output_file_name, start_id=start_id)
        start_id += end_id

