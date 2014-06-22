import csv

__author__ = 'kensk8er'

if __name__ == '__main__':
    file1 = open('result/result_total_.csv', 'r')
    answers1 = []
    answers2 = []
    new_answers = []
    data_num = 452167

    row = 0
    for line in file1:
        # print progress
        if row % 1000 == 0:
            print '\r', float(row) / data_num * 100, '%',
        line = line.rstrip()
        data = line.split(',')
        categories = data[1].split(' ')
        categories = set(categories)
        answers1.append(categories)
        row += 1

    file2 = open('result/KnnBaseline_3_wid_.txt', 'r')
    row = 0
    for line in file2:
        # print progress
        if row % 1000 == 0:
            print '\r', float(row) / data_num * 100, '%',
        line = line.rstrip()
        data = line.split(',')
        categories = data[1].split(' ')
        categories = set(categories)
        answers2.append(categories)
        row += 1

    # take uniton
    for i in xrange(row):
        # print progress
        if i % 1000 == 0:
            print '\r', float(i) / data_num * 100, '%',
        new_answers.append(answers1[i] | answers2[i])

    # print
    writecsv = csv.writer(file('result/new.csv', 'wb'), lineterminator='\n')
    for i in xrange(row):
        # print progress
        if i % 1000 == 0:
            print '\r', float(i) / data_num * 100, '%',
        categories = " ".join(new_answers[i])
        writecsv.writerow([i + 1, categories])
