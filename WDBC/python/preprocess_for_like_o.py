# -*- coding: utf-8 -*-
import csv

index_array_for_iid_pid_like_o = [0, 1, 36]

numberOfRow = 0  # number of the row in the original dataset
numberOfNewRow = 0  # number of the row in the original dataset
newData = []
newHeader = []
# this opens the csv file with 'rU' because there was otherwise an error
# with the large csv file
original_file_location = '../data/speeddating_preprocessed_id_mean_train.csv'
data = open(original_file_location, 'rU')  # the original data set
with data as aFile:
    csvReader = csv.reader(aFile)

    for row in csvReader:  # for every row in the dataset
        numberOfRow += 1
        #  Set Header
        if numberOfRow == 1:
            for i in index_array_for_iid_pid_like_o:
                newHeader.append(row[i])
            newData.append(newHeader)
            continue

    print(newHeader)

    print(newData)

    # myFile = open('../data/speeddating_preprocessed_id_mean_.csv', 'w')
    # with myFile:
    #     writer = csv.writer(myFile)
    #     writer.writerows(newData)

    data.close()

