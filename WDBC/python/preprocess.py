# -*- coding: utf-8 -*-
import csv

#  Use Only variables below

# "gender",0
# "age",95
# "age_o",104
# "samerace",0
# "importance_same_race",79
# "pref_o_attractive",89
# "pref_o_sincere",89
# "pref_o_intelligence",89
# "pref_o_funny",98
# "pref_o_ambitious",107
# "pref_o_shared_interests",129
# "attractive_important",79
# "sincere_important",79
# "intelligence_important",79
# "funny_important",89
# "ambition_important",99
# "shared_interests_important",121
# "attractive",105
# "sincere",105
# "intelligence",105
# "funny",105
# "ambition",105
# "attractive_partner",202
# "sincere_partner",277
# "intelligence_partner",296
# "funny_partner",350
# "ambition_partner",712
# "shared_interests_partner",1067
# "like",240
# "met",375
# "decision_o",0
index_array = [2,3,4,9,10,15,16,17,18,19,20,39,40,41,42,43,44,51,52,53,54,55,61,62,63,64,65,66,115,119,121]

numberOfRow = 0  # number of the row in the original dataset
numberOfNewRow = 0  # number of the row in the original dataset
newData = []
newHeader = []
# this opens the csv file with 'rU' because there was otherwise an error
# with the large csv file
original_file_location = '../data/speeddating_train.csv'
data = open(original_file_location, 'rU')  # the original data set
with data as aFile:
    csvReader = csv.reader(aFile)

    for row in csvReader:  # for every row in the dataset
        numberOfRow += 1

        #  Set Header
        if numberOfRow == 1:
            for i in index_array:
                newHeader.append(row[i])
            newData.append(newHeader)
            continue

        isNull = False
        for i in index_array:
            if row[i] in (None, '?'):
                isNull = True
                break

        if not isNull:
            newRow = []
            for i in index_array:
                column = row[i]

                # if gender : male = 0, female : 1
                if i == 2:
                    if column == 'male':
                        column = 0
                    else:
                        column = 1

                newRow.append(column)

            newData.append(newRow)
            numberOfNewRow += 1

    print("numberOfRow: ", numberOfRow, "new Row: ", numberOfNewRow)

    myFile = open('speeddating_train_preprocessed.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(newData)

    data.close()

