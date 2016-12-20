# python 2
# https://www.kaggle.com/c/titanic/details/getting-started-with-python

import csv as csv
import numpy as np

### OPEN FILES ###
# open train file
train_file = open('./train.csv', 'rb')
train_file_obj = csv.reader(train_file)
# get headers
train_header = train_file_obj.next()

# open test file
test_file = open('./test.csv', 'rb')
test_file_obj = csv.reader(test_file)
test_header = test_file_obj.next()

# open pred file to write our prediction
pred_file = open('./prediction_gender_pclass_fare.csv', 'wb')
pred_file_obj = csv.writer(pred_file)
# set pred_file header
pred_file_obj.writerow(["PassengerId", "Survived"])



### DATA PROCESSING ###
# prepare data
train_data = []
for row in train_file_obj:
    train_data.append(row)

train_data = np.array(train_data)

test_data = []
for row in test_file_obj:
    test_data.append(row)

test_data = np.array(test_data)

# fare price
fare_ceiling = 40
fare_bracket_size = 10
num_price_brackets = fare_ceiling / fare_bracket_size

train_fare = train_data[:,9].astype(np.float)
train_fare[train_fare >= fare_ceiling] = fare_ceiling - 1.0
train_fare = (train_fare / fare_bracket_size).astype(int)
train_data[:,9] = train_fare

test_data[test_data[:,8] == '', 8] = 3 - test_data[test_data[:,8] == '', 1].astype(int)
test_fare = test_data[:,8].astype(np.float)
test_fare[test_fare >= fare_ceiling] = fare_ceiling - 1.0
test_fare = (test_fare / fare_bracket_size).astype(int)
test_data[:,8] = test_fare


# classes
train_data[:,2] = train_data[:,2].astype(int) - 1
test_data[:,1] = test_data[:,1].astype(int) - 1
num_classes = len(np.unique(train_data[:,2]))

# gender
train_data[train_data[:,4] == "female", 4] = 0
train_data[train_data[:,4] == "male", 4] = 1

test_data[test_data[:,3] == "female", 3] = 0
test_data[test_data[:,3] == "male", 3] = 1
num_genders = len(np.unique(train_data[:, 4]))

# init survival table
survival_table = np.zeros((num_genders, num_classes, num_price_brackets))

for gender in xrange(num_genders):
    for pclass in xrange(num_classes):
        for price in xrange(num_price_brackets):
            stats = train_data[(train_data[:,4].astype(int) == gender) & (train_data[:,2].astype(int) == pclass) & (train_data[:,9].astype(int) == price), 1]
            if len(stats) > 0:
                survival_table[gender, pclass, price] = np.mean(stats.astype(np.float))
            else:
                survival_table[gender, pclass, price] = 0.0

# quantize
survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1


### PREDICT AND WRITE RESTULS ###
for row in test_data:
    pred_file_obj.writerow([row[0], "%d" % int(survival_table[row[3], row[1], row[8]])])


# close files
train_file.close()
test_file.close()
pred_file.close()