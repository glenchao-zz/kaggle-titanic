# python 3
# https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests

# http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
# https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii

import pandas as pd
import numpy as np
import csv as csv

def prepareData(filePath):
    df = pd.read_csv(filePath, header=0)

    # create "Gender" from "Sex"
    df["Gender"] = df["Sex"].map({"female": 0, "male": 1}).astype(int)

    # create "AgeFill" from "Age" to fill NA values
    df["AgeFill"] = df["Age"]
    median_ages = np.zeros((2, 3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i, j] = df[(df["Gender"] == i) & (df["Pclass"] == j+1)]["Age"].dropna().median()

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df["Age"].isnull()) & (df["Gender"] == i) & (df["Pclass"] == j+1), "AgeFill"] = median_ages[i, j]

    df.loc[df.Fare.isnull(), "Fare"] = df.Fare.median()

    # create "AgeIsNull" to track entries where age is null
    df["AgeIsNull"] = pd.isnull(df["Age"]).astype(int)

    # create "FamilySize" from "Parch" and "SibSp"
    df["FamilySize"] = df["Parch"] + df["SibSp"]

    # create "Age*Class" from "Age" and "Pclass"
    # This amplifies 3rd class (3 is a higher multiplier) at the same time it amplifies
    # older ages. Both of these were less likely to survive, so in theory this could be useful.
    df["Age*Class"] = df["AgeFill"] * df["Pclass"]

    # drop string columns
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    # drop "Age" since it has na values
    df = df.drop(['Age'], axis=1)

    # convert panda dataframe to numpy arrays
    return df.values

train_data = prepareData("train.csv")
test_data = prepareData("test.csv")

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,2::],train_data[0::,1])
output = forest.predict(test_data[0::,1::])

# open pred file to write our prediction
pred_file = open('./prediction_randomforest.csv', 'w', newline='')
pred_file_obj = csv.writer(pred_file)
# set pred_file header
pred_file_obj.writerow(["PassengerId", "Survived"])
for i in range(0, len(output)):
    pred_file_obj.writerow([int(test_data[i,0]), int(output[i])])

pred_file.close()