# https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii

import pandas as pd
import numpy as np
import pylab as P # use for ploting

df = pd.read_csv('./train.csv', header=0)

# useful pandas dataframe functions and props
# df.dtypes
# df.head(#)
# df.info()
# df.describe()

# plot "Age"
df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()

# create "Gender" from "Sex"
df["Gender"] = df["Sex"].map({"female": 0, "male": 1}).astype(int)

# create "AgeFill" from "Age" to fill NA values
df["AgeFille"] = df["Age"]
median_ages = np.zeros((2, 3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages = df[(df["Gender"] == i) & (df["Pclass"] == j+1)]["Age"].dropna().median()

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[(df["Age"].isnull()) & (df["Gender"] == i) & (df["Pclass"] == j+1), "AgeFill"] \
            = median_ages(i, j)

# create "AgeIsNull" to track entries where age is null
df["AgeIsNull"] = pd.isnull(df["Age"]).astype(int)

# create "FamilySize" from "Parch" and "SibSp"
df["FamilySize"] = df["Parch"] + df["SibSp"]

# create "Age*Class" from "Age" and "Pclass"
# This amplifies 3rd class (3 is a higher multiplier) at the same time it amplifies
# older ages. Both of these were less likely to survive, so in theory this could be useful.
df["Age*Class"] = df["Age"] * df["Pclass"]

# drop string columns
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# drop "Age" since it has na values
df = df.drop(['Age'], axis=1)

# convert panda dataframe to numpy arrays
train_data = df.values
