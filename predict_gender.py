# python 2
import csv as csv

### OPEN FILES ###
# open file
file = open('./test.csv', 'rb')
file_obj = csv.reader(file)
# get headers
header = file_obj.next()

# open pred file to write our prediction
pred_file = open('./prediction_gender.csv', 'wb')
pred_file_obj = csv.writer(pred_file)
# set pred_file header
pred_file_obj.writerow(["PassengerId", "Survived"])

for row in file_obj:
    pred_file_obj.writerow([row[0], 1 if row[3] == "female" else 0])

file.close()
pred_file.close()