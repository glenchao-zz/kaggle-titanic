# kaggle-titanic
https://www.kaggle.com/c/titanic

* `predict_gender.py` generates `prediction_gender.csv` by simply predicting all female survived and all male died
* `predict_numpy.py` uses numpy lib to generate `predction_gender_pclass_fare.csv` by calculating the probably of
survival given the `gender`, `pclass`, and `fare` of the passenger.
* `pandas_dataframe.py` uses the pandas package to clean data and engineer features