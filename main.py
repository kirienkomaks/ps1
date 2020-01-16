import pandas as pd
import numpy as np
from numpy import genfromtxt
import csv

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import utils

from sklearn.linear_model  import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.naive_bayes  import GaussianNB
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

#my_data = genfromtxt('SUB1107-BioSigs.csv', delimiter=';')
#my_data = my_data[~np.isnan(my_data)]
#datas=np.delete(my_data,2,1)

#y_train=[]
#with open('SUB1107-BioSigs.csv') as csvfile:
    #my_data = csv.reader(csvfile, delimiter=';')


data = pd.read_csv('SUB1107-BioSigs.csv',  delimiter=';', usecols=['TIMESTAMP','ECG','EDA'],nrows=40000)
data = data.dropna(axis=0)

data_first = data['ECG'].astype(float)
data_first = data_first.values
data_y = data['EDA'].astype(float)
data_y = data_y.values

#print(data_first)
#print(data_y)
#############
#data_first = data_first.reshape(-1,1)
#data_y = data_y.reshape(-1,1)

test_data_first = np.array([1555650712.942,1555650712.943,1555650712.944])
#test_data_first = test_data_first.reshape(-1,1)

test_data_y = np.array([-0.02933250340207528,-0.02760435620159596,-0.025863119205993924])
test_data_y = test_data_y.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(data_first,data_y,test_size=0.2, random_state=4)
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

sc =StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#print(utils.multiclass.type_of_target(data_y))

#print(utils.multiclass.type_of_target(data_y.astype('int')))

#print(utils.multiclass.type_of_target(test_data_y))

print("Running learning machine...")

model = svm.SVR()
#model = LinearRegression()
#model = GaussianNB()
model.fit(X_train, y_train.ravel())
print("End learning!")

predicted_val = model.predict(test_data_y)
print("Result:   ",predicted_val)