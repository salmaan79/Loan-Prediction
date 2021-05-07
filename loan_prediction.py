# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:58:28 2020

@author: Salmaan Ahmed Ansari
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train_ctrUa4K.csv', sep = ',')
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 12:13].values
print(X)
print(y)



# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:, 0:5])
X[:, 0:5] = imputer.transform(X[:, 0:5])
print(X)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:, 8:11])
X[:, 8:11] = imputer.transform(X[:, 8:11])
print(X)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X[:, 5:8])
X[:, 5:8] = imputer.transform(X[:, 5:8])
print(X)


"""#taking care of missing data
dataset['Property_Area'].fillna(dataset['Property_Area'].mode()[0], inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
"""

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
le_X = LabelEncoder()
X[:, 0] = le_X.fit_transform(X[:, 0])
X[:, 1] = le_X.fit_transform(X[:, 1])
X[:, 3] = le_X.fit_transform(X[:, 3])
X[:, 4] = le_X.fit_transform(X[:, 4])
X[:, 10] = le_X.fit_transform(X[:, 10])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [10])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:, 1:]


#encoding dependent variable
from sklearn.preprocessing import LabelEncoder
le_X = LabelEncoder()
y[:, 0] = le_X.fit_transform(y[:, 0])
y=y.astype('int')




# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(C = 0.8, kernel = 'linear', random_state = 0)
classifier.fit(X, y)











dataset_test = pd.read_csv('test_lAUu6dG.csv')
X_tes = dataset_test.iloc[:, 1:12].values





# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X_tes[:, 0:5])
X_tes[:, 0:5] = imputer.transform(X_tes[:, 0:5])
print(X_tes)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X_tes[:, 8:11])
X_tes[:, 8:11] = imputer.transform(X_tes[:, 8:11])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X_tes[:, 5:8])
X_tes[:, 5:8] = imputer.transform(X_tes[:, 5:8])
print(X_tes)


"""#taking care of missing data
dataset['Property_Area'].fillna(dataset['Property_Area'].mode()[0], inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
"""

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
le_X = LabelEncoder()
X_tes[:, 0] = le_X.fit_transform(X_tes[:, 0])
X_tes[:, 1] = le_X.fit_transform(X_tes[:, 1])
X_tes[:, 3] = le_X.fit_transform(X_tes[:, 3])
X_tes[:, 4] = le_X.fit_transform(X_tes[:, 4])
X_tes[:, 10] = le_X.fit_transform(X_tes[:, 10])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [10])], remainder='passthrough')
X_tes = np.array(ct.fit_transform(X_tes))
X_tes = X_tes[:, 1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_tes = sc.fit_transform(X_tes)



# Predicting the Test set results
y_tes = classifier.predict(X_tes)

y_tes



