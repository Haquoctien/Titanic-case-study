# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 08:28:20 2019

@author: tien
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score

# đọc file
f = open("Dataset.data")
data = np.array([line.split() for line in f.readlines()])
f.close()
# tách features và nhãn => đầu vào
X = data[:, :3]
Y = data[:, 3]

# one hot encode các features và nhãn để công bằng trong việc so sánh các giá
# trị của các features
X_ohe = OneHotEncoder(handle_unknown="ignore")
X_ohe.fit(X)
X = X_ohe.transform(X)

Y_ohe = LabelBinarizer()
Y_ohe.fit(Y)
Y = Y_ohe.transform(Y)

# tách training set và testing set từ data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1, stratify = Y)

nb = BernoulliNB()
nb.fit(x_train, y_train.ravel())
y_pred = nb.predict(x_test)
roc_auc_score(y_test, y_pred)