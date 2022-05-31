 # -*- coding: utf-8 -*-

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SVM libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Install to use read_excel
#!pip install openpyxl

# Read data set
data = pd.read_excel('Raisin_Dataset.xlsx', sheet_name = 'Raisin_Dataset')
data

# Data frame structure
data.info()

# Convert columns to numeric
data['Class'] = data['Class'].str.replace('Kecimen', '0')
data['Class'] = data['Class'].str.replace('Besni', '1')

cols = data.columns
data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')

# Data frame structure
data.info()

# Classes are distrubuted evenly
data['Class'].value_counts()

# Data for SVM
X = data
X = X.drop(['Class'], axis = 1)
Y = data.Class # class

# Parameters for classificator - rbf kernel
gamma = [0.0005, 0.005, 0.01, 0.05, 0.2, 0.8, 1.5, 2.5, 5, 10, 20, 50, 100]
gamma = np.array(gamma)
C = [1, 10, 100, 1000, 10000, 100000]
C = np.array(C)

# Tables to save accuracy - rbf kernel
Accuracy_CV = np.zeros((10,1))
Accuracy = np.zeros((len(gamma), len(C)))

# SVM - rbf kernel
for i in range(0, len(C)): # C
    for j in range(0, len(gamma)): # gamma
        for k in range(1, 10): # Crossvalidation
            # Split data into test and train sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10) # 10 times cs so test_size is 10% of data set 
            
            # Standarization
            sc = StandardScaler()
            sc.fit(X_train)
            X_train = sc.transform(X_train)
            X_test = sc.transform(X_test)
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            
            # Model
            svclassifier = SVC(kernel = 'rbf', C = C[i], gamma = gamma[j])
            
            svclassifier.fit(X_train, Y_train)
            y_pred = svclassifier.predict(X_test)
            
            # Accuracy - how many values from y_pred are equal to Y_test
            Accuracy_CV[k] = sum(y_pred == Y_test)/len(Y_test)

        Accuracy[j,i] = np.mean(Accuracy_CV) # rows - gamma, columns - C
        
Accuracy # display accuracy table

# rbf kernel accuracy plot
# axis X - gamma, different lines - C     
for p in range(0, len(C)): # number of lines = number of C values 
    plt.plot(Accuracy[:,p], label = C[p]) # every line plotted separately in order to have a name
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.xticks(np.arange(len(gamma)), gamma) # correct axis X ticks
plt.legend(title = 'C')

#X_rbf115_train, X_rbf115_test, Y_rbf115_train, Y_rbf115_test = train_test_split(X, Y, test_size = 0.10) # 10 times cs so test_size is 10% of data set 

#svclassifier_rbf115 = SVC(kernel = 'rbf', C = 1, gamma = 1.5)

#svclassifier_rbf115.fit(X_rbf115_train, Y_rbf115_train)
#y_pred_rbf115 = svclassifier_rbf115.predict(X_rbf115_test)

# Parameters for classificator - polynomial kernel
degree = [1,2,3,4,5,6,7,8,9]
degree = np.array(degree)

# Tables to save accuracy - polynomial kernel
Accuracy_CV_poly = np.zeros((10,1))
Accuracy_poly = np.zeros(len(degree))

# SVM - polynomial kernel
for i in range(0, len(degree)): # degree
    for k in range(1, 10): # Crossvalidation
        # Split data into test and train sets
        X_train_poly, X_test_poly, Y_train_poly, Y_test_poly = train_test_split(X, Y, test_size = 0.10) # 10 times cs so test_size is 10% of data set 
            
        # Standarization
        sc = StandardScaler()
        sc.fit(X_train_poly)
        X_train_poly = sc.transform(X_train_poly)
        X_test_poly = sc.transform(X_test_poly)
        X_train_poly = pd.DataFrame(X_train_poly)
        X_test_poly = pd.DataFrame(X_test_poly)
            
        # Model
        svclassifier = SVC(kernel='poly', degree = degree[i])
        svclassifier.fit(X_train_poly, Y_train_poly)
        
        y_pred_poly = svclassifier.predict(X_test_poly)
            
        # Accuracy - how many values from y_pred are equal to Y_test
        Accuracy_CV_poly[k] = sum(y_pred_poly == Y_test_poly)/len(Y_test_poly)

    Accuracy_poly[i] = np.mean(Accuracy_CV_poly) # rows - gamma, columns - C
    
Accuracy_poly # display accuracy table

# polynomial kernel accuracy plot
# axis X - degree
plt.plot(Accuracy_poly)
plt.xlabel('degree')
plt.ylabel('Accuracy')
#plt.xticks(np.arange(len(degree)), degree) # correct axis X ticks

