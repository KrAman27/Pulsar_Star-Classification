# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:42:25 2020

@author: aman kumar
"""

"""Assignment-6
NOVICE/MEDIAN
Pulsar Classifier:
Dataset Description:
Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They
are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of
matter .
Each pulsar produces a slightly different emission pattern, which varies slightly with each rotation .
Thus a potential signal detection known as a 'candidate', is averaged over many rotations of the
pulsar, as determined by the length of an observation. In the absence of additional info, each
candidate could potentially describe a real pulsar. However in practice almost all detections are
caused by radio frequency interference (RFI) and noise, making legitimate signals hard to find.
Machine learning tools are now being used to automatically label pulsar candidates to facilitate rapid
analysis. Classification systems in particular are being widely adopted,
which treat the candidate data sets as binary classification problems. Here the legitimate pulsar
examples are a minority positive class, and spurious examples the majority negative class.
The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639 real
pulsar examples. These examples have all been checked by human annotators.
Each row lists the variables first, and the class label is the final entry. The class labels used are 0
(negative) and 1 (positive).
Each candidate is described by 8 continuous variables, and a single class variable. The first four are
simple statistics obtained from the integrated pulse profile (folded profile). This is an array of
continuous variables that describe a longitude-resolved version of the signal that has been averaged
in both time and frequency . The remaining four variables are similarly obtained from the DM-SNR
curve . These are summarised below:
1. Mean of the integrated profile.
2. Standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. Skewness of the integrated profile.
5. Mean of the DM-SNR curve.
6. Standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. Skewness of the DM-SNR curve.
9. Class
TASK : Build classifier models and ensemble models(ONLY for MEDIAN) to train on the given
dataset."""


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('pulsar_stars.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,[-1]].values

#training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#Fitting K-NN classifier to the training set  
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(X_train, y_train)

#predicting the test results
y_pred = classifier.predict(X_test)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

