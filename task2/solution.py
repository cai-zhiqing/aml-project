import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fft import fft
from statsmodels.tsa.stattools import pacf, acf

from tslearn.svm import TimeSeriesSVC
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score,f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE, ADASYN,RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier

import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from biosppy.signals import ecg

def writeToFile(results):
    print("Writing results to file")
    res_df = pd.read_csv('sample.csv')
    res_df['y'] = results
    res_df.to_csv('results.csv',index=False,header=True)

def open_data():
    x_train_df = pd.read_csv('X_train.csv').drop(labels='id',axis=1).to_numpy()
    x_test_df = pd.read_csv('X_test.csv').drop(labels='id',axis=1).to_numpy()
    y_train_df = pd.read_csv('y_train.csv').drop(labels='id',axis=1).to_numpy()
    return x_train_df, y_train_df, x_test_df

def pad_array(input_array, target_length):
    current_length = len(input_array)

    if current_length >= target_length:
        return input_array
    else:
        # Calculate the number of zeros to pad
        num_zeros = target_length - current_length
        # Use numpy.pad to add zeros to the end of the array
        padded_array = np.pad(input_array, (0, num_zeros), 'constant')
        return padded_array

def extract_features(X,sampling_rate=300):
    if np.isnan(X).any():
        first_nan_index = np.where(np.isnan(X))[0][0]
        X = X[:first_nan_index]
    mean = X.mean()
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(X, sampling_rate, show=False)
    peaks_values = filtered[rpeaks]
    mean = filtered.mean()
    fft_data = np.fft.fft(filtered)
    fft_abs = np.abs(fft_data)
    heart_rate = pad_array(heart_rate, 100)
    peaks_values = pad_array(peaks_values, 100)
    highest_freq_indices = np.argsort(fft_abs)[-100:]
    highest_freq_values = fft_abs[highest_freq_indices]
    autoco = acf(X, missing='drop', nlags=100)
    pautoco = pacf(X, nlags=100)
    features = np.hstack((mean, autoco, pautoco, heart_rate[:100], peaks_values[:100], highest_freq_values))
    return features

x_train, y_train, x_test = open_data()
print("Data loaded")
train = np.apply_along_axis(extract_features, arr=x_train, axis=1)
print("Extracted training features")
test = np.apply_along_axis(extract_features, arr=x_test, axis=1)
print("Extracted test features")

# Split the data into training and validation sets
train_data, validation_data, train_labels, validation_labels = train_test_split(train, y_train, test_size=0.2, random_state=42)

# Train the model on the training set
clf = GradientBoostingClassifier()
clf.fit(train_data, train_labels.ravel())
# Validate the model on the validation set
validation_preds = clf.predict(validation_data)

#Improvements: Possibly different classifiers, e.g. weighted classifiers or upsampling
accuracy = f1_score(validation_labels, validation_preds, average='micro')
print(f"Validation Accuracy: {accuracy}")

# Train the final model on the entire training set
clf.fit(train, y_train.ravel())

# Predict on the test set
test_preds = clf.predict(test)

# Write the predictions to a file
writeToFile(test_preds)
