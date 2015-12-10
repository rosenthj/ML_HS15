'''
Created on Nov 14, 2015

@author: jonathan
'''

import numpy as np
import csv
import h5py
from numpy import hstack
from scipy.stats.stats import pearsonr
from sklearn.preprocessing.data import MinMaxScaler

#get labeled and unlabeled are legacy from a semisupervised problem.
def get_unlabeled(x, y):
    return x[y[:]==-1], y[y[:]==-1]

def get_labeled(x, y):
    return x[y[:]!=-1], y[y[:]!=-1]    

def _read_csv_features_labels(inpath):
    X = []
    Y = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            X.append([float(x) for x in row[1:-1]])
            Y.append([int(float(row[-1]))])
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    return (X, Y[:,0])#Return (features, labels)


def _read_csv_features(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            X.append([float(x) for x in row[1:]])
    return np.atleast_2d(X)

def _read_csv_labels(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            X.append([int(x) for x in row])
    return np.atleast_1d(np.concatenate(X))

#For h5 data format
def _read_h5data(inpath):
    return h5py.File(inpath, "r")["data"][...]

def _read_h5labels(inpath):
    f = h5py.File(inpath, "r")
    return np.squeeze(np.asarray(f["label"]))

def read_features_labels(inpath, fmt="csv"):
    if fmt=="csv": return _read_csv_features_labels(inpath)
    if fmt=="h5": return _read_h5data(inpath), _read_h5labels(inpath)
    
    
def read_labels(inpath, fmt="csv"):
    if fmt=="csv": return _read_csv_labels(inpath)
    if fmt=="h5": return _read_h5labels(inpath)
    
def read_features(inpath, fmt="csv"):
    if fmt=="csv": return _read_csv_features(inpath)
    if fmt=="h5": return _read_h5data(inpath)
    
def get_ids(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            X.append(int(row[0]))
    return X
    
def load_previous_predictions(dataset_name, predictions):
    train_sets = []
    test_sets = []
    for solution in predictions:
        X = []
        inpath = dataset_name + '_' + solution + '_train.csv'
        with open(inpath, 'r') as fin:
            reader = csv.reader(fin, delimiter=',')
            for row in reader:
                X.append([float(x) for x in row])
        train_sets.append(X)
    for solution in predictions:
        X = []
        inpath = dataset_name + '_' + solution + '_validate_and_test.csv'
        with open(inpath, 'r') as fin:
            reader = csv.reader(fin, delimiter=',')
            for row in reader:
                X.append([float(x) for x in row])
        test_sets.append(X)
    return hstack(train_sets), hstack(test_sets)

def pearson(A, B, scale=True):
    correlation = 0
    if scale:
        scaler = MinMaxScaler()
        A = scaler.fit_transform(A)
        B = scaler.fit_transform(B)
    for i in range(A.shape[1]):
        correlation = correlation + pearsonr(A[:,i],B[:,i])[0]
    return correlation / A.shape[1]

def pearson_datapoint(dataset_name, solution_a, solution_b):
    A = load_previous_predictions(dataset_name, [solution_a])[0]
    B = load_previous_predictions(dataset_name, [solution_b])[0]
    return pearson(A,B)

def pearson_data(dataset_name, solutions):
    for i in range(len(solutions)-1):
        for j in range(i+1,len(solutions)):
            print (solutions[i],solutions[j],pearson_datapoint(dataset_name, solutions[i], solutions[j]))
    