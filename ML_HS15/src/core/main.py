'''
Created on Aug 16, 2015

@author: jonathan
'''

from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.ensemble.forest import RandomForestClassifier, RandomForestRegressor
from math import sqrt

from estimators import boundary_forest as bt
from transformers import feature_selection as fs
from transformers import feature_extraction as fe
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn import multiclass as skmult

if __name__ == '__main__':
    pass

import sklearn.svm as svm
# Provides Matlab-style matrix operations
import numpy as np
# Provides Matlab-style plotting
import matplotlib.pylab as plt
# For reading and writing csv files
import csv
# For parsing date/time strings
import datetime as dtime
# For parsing .h5 file format
import h5py
# Contains linear models, e.g., linear regression, ridge regression, LASSO, etc.
import sklearn.linear_model as sklin
# Allows us to create custom scoring functions
import sklearn.metrics as skmet
# Provides train-test split, cross-validation, etc.
import sklearn.cross_validation as skcv
# Provides grid search functionality
import sklearn.grid_search as skgs
import sklearn.feature_extraction as skfe
# Provides feature selection functionality
import sklearn.feature_selection as skfs
# Provides access to ensemble based classification and regression.
import sklearn.ensemble as sken
import sklearn.tree as sktree
import sklearn.neighbors as nn
import sklearn.preprocessing as skprep
import sknn.mlp as skneural
import sklearn.cluster as cluster
import sklearn.pipeline as skpipe
import sklearn.decomposition as skdec
import sklearn.naive_bayes as skbayes
import sklearn.semi_supervised as sksemi
import sklearn.cross_validation as skcross

MonthsTable = [0,3,3,6,1,4,6,2,5,0,3,5]

datasetName = 'Sleep'
datasetFmt = '%i,%i'#Output file format assumes one row per test case. For categorical integer values use '%i' otherwise '%f'

datasetTrain = datasetName + '_train.csv'
datasetTest = datasetName + '_validate_and_test.csv'

#Returns day of week
def get_weekday(y, m, d):
    return np.mod(y-2000+m+d+((y-2000)/4) + 6, 7)

def f_log(x):
    return np.log2(np.abs(x)+0.01)

def create_vector(a):
    return [a]#, f_log(a), np.exp2(a)]

def complex_vec(a):
    return [a]


def create_complex_vec(vec):
    return np.concatenate([complex_vec(x) for x in vec])

def get_cat_vec(c, num_cat, x=1):
    A = [0] * num_cat
    A[c] = x
    return A

def get_features(geomVec, catVec):
    return np.concatenate([create_complex_vec([int(x) for x in geomVec]), [int(x) for x in catVec]])

def logscore(gtruth, pred):
    pred = np.clip(pred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + pred)
    return np.sqrt(np.mean(np.square(logdif)))

def elementsnotequal(x, y):
    if (x == y):
        return 0
    return 1

def labelLogScore(x, y):
    return -np.log(max(0.000001, y[x]))

def labelsnotequal(x, y):
    return map(elementsnotequal,x, y)

def multiLabelScore(gtruth, pred):
    return np.mean(map(labelsnotequal,gtruth, pred))

def singleLabelScore(gtruth, pred):
    return np.mean(map(elementsnotequal,gtruth, pred))

def singleLabelLogScore(gtruth, pred):
    return np.mean(map(labelLogScore,gtruth, pred))

def rootMeanSquaredError(gtruth, pred):
    squaredError = skmet.mean_squared_error(gtruth, pred)
    return np.sqrt(squaredError)

#get labeled and unlabeled are legacy from a semisupervised problem.
def get_unlabeled(x, y):
    return x[y[:]==-1], y[y[:]==-1]

def get_labeled(x, y):
    return x[y[:]!=-1], y[y[:]!=-1]


def read_csv_features_label(inpath):
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

def read_csv_features(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            X.append([float(x) for x in row[1:]])
    return np.atleast_2d(X)

def read_labels(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            X.append([int(x) for x in row])
    return np.atleast_1d(np.concatenate(X))

#For h5 data format
def read_h5data(inpath):
    return h5py.File(inpath, "r")["data"][...]

def read_h5labels(inpath):
    f = h5py.File(inpath, "r")
    return np.squeeze(np.asarray(f["label"]))

#normalizer = skprep.Normalizer()
print('loading training data')
#X = read_data('train.csv')
#print('first row: ', X[0])

#print('loading training labels')
#Y = read_labels('train_y.csv')
X, Y = read_csv_features_label(datasetTrain)
print('first row X: ', X[0], ' Y: ' , Y[0])

print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)

#print('normalizing!')
#X = normalizer.fit_transform(X)

# LABELING
# labelProp = sksemi.label_propagation.LabelSpreading(kernel='rbf', gamma=150, n_neighbors=3, alpha=0.15, max_iter=600000, tol=0.001)
# print('fitting label spreader')
# labelProp.fit(X, Y)
# print('predicting labels for Y')
# Y=labelProp.transduction_
# print('Shape of Y:', Y.shape)
# print('first row: ', Y[0])

# SCORER
scorer = skmet.make_scorer(score_func=singleLabelScore, greater_is_better=False)

# PREPROCESSING
# SCALING
minMaxScaler = skprep.MinMaxScaler(feature_range=(0.0,1.0))
#normalizer = skprep.Normalizer()
columnDeleter = fs.FeatureDeleter()

# FEATURE SELECTION
varianceThresholdSelector = skfs.VarianceThreshold(threshold=(0))
percentileSelector = skfs.SelectPercentile(score_func=skfs.f_classif, percentile=20)
kBestSelector = skfs.SelectKBest(skfs.f_classif, 1000)

# FEATURE EXTRACTION
#rbmPipe = skpipe.Pipeline(steps=[('scaling', minMaxScaler), ('rbm', rbm)])
nmf = skdec.NMF(n_components=150)
pca = skdec.PCA(n_components=80)
sparsePCA = skdec.SparsePCA(n_components=700, max_iter=3, verbose=2)
kernelPCA = skdec.KernelPCA(n_components=150)# Costs huge amounts of ram
randomizedPCA= skdec.RandomizedPCA(n_components=500)

# REGRESSORS
randomForestRegressor = sken.RandomForestRegressor(n_estimators=256)
gradientBoostingRegressor = sken.GradientBoostingRegressor(n_estimators=60)
supportVectorRegressor = svm.SVR()

# CLASSIFIERS
supportVectorClassifier = svm.SVC(probability=True, verbose=True)
linearSupportVectorClassifier = svm.LinearSVC(dual=False)
nearestNeighborClassifier = nn.KNeighborsClassifier()
extraTreesClassifier = sken.ExtraTreesClassifier(n_estimators=256)
baggingClassifier = sken.BaggingClassifier(base_estimator=sken.GradientBoostingClassifier(n_estimators=200,max_features=4), max_features=0.5,n_jobs=2, verbose=1)
gradientBoostingClassifier = sken.GradientBoostingClassifier(n_estimators=30, max_features=4, learning_rate=0.1, verbose=0)
randomForestClassifier = sken.RandomForestClassifier(n_estimators=2)
logisticClassifier = sklin.LogisticRegression(C=80)
ridgeClassifier = sklin.RidgeClassifier(alpha=0.1, solver='svd')
bayes = skbayes.MultinomialNB()
sgd = sklin.SGDClassifier(loss = 'log', penalty='elasticnet', shuffle=True, n_jobs=1)
sgdn = sklin.SGDClassifier(loss = 'hinge', penalty='l1', n_jobs=1)
boundary_forest = bt.BoundaryForestClassifier(num_trees=4)


# FEATURE UNION
featureUnion = skpipe.FeatureUnion(transformer_list=[('PCA', pca)])


# PIPE DEFINITION
classifier = skpipe.Pipeline(steps=[('scaler', MinMaxScaler()),('estimator', sken.RandomForestClassifier())])
print ('Successfully prepared classifier pipeline!')


#X = np.vstack((X,np.roll(X,1),np.roll(X,-1),np.roll(X,28),np.roll(X,-28)))
#Y = np.hstack((Y,Y,Y,Y,Y))
#print 'X and Y shapes ', X.shape, ' ', Y.shapee

RMSE = skmet.make_scorer(rootMeanSquaredError, greater_is_better = False)
categoricalAccuracyScorer = skmet.make_scorer(skmet.accuracy_score, greater_is_better = True)

# GRID DEFINITION
classifierSearcher = GridSearchCV(classifier, dict(estimator__n_estimators=[64,256]), cv=skcross.ShuffleSplit(2025,n_iter=100,test_size=0.2),scoring=categoricalAccuracyScorer, n_jobs=2, verbose=1)

print ('fitting classifier pipeline grid on training data subset for accuracy estimate')
classifierSearcher.fit(X, Y)
print ('best estimator by mean:', classifierSearcher.best_estimator_)
print ('best score:', classifierSearcher.best_score_)
classifier = classifierSearcher.best_estimator_

print ('Grid search results')
for values in classifierSearcher.grid_scores_:
    squaredSum = count = 0
    for result in values[2]:
        squaredSum += (values[1] - result) * (values[1] - result)
        count += 1
    stdDev = sqrt(squaredSum/count)
    print (values[0], "mean:", values[1], "std deviation:", stdDev, "m-s:", values[1]-stdDev)


print ('fitting classifier pipeline on training data')
classifier.fit(X, Y)

print ('loading test data')
testX = read_csv_features(datasetTest)
#testX = normalizer.transform(testX)
print ('Shape of testX:', testX.shape)

# print ('classifier pipe is predicting result of validation data')
# Ypred = classifier.predict_proba(valX)
# print('Ypred shape', Ypred.shape)
# print('predicted result validate.csv')
# np.savetxt('result_validate_quick.txt', Ypred, delimiter=',', fmt='%f')
# 
print ('classifier pipe is predicting result of test data')
Ypred = classifier.predict(testX)
print('Ypred shape', Ypred.shape)
print('predicted result', datasetName + '_result.csv')
IDs = np.arange(2026,2701)
np.savetxt(datasetName + '_result.csv', np.column_stack((IDs,Ypred)), delimiter=',', fmt=datasetFmt)
