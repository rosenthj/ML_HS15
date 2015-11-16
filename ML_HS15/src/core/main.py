'''
Created on Aug 16, 2015

@author: jonathan
'''

from math import sqrt
from numpy import hstack
from sklearn.cross_validation import ShuffleSplit
from sklearn.decomposition.kernel_pca import KernelPCA
from sklearn.decomposition.nmf import NMF
from sklearn.decomposition.pca import PCA, RandomizedPCA
from sklearn.decomposition.sparse_pca import SparsePCA
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesClassifier, \
    RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.feature_selection.univariate_selection import SelectPercentile, \
    f_classif, SelectKBest
from sklearn.feature_selection.variance_threshold import VarianceThreshold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics.regression import mean_squared_error
from sklearn.metrics.scorer import make_scorer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing.data import MinMaxScaler

from core.data_handling import read_features_labels, read_features, \
    load_previous_predictions, pearson_data
from estimators.boundary_forest import BoundaryForestClassifier
import numpy as np
import sklearn.svm as svm
from transformers import feature_selection as fs


if __name__ == '__main__':
    pass


datasetName = 'Sleep'
datasetFmt = '%i,%i'#Output file format assumes one row per test case. For categorical integer values use '%i' otherwise '%f'
datasetTrain = datasetName + '_train.csv'
datasetTest = datasetName + '_validate_and_test.csv'

use_model_stacking = True
model_stacking_models = ['knn','svm','logreg','gbm','rf']

MonthsTable = [0,3,3,6,1,4,6,2,5,0,3,5]

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
    squaredError = mean_squared_error(gtruth, pred)
    return np.sqrt(squaredError)

#normalizer = skprep.Normalizer()
print('loading training data')
#X = read_data('train.csv')
#print('first row: ', X[0])

#print('loading training labels')
#Y = read_labels('train_y.csv')
X, Y = read_features_labels(datasetTrain)
if use_model_stacking: X = hstack([X,load_previous_predictions(datasetName, model_stacking_models)[0]])
print('first row X: ', X[0], ' Y: ' , Y[0])

print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)
pearson_data(datasetName, model_stacking_models)

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
scorer = make_scorer(score_func=singleLabelScore, greater_is_better=False)

# PREPROCESSING
# SCALING
minMaxScaler = MinMaxScaler(feature_range=(0.0,1.0))
#normalizer = skprep.Normalizer()
columnDeleter = fs.FeatureDeleter()

# FEATURE SELECTION
varianceThresholdSelector = VarianceThreshold(threshold=(0))
percentileSelector = SelectPercentile(score_func=f_classif, percentile=20)
kBestSelector = SelectKBest(f_classif, 1000)

# FEATURE EXTRACTION
#rbmPipe = skpipe.Pipeline(steps=[('scaling', minMaxScaler), ('rbm', rbm)])
nmf = NMF(n_components=150)
pca = PCA(n_components=80)
sparse_pca = SparsePCA(n_components=700, max_iter=3, verbose=2)
kernel_pca = KernelPCA(n_components=150)# Costs huge amounts of ram
randomized_pca= RandomizedPCA(n_components=500)

# REGRESSORS
random_forest_regressor = RandomForestRegressor(n_estimators=256)
gradient_boosting_regressor = GradientBoostingRegressor(n_estimators=60)
support_vector_regressor = svm.SVR()

# CLASSIFIERS
support_vector_classifier = svm.SVC(probability=True, verbose=True)
linear_support_vector_classifier = svm.LinearSVC(dual=False)
nearest_neighbor_classifier = KNeighborsClassifier()
extra_trees_classifier = ExtraTreesClassifier(n_estimators=256)
bagging_classifier = BaggingClassifier(base_estimator=GradientBoostingClassifier(n_estimators=200,max_features=4), max_features=0.5,n_jobs=2, verbose=1)
gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=30, max_features=4, learning_rate=0.1, verbose=0)
random_forest_classifier = RandomForestClassifier(n_estimators=2)
logistic_regression = LogisticRegression(C=80)
ridge_classifier = RidgeClassifier(alpha=0.1, solver='svd')
bayes = MultinomialNB()
sgd = SGDClassifier()
boundary_forest = BoundaryForestClassifier(num_trees=4)


# FEATURE UNION
feature_union = FeatureUnion(transformer_list=[('PCA', pca)])


# PIPE DEFINITION
classifier = Pipeline(steps=[('scaler', MinMaxScaler()),('estimator', LogisticRegression(penalty='l1'))])
print ('Successfully prepared classifier pipeline!')


#X = np.vstack((X,np.roll(X,1),np.roll(X,-1),np.roll(X,28),np.roll(X,-28)))
#Y = np.hstack((Y,Y,Y,Y,Y))
#print 'X and Y shapes ', X.shape, ' ', Y.shapee

RMSE = make_scorer(rootMeanSquaredError, greater_is_better = False)
categorical_accuracy_scorer = make_scorer(accuracy_score, greater_is_better = True)

# GRID DEFINITION
classifier_searcher = GridSearchCV(classifier, dict(estimator__C=[0.09]),cv=ShuffleSplit(2025,n_iter=1000,test_size=0.2),scoring=categorical_accuracy_scorer, n_jobs=2, verbose=1)

#({'estimator__C': 0.07}, 'mean:', 0.91472530864197532, 'std deviation:', 0.012816619187771739, 'm-s:', 0.90190868945420355)

print ('fitting classifier pipeline grid on training data subset for accuracy estimate')
classifier_searcher.fit(X, Y)
print ('best estimator by mean:', classifier_searcher.best_estimator_)
print ('best score:', classifier_searcher.best_score_)
classifier = classifier_searcher.best_estimator_

print ('Grid search results')
for values in classifier_searcher.grid_scores_:
    squaredSum = count = 0
    for result in values[2]:
        squaredSum += (values[1] - result) * (values[1] - result)
        count += 1
    stdDev = sqrt(squaredSum/count)
    print (values[0], "mean:", values[1], "std deviation:", stdDev, "m-s:", values[1]-stdDev)


print ('fitting classifier pipeline on training data')
classifier.fit(X, Y)

print ('loading test data')
testX = read_features(datasetTest)
if use_model_stacking: testX = hstack([testX,load_previous_predictions(datasetName, model_stacking_models)[1]])
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
