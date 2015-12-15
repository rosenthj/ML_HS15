'''
Created on Nov 14, 2015

@author: jonathan
'''

from math import sqrt
from numpy import hstack, vstack, savetxt
from sklearn.decomposition.kernel_pca import KernelPCA
from sklearn.decomposition.sparse_pca import SparsePCA
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.preprocessing.data import MinMaxScaler

from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.cross_validation import ShuffleSplit
from sklearn.decomposition.nmf import NMF
from sklearn.decomposition.pca import PCA, RandomizedPCA
from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesClassifier, \
    RandomForestClassifier
from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesClassifier, \
    RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.feature_selection.univariate_selection import SelectPercentile, \
    f_classif, SelectKBest
from sklearn.feature_selection.univariate_selection import SelectPercentile, \
    f_classif, SelectKBest
from sklearn.feature_selection.variance_threshold import VarianceThreshold
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics.metrics import accuracy_score
from sklearn.metrics.metrics import mean_squared_error
from sklearn.metrics.scorer import make_scorer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.pipeline import Pipeline
from sknn.mlp import Classifier
from sknn.mlp import Classifier
from sknn.nn import Layer
from sknn.nn import Layer

from core.data_handling import read_features_labels, read_features
from core.data_handling import read_features_labels, read_features, \
    load_previous_predictions, pearson_data, get_ids
from estimators.boundary_forest import BoundaryForestClassifier
import numpy as np
import sklearn.svm as svm
from transformers import feature_selection as fs
from transformers.feature_selection import FeatureSelector


if __name__ == '__main__':
    pass

solution_name = 'nn'

dataset_name = 'Cancer'
datasetTrain = dataset_name + '_train.csv'
datasetTest = dataset_name + '_validate_and_test.csv'
train_solution_name = dataset_name + '_' + solution_name + '_train.csv'
test_solution_name = dataset_name + '_' + solution_name + '_validate_and_test.csv'

X, y = read_features_labels(datasetTrain)
test_X = read_features(datasetTest)
folds = KFold(n=X.shape[0], n_folds=20)

support_vector_classifier = svm.SVC(probability=True, verbose=True)
linear_support_vector_classifier = svm.LinearSVC(dual=False)
nearest_neighbor_classifier = KNeighborsClassifier(n_neighbors=7)
extra_trees_classifier = ExtraTreesClassifier(n_estimators=128)
bagging_classifier = BaggingClassifier(base_estimator=GradientBoostingClassifier(n_estimators=200,max_features=4), max_features=0.5,n_jobs=2, verbose=1)
gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=30, max_features=4, learning_rate=0.1, verbose=0)
random_forest_classifier = RandomForestClassifier(n_estimators=128)
logistic_regression = LogisticRegression(C=80)
ridge_classifier = RidgeClassifier(alpha=0.1, solver='svd')
bayes = MultinomialNB()
sgd = SGDClassifier()
boundary_forest = BoundaryForestClassifier(num_trees=4)


base_estimator = nearest_neighbor_classifier
classifier = Pipeline(steps=[('prep', MinMaxScaler()),('estimator', base_estimator)])


predictions = []
for train_index, test_index in folds:
    classifier.fit(X[train_index], y[train_index])
    predictions.append(classifier.predict_proba(X[test_index]))
    
#use h-stack if using predict, use vstack for predict proba here. In the future this can be improved.
predictions = vstack(predictions)
print ('finished cross validate predictions')

savetxt(train_solution_name, predictions, delimiter=',', fmt='%f')

classifier.fit(X,y)
savetxt(test_solution_name, classifier.predict_proba(test_X), delimiter=',', fmt='%f')

print ('finished')