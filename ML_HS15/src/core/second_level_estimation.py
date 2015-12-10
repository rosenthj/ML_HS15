'''
Created on Nov 14, 2015

@author: jonathan
'''
from numpy import vstack, savetxt
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import MinMaxScaler
from sknn.nn import Layer
from sknn.mlp import Classifier

from core.data_handling import read_features_labels, read_features


if __name__ == '__main__':
    pass

solution_name = 'neural'

dataset_name = 'Cancer'
datasetTrain = dataset_name + '_train.csv'
datasetTest = dataset_name + '_validate_and_test.csv'
train_solution_name = dataset_name + '_' + solution_name + '_train.csv'
test_solution_name = dataset_name + '_' + solution_name + '_validate_and_test.csv'

X, y = read_features_labels(datasetTrain)
test_X = read_features(datasetTest)
folds = KFold(n=X.shape[0], n_folds=20)

base_estimator = Classifier(layers=[Layer("Rectifier",name="hidden0",units=10),Layer("Softmax")],n_iter=20,learning_rate=0.2 ,learning_rule="adagrad")
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