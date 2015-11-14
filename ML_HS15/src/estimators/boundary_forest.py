'''
Created on Aug 23, 2015

@author: jonathan
'''

import numpy as np

def _most_common_element(element_list):
    return max(set(element_list), key=element_list.count)

_weights = None

def _getDistance(x1, x2):
    x = np.subtract(x1, x2)
    return np.dot(x,x)
    #return np.dot(x * _weights, x)

class _Node:
    
    def __init__(self, position, value):
        self.position = position
        self.value = value
        self.neighbors = []
    
class _BoundaryTree:
    
    def __init__(self):
        self.rootNode = None
        self.nodeCount = 0
        
    def get_nearest_neighbor(self, x):
        if self.rootNode == None: return None
        nearest = self.rootNode
        minDis = _getDistance(nearest.position, x)
        while nearest.neighbors:
            closest = nearest
            for neighbor in nearest.neighbors:
                dis = _getDistance(neighbor.position, x)
                if dis < minDis:
                    closest = neighbor
                    minDis = dis
            if closest == nearest: break
            nearest = closest
        return nearest
    
    def fit(self, X, y):
        self.rootNode = _Node(X[0], y[0]); self.nodeCount = 1
        self.fit_live(X, y)
        
    def fit_live(self, X, y):
        if self.nodeCount == 0:
            self.fit(X, y)
            return
        numRows = X.shape[0]
        for row in range(numRows):
            nearestNeighbor = self.get_nearest_neighbor(X[row])
            if nearestNeighbor.value != y[row]:
                nearestNeighbor.neighbors.append(_Node(X[row], y[row]))
                self.nodeCount += 1
    
    def transform(self, X):
        res = []
        for row in X:
            res.append(self.get_nearest_neighbor(row).value)
        return np.asarray(res)
    
def _mse(pred, true):
    x = np.subtract(pred, true)
    return np.mean(x * x)
    
def set_weights(X,y, iterations = 100):
    global _weights
    _weights = np.ones(X.shape[1])
    learning_rate = 1.05
    max_index = 4*(X.shape[0])/5
    trainX = X[0:max_index, :]; trainY = y[0:max_index]
    testX = X[max_index: , :]; testY = y[max_index:]
    current = _BoundaryTree()
    current.fit(trainX, trainY)
    current_error = _mse(current.transform(testX), testY)
    old_error = -1
    for i in range(0,iterations):
        if (i % 10 == 0):
            print 'current BT error: ', current_error
        if (old_error == current_error):
            learning_rate = (learning_rate + 1)/2
        old_error = current_error
        for w in range(0, X.shape[1]):
            _min = _BoundaryTree()
            _max = _BoundaryTree()
            _weights[w] = _weights[w]/learning_rate
            _min.fit(trainX, trainY)
            min_error = _mse(min.transform(testX), testY)
            _weights[w] = _weights[w] * learning_rate * learning_rate
            _max.fit(trainX, trainY)
            max_error = _mse(max.transform(testX), testY)
            if (min_error < max_error and min_error < current_error):
                current_error = min_error
                current = _min
                _weights[w] = _weights[w] / learning_rate / learning_rate
            elif (max_error < current_error and max_error < min_error):
                current_error = max_error
                current = _max
            else:
                _weights[w] = _weights[w] / learning_rate             
            

            
            
        

class BoundaryForestClassifier:
    
    def __init__(self, num_trees=16):
        self.num_trees = num_trees
        self.trees = []
        for _ in range(num_trees):
            self.trees.append(_BoundaryTree())
        
    def get_params(self, deep=True):
        return {'num_trees': self.num_trees}
    
    def set_params(self, **params):
        if not params: return self
        self.num_trees = params['num_trees']
        self.trees = []
        for _ in range(self.num_trees):
            self.trees.append(_BoundaryTree())
    
    def fit(self, X, y):
        #set_weights(X, y)
        step_size = X.shape[0] / self.num_trees
        for tree in range(self.num_trees):
            self.trees[tree].fit(X[(tree*step_size):], y[(tree*step_size):])
            self.trees[tree].fit_live(X[0:(tree*step_size)], y[0:(tree*step_size)])
            #print 'tree node count: ', self.trees[tree].nodeCount
            
    def fit_live(self, X, y):
        step_size = X.shape[0] / self.num_trees
        for tree in range(self.num_trees):
            self.trees[tree].fit_live(X[(tree*step_size):], y[(tree*step_size):])
            self.trees[tree].fit_live(X[0:(tree*step_size)], y[0:(tree*step_size)])
            #print 'tree node count: ', self.trees[tree].nodeCount
                
    def predict(self, X):
        y = []
        for row in X:
            tree_predictions = []
            for tree in self.trees:
                tree_predictions.append(tree.get_nearest_neighbor(row).value)
            y.append(_most_common_element(tree_predictions))
        return np.asarray(y)
    
class BoundaryForestRegressor(BoundaryForestClassifier):
    
    def predict(self, X):
        y = []
        for row in X:
            tree_predictions = 0
            for tree in self.trees:
                tree_predictions = tree_predictions + tree.get_nearest_neighbor(row).value
            y.append(tree_predictions / self.num_trees)
        return np.asarray(y)