'''
Created on Oct 7, 2015

@author: jonathan
'''

import numpy as np

class FeatureMultiplier(object):
    '''
    classdocs
    '''

    def __init__(self, left=0, right=0):
        self.left = left
        self.right = right
    
    def get_params(self, deep=True):
        return {'left': self.left, 'right': self.right}
    
    def set_params(self, **params):
        if not params: return self
        if 'left' in params:
            self.left = params['left']
        if 'right' in params:
            self.right = params['right']

    
    def fit(self, X, y):
        return
    
    def fit_transform(self, X, y=None):            
        return self.transform(X)
    
    def transform(self, X, y=None):
        A = np.transpose(np.atleast_2d(X[:,self.left] * X[:,self.right]))
        return np.hstack((X,A))
    
class FeatureLogarithmic(object):
    '''
    classdocs
    '''
    
    def __init__(self, feature_id=0):
        print ('initialising class')
        self.feature_id = feature_id
    
    def get_params(self, deep=True):
        return {'feature_id': self.feature_id}
    
    def set_params(self, **params):
        if not params: return self
        self.feature_id = params['feature_id']

    
    def fit(self, X, y):
        return
    
    def fit_transform(self, X, y=None):            
        return self.transform(X)
    
    def transform(self, X, y=None):
        if (self.feature_id != -1):
            if (isinstance(self.feature_id, list)):
                return np.hstack((X,np.atleast_2d(np.log2(X[:,self.feature_id]))))
            return np.hstack((X,np.transpose(np.atleast_2d(np.log2(X[:,self.feature_id])))))
        return X