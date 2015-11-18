'''
Created on Oct 6, 2015

@author: jonathan
'''

import numpy as np

class FeatureDeleter:
    '''
    classdocs
    '''
    def __init__(self, feature_id=-1):
        self.feature_id = feature_id
        
    def get_params(self, deep=True):
        return {'feature_id': self.feature_id}
    
    def set_params(self, **params):
        if not params: return self
        self.feature_id = params['feature_id']
    
    def fit(self, X, y):
        return
    
    def fit_transform(self, X, y=None):            
        if (self.feature_id != -1):
            return np.delete(X, self.feature_id, 1)
        return X
    
    def transform(self, X, y=None):
        if (self.feature_id != -1):
            return np.delete(X, self.feature_id, 1)
        return X