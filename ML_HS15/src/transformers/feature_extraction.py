'''
Created on Oct 7, 2015

@author: jonathan
'''

import numpy as np
import sklearn.preprocessing as skprep

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
        if self.left == -1: return X            
        return self.transform(X)
    
    def transform(self, X, y=None):
        if self.left == -1: return X
        A = np.transpose(np.atleast_2d(X[:,self.left] * X[:,self.right]))
        return np.hstack((X,A))
    
    
class FeatureLogarithmic(object):
    '''
    classdocs
    '''
    
    def __init__(self, feature_id=0):
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
    
class FeatureCategorizer(object):
    '''
    classdocs
    '''
    
    def __init__(self, feature_id=0):
        self.feature_id = feature_id
        self.le = skprep.LabelEncoder()
        self.ohe = skprep.OneHotEncoder(sparse=False)
    
    def get_params(self, deep=True):
        return {'feature_id': self.feature_id}
    
    def set_params(self, **params):
        if not params: return self
        self.feature_id = params['feature_id']

    
    def fit(self, X, y=None):
        if self.feature_id == -1: return
        x = self.le.fit_transform(X[:,self.feature_id])
        x = np.transpose(np.atleast_2d(x))
        self.ohe.fit(x)
        return
    
    def fit_transform(self, X, y=None):
        if self.feature_id == -1: return X         
        self.fit(X)
        return self.transform(X)
    
    def transform(self, X, y=None):
        if (self.feature_id != -1):
            x = self.ohe.transform(np.transpose(np.atleast_2d(self.le.transform(X[:,self.feature_id]))))
            x = np.hstack((X,x))
            return x
        return X
    
    
class MultiFeatureCategorizer(object):
    '''
    classdocs
    '''
    
    def __init__(self, feature_id=0):
        self.feature_id = feature_id
        self.transformers = []
    
    def get_params(self, deep=True):
        return {'feature_id': self.feature_id}
    
    def set_params(self, **params):
        if not params: return self
        self.feature_id = params['feature_id']

    
    def fit(self, X, y=None):
        if not isinstance(self.feature_id, np.ndarray) and self.feature_id == -1: return
        self.transformers = []
        if isinstance(self.feature_id, np.ndarray) or isinstance(self.feature_id, list):
            for w in self.feature_id:
                fe = FeatureCategorizer(feature_id=w)
                fe.fit(X)
                self.transformers.append(fe)
        else:
            fe = FeatureCategorizer(feature_id=self.feature_id)
            fe.fit(X)
            self.transformers.append(fe)
        return
    
    def fit_transform(self, X, y=None):
        if not isinstance(self.feature_id, np.ndarray) and self.feature_id == -1: return X         
        self.fit(X)
        return self.transform(X)
    
    def transform(self, X, y=None):
        if isinstance(self.feature_id, np.ndarray) or self.feature_id != -1:
            A = X
            for tran in self.transformers:
                A = tran.transform(A)
            return A
        return X
    