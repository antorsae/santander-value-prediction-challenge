#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb
#import lightgbm as lgb
from scipy.stats import skew, kurtosis

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone, is_classifier
from sklearn.model_selection._split import check_cv
from sklearn.model_selection import GridSearchCV


from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, LatentDirichletAllocation, SparsePCA, TruncatedSVD, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest

def score_features(X, y, estimator=None):
    return clone(estimator).fit(X, y).feature_importances_

class UniqueTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, axis=1, accept_sparse=False):
        if axis == 0:
            raise NotImplementedError('axis is 0! Not implemented!')
        if accept_sparse:
            raise NotImplementedError('accept_sparse is True! Not implemented!')
        self.axis = axis
        self.accept_sparse = accept_sparse
        
    def fit(self, X, y=None):
        _, self.unique_indices_ = np.unique(X, axis=self.axis, return_index=True)
        return self
    
    def transform(self, X, y=None):
        return X[:, self.unique_indices_]


class StatsTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, stat_funs=None, verbose=0, n_jobs=-1, pre_dispatch='2*n_jobs'):
        self.stat_funs = stat_funs
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
    
    def _get_stats(self, row):
        stats = []
        for fun in self.stat_funs:
            stats.append(fun(row))
        return stats

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        parallel = Parallel(
            n_jobs=self.n_jobs,
            pre_dispatch=self.pre_dispatch,
            verbose=self.verbose
        )
        stats_list = parallel(delayed(self._get_stats)(X[i_smpl, :]) for i_smpl in range(len(X)))
        return np.array(stats_list)



class ClassifierTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator=None, n_classes=2, cv=3):
        self.estimator = estimator
        self.n_classes = n_classes
        self.cv = cv
    
    def _get_labels(self, y):
        y_labels = np.zeros(len(y))
        y_us = np.sort(np.unique(y))
        step = int(len(y_us) / self.n_classes)
        
        for i_class in range(self.n_classes):
            if i_class + 1 == self.n_classes:
                y_labels[y >= y_us[i_class * step]] = i_class
            else:
                y_labels[
                    np.logical_and(
                        y >= y_us[i_class * step],
                        y < y_us[(i_class + 1) * step]
                    )
                ] = i_class
        return y_labels
        
    def fit(self, X, y):
        y_labels = self._get_labels(y)
        cv = check_cv(self.cv, y_labels, classifier=is_classifier(self.estimator))
        self.estimators_ = []
        
        for train, _ in cv.split(X, y_labels):
            self.estimators_.append(
                clone(self.estimator).fit(X[train], y_labels[train])
            )
        return self
    
    def transform(self, X, y=None):
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        
        X_prob = np.zeros((X.shape[0], self.n_classes))
        X_pred = np.zeros(X.shape[0])
        
        for estimator, (_, test) in zip(self.estimators_, cv.split(X)):
            X_prob[test] = estimator.predict_proba(X[test])
            X_pred[test] = estimator.predict(X[test])
        return np.hstack([X_prob, np.array([X_pred]).T])

class XGBRegressorCV(BaseEstimator, RegressorMixin):
    
    def __init__(self, regressor=xgb.XGBRegressor, xgb_params=None, fit_params=None, cv=3):
        self.regressor  = regressor
        self.xgb_params = xgb_params
        self.fit_params = fit_params
        self.cv = cv
    
    @property
    def feature_importances_(self):
        feature_importances = []
        for estimator in self.estimators_:
            feature_importances.append(
                estimator.feature_importances_
            )
        return np.mean(feature_importances, axis=0)
    
    @property
    def evals_result_(self):
        evals_result = []
        for estimator in self.estimators_:
            evals_result.append(
                estimator.evals_result_
            )
        return np.array(evals_result)
    
    @property
    def best_scores_(self):
        best_scores = []
        for estimator in self.estimators_:
            best_scores.append(
                estimator.best_score
            )
        return np.array(best_scores)
    
    @property
    def cv_scores_(self):
        return self.best_scores_ 
    
    @property
    def cv_score_(self):
        return np.mean(self.best_scores_)
    
    @property
    def best_iterations_(self):
        best_iterations = []
        for estimator in self.estimators_:
            best_iterations.append(
                estimator.best_iteration
            )
        return np.array(best_iterations)
    
    @property
    def best_iteration_(self):
        return np.round(np.mean(self.best_iterations_))
    
    def fit(self, X, y, **fit_params):
        cv = check_cv(self.cv, y, classifier=False)
        self.estimators_ = []
        
        for train, valid in cv.split(X, y):
            self.estimators_.append(
                self.regressor(**self.xgb_params).fit(
                    X[train], y[train],
                    eval_set=[(X[valid], y[valid])],
                    **self.fit_params
                )
            )
            #print(self.estimators_)

        return self
    
    def predict(self, X):
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred, axis=0)

class _StatFunAdaptor:
    
    def __init__(self, stat_fun, *funs, **stat_fun_kwargs):
        self.stat_fun = stat_fun
        self.funs = funs
        self.stat_fun_kwargs = stat_fun_kwargs

    def __call__(self, x):
        x = x[x != 0]
        for fun in self.funs:
            x = fun(x)
        if x.size == 0:
            return -99999
        return self.stat_fun(x, **self.stat_fun_kwargs)


def diff2(x):
    return np.diff(x, n=2)


def get_stat_funs():
    """
    Previous version uses lambdas.
    """
    stat_funs = []
    
    stats = [len, np.mean, np.min, np.max, np.median, np.std, skew, kurtosis] + 19 * [np.percentile]
    stats_kwargs = [{} for i in range(8)] + [{'q': i} for i in np.linspace(0.05, 0.95, 19)]

    for stat, stat_kwargs in zip(stats, stats_kwargs):
        stat_funs.append(_StatFunAdaptor(stat, **stat_kwargs))
        stat_funs.append(_StatFunAdaptor(stat, np.diff, **stat_kwargs))
        stat_funs.append(_StatFunAdaptor(stat, diff2, **stat_kwargs))
        stat_funs.append(_StatFunAdaptor(stat, np.unique, **stat_kwargs))
        stat_funs.append(_StatFunAdaptor(stat, np.unique, np.diff, **stat_kwargs))
        stat_funs.append(_StatFunAdaptor(stat, np.unique, diff2, **stat_kwargs))
    return stat_funs


def get_rfc(jj):
    return RandomForestClassifier(
        n_estimators=1000,
        max_features=0.5,
        max_depth=None,
        max_leaf_nodes=270,
        min_impurity_decrease=0.0001,
        random_state=123,
        n_jobs=-1
    )


def get_data():
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    y_train_log = np.log1p(train['target'])
    id_test = test['ID']
    del test['ID']
    del train['ID']
    del train['target']
    #f = 1e4
    return train.values, y_train_log.values, test.values, id_test.values
    #return train.values/f, y_train_log.values, test.values/f, id_test.values
    #return np.log1p(train.values), y_train_log.values, np.log1p(test.values), id_test.values

def main():
    
    xgb_params = {
        #'tree_method' : 'gpu_hist',
        'n_estimators': 1500,
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.02,
        'max_depth': 10, #22,
        'min_child_weight': 57,
        'gamma' : 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456
    }

    lgb_params = {
        'n_estimators': 1000,
        'boosting_type': 'gbdt',
        'learning_rate': 0.007,
        'num_leaves' : 4000,
        'max_depth': 22,
        'n_jobs': -1,
        'random_state': 456
    }
    
    fit_params = {
        'early_stopping_rounds': 50,
        'eval_metric': 'rmse',
        'verbose': False,
    }

    xgb_cv = XGBRegressorCV(
                    regressor  = xgb.XGBRegressor,
                    xgb_params = xgb_params,
                    fit_params = fit_params,
                    cv=10,
                )
    
    pipe = Pipeline(
        [
            #('vt', VarianceThreshold(threshold=0.0)),
            #('ut', UniqueTransformer()),
            ('fu', FeatureUnion(
                    [
                        ('pca1', PCA(n_components=100)),
                        ('spca1', SparsePCA(n_components=100)),
                        ('lda',  LatentDirichletAllocation(n_components=100)),
                        ('fa1', FactorAnalysis(n_components=100)),
                        ('tsvd', TruncatedSVD(n_components=100)),
                        ('ct-2', ClassifierTransformer(get_rfc(50), n_classes=2, cv=5)),
                        ('ct-3', ClassifierTransformer(get_rfc(75), n_classes=3, cv=5)),
                        ('ct-4', ClassifierTransformer(get_rfc(100), n_classes=4, cv=5)),
                        ('ct-5', ClassifierTransformer(get_rfc(125), n_classes=5, cv=5)),
                        ('ct-10', ClassifierTransformer(get_rfc(150), n_classes=10, cv=5)),
                        ('ct-20', ClassifierTransformer(get_rfc(175), n_classes=20, cv=5)),
                        ('ct-100', ClassifierTransformer(get_rfc(200), n_classes=100, cv=5)),
                        ('st', StatsTransformer(stat_funs=get_stat_funs(), verbose=2))
                    ]
                )
            ),
            #('skb', SelectKBest(score_func=lambda X, y: score_features(X, y, estimator=xgb_cv), k=120)),
            ('xgb', xgb_cv),
        ],
        #memory = '.pipeline'
    )
    
    X_train, y_train_log, X_test, id_test = get_data()
    
    param_grid = [dict(xgb__max_depth = [11,22,44])]

    #pipe = GridSearchCV(_pipe, param_grid=param_grid)

    pipe.fit(X_train, y_train_log)
    #print(pipe.named_steps['xgb-cv'])
    print(pipe.named_steps['xgb'].cv_scores_)
    cv_score = pipe.named_steps['xgb'].cv_score_
    print(cv_score)

    #assert False
    
    y_pred_log = pipe.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    filename = f'pipeline_kernel_xgb_fe_cv{cv_score}.csv'

    submission = pd.DataFrame()
    submission['ID'] = id_test
    submission['target'] = y_pred
    submission.to_csv(filename, index=None)

    print(f"kaggle competitions submit -c santander-value-prediction-challenge -f {filename} -m 'na'")
    
if __name__ == '__main__':
    main()
