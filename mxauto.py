import os
import time
import pickle
import numpy as np
import pandas as pd
import pickle
from xgtrain import StatsTransformer, get_stat_funs, UniqueTransformer
from sklearn.base import clone
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, LatentDirichletAllocation, SparsePCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import operator
import multiprocessing

import xgboost as xgb
from sklearn.utils import resample
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from copy import copy
import argparse
import itertools

allowed_decompositions = set(list('plstigra'))
allowed_regressors     = set(list('cx'))

parser = argparse.ArgumentParser()
parser.add_argument('-d', nargs='*',                type=str, help='Decompositions to use, any of {" ".join(allowed_decompositions)}')
parser.add_argument('-r', nargs='+', default=['x'], type=str, help='Regressors to use, any of {" ".join(allowed_decompositions)}')

parser.add_argument('-n',  '--num-decompose',  default=20,   type=int, help='Components for dimensionality redux')
parser.add_argument('-md', '--meta-depth',     default=0,    type=int, help='Depth for calculating meta-features splits(0 to disable), e.g. -md 6')
parser.add_argument('-mb', '--meta-base',      default=2,    type=int, help='Base for calculating meta-features splits, e.g. -mb 7')
parser.add_argument('-mm', '--meta-mul',       action='store_true',    help='Use multiplications instead of powers for meta-features splits')
parser.add_argument('-f',  '--folds',          default=5,    type=int, help='K-Folds')
parser.add_argument('-b',  '--bootstrap-runs', default=1,    type=int, help='Bagging runs')
parser.add_argument('-s',  '--select-k-best',  default=None, type=int, help='Select K Best features, e.g. -s 120')
parser.add_argument(       '--drop',           action='store_true',    help='Drop redundant colums')
parser.add_argument(       '--dropX',          action='store_true',    help='Drop X altogether')
parser.add_argument(       '--olivier',        action='store_true',    help='Only use features in https://www.kaggle.com/ogrellier/santander-46-features/code')
parser.add_argument('-w',  '--weighted',       action='store_true',    help='Weight training samples based on similarity vs test distribution')

a = parser.parse_args()
print(a.d)

if a.d == []:
	a.d = list(allowed_decompositions)

if a.d:
	decompositions = sorted(set(itertools.chain.from_iterable([list(s) for s in a.d])))
else:
	decompositions = []

# p PCA
# l LatentDirichletAllocation
# s SparsePCA
# t TruncatedSVD
# i FastICA
# g GaussianRandomProjection
# r SparseRandomProjection
# a autoencoder
if decompositions != []:
	assert allowed_decompositions.issuperset(set(decompositions))

regressors     = sorted(set(itertools.chain.from_iterable([list(s) for s in a.r])))
# c CatBoostRegressor
# x XGBRegressor
assert allowed_regressors.issuperset(set(regressors))

print(f"Using {decompositions} compositions and {regressors} regressors")

def get_olivier_features(): # from https://www.kaggle.com/ogrellier/santander-46-features/code
    return [
        'f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a',
        '491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
        '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2',
        'adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
        '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0',
        '58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
        '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b',
        'b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1'
    ]

batch_size = 1500
epochs = (6,6,6,5,5)
num_decompose = a.num_decompose

time_fs = time.time()
np.random.seed(22)

X_FEATHER = 'cache/mx_X.fth'
MX_PKL = 'cache/mx.pkl'

print('read data:')
if not os.path.exists(X_FEATHER):

	df_train = pd.read_csv('./train.csv')
	df_test = pd.read_csv('./test.csv')
	df_s = df_train.std(axis=0)
	drop_cols = df_s[df_s==0].index
	Y = np.log1p(df_train.target.values)
	ID = df_test.ID.values
	df_train = df_train.drop(['ID','target'], axis=1)
	df_test = df_test.drop(['ID'], axis=1)
	X = pd.concat([df_train,df_test], axis=0, sort=False, ignore_index=True)
	#X = np.log1p(X)
	#X = X.div(X.max(), axis='columns')
	#y_min = np.min(Y)
	#y_max = np.max(Y)
	#Y = (Y - y_min) / (y_max - y_min)
	X.to_feather(X_FEATHER)
	#Y.to_feather(Y_FEATHER)
	pickle.dump( (Y, ID, drop_cols), open( MX_PKL, "wb" ) )

	del df_train, df_test, df_s
else:
	#Y = pd.read_feather(Y_FEATHER)
	X = pd.read_feather(X_FEATHER)
	(Y, ID, drop_cols) = pickle.load(open(MX_PKL, "rb" ) )	

def get_manifold(X):
	from mxnet import nd, Context
	from mxnet import ndarray as F
	from mxnet.gluon import Block, nn
	from mxnet.initializer import Uniform

	class Model(Block):
		def __init__(self, num_dim, **kwargs):
			super(Model, self).__init__(**kwargs)
			wi1 = Uniform(0.25)
			wi2 = Uniform(0.1)
			with self.name_scope():
				self.encoder1 = nn.Dense(num_dim//4, in_units=num_dim, weight_initializer=wi1)
				self.encoder2 = nn.Dense(num_dim//16, in_units=num_dim//4, weight_initializer=wi1)
				self.encoder3 = nn.Dense(num_dim//64, in_units=num_dim//16, weight_initializer=wi2)
				self.encoder4 = nn.Dense(num_dim//256, in_units=num_dim//64, weight_initializer=wi2)
				self.decoder4 = nn.Dense(num_dim//64, in_units=num_dim//256, weight_initializer=wi2)
				self.decoder3 = nn.Dense(num_dim//16, in_units=num_dim//64, weight_initializer=wi2)
				self.decoder2 = nn.Dense(num_dim//4, in_units=num_dim//16, weight_initializer=wi1)
				self.decoder1 = nn.Dense(num_dim, in_units=num_dim//4, weight_initializer=wi1)
			self.layers = [(self.encoder1,self.decoder1),
						(self.encoder2,self.decoder2),
						(self.encoder3,self.decoder3),
						(self.encoder4,self.decoder4)]

			for layer in self.layers:
				self.register_child(layer[0])
				self.register_child(layer[1])
				
		def onelayer(self, x, layer):
			xx = F.tanh(layer[0](x))
			#xx = nn.HybridLambda('tanh')(layer[0](x))
 
			return layer[1](xx)
		
		def oneforward(self, x, layer):
			return F.tanh(layer[0](x))
		
		def forward(self, x):
			n_layer = len(self.layers)
			for i in range(n_layer):
				x = F.tanh(self.layers[i][0](x))
			for i in range(n_layer-1):
				x = F.tanh(self.layers[n_layer-i-1][1](x))
			return self.layers[0][1](x)
		
		def manifold(self, x):
			n_layer = len(self.layers)
			for i in range(n_layer-1):
				x = F.tanh(self.layers[i][0](x))
			return self.layers[n_layer-1][0](x)

	from mxnet import autograd
	from mxnet import gpu, cpu
	from mxnet.gluon import Trainer
	from mxnet.gluon.loss import L2Loss

	# Stacked AutoEncoder
	#model.initialize(ctx=[cpu(0),cpu(1),cpu(2),cpu(3)])
	#ctx = [gpu(1)]
	#ctx = [cpu(i) for i in range(16)]
	with  Context(gpu(0)) as ctx:
		model = Model(X.shape[1])
		model.initialize(ctx=ctx)#,cpu(2),cpu(3)])

		# Select Trainign Algorism
		trainer = Trainer(model.collect_params(),'adam')
		loss_func = L2Loss()

		# Start Pretraining
		print('start pretraining of StackedAE...')
		loss_n = [] # for log

		buffer = nd.array(X.values)
		for layer_id, layer in enumerate(model.layers):
			print('layer %d of %d...'%(layer_id+1,len(model.layers)))
			trainer.set_learning_rate(0.02)
			for epoch in range(1, epochs[layer_id] + 1):
				# random indexs for all datas
				indexs = np.random.permutation(buffer.shape[0])
				for bs in range(0,buffer.shape[0],batch_size):
					be = min(buffer.shape[0],bs+batch_size)
					data = buffer[indexs[bs:be]]
					# forward
					with autograd.record():
						output = model.onelayer(data, layer)
						# make loss
						loss = loss_func(output, data)
						# for log
						loss_n.append(np.mean(loss.asnumpy()))
						del output
					# backward
					loss.backward()
					# step training to one batch
					trainer.step(batch_size, ignore_stale_grad=True)
					del data, loss
				# show log
				print('%d/%d epoch loss=%f...'%(epoch,epochs[layer_id],np.mean(loss_n)))
				loss_n = []
				del bs, be, indexs
			buffer = model.oneforward(buffer, layer)
		del layer, loss_n, buffer

		print('start training of StackedAE...')
		loss_n = []
		buffer = nd.array(X.values)
		trainer.set_learning_rate(0.02)
		for epoch in range(1, epochs[-1] + 1):
			# random indexs for all datas
			indexs = np.random.permutation(buffer.shape[0])
			for bs in range(0,buffer.shape[0],batch_size):
				be = min(buffer.shape[0],bs+batch_size)
				data = buffer[indexs[bs:be]]
				# forward
				with autograd.record():
					output = model(data)
					# make loss
					loss = loss_func(output, data)
					# for log
					loss_n.append(np.mean(loss.asnumpy()))
					del output
				# backward
				loss.backward()
				# step training to one batch
				trainer.step(batch_size, ignore_stale_grad=True)
				del data, loss
			# show log
			print('%d/%d epoch loss=%f...'%(epoch,epochs[-1],np.mean(loss_n)))
			loss_n = []
			del bs, be, indexs
		del trainer, loss_func, loss_n, buffer

		print('making manifold...')
		manifold_X = pd.DataFrame()
		for bs in range(0,X.shape[0],batch_size):
			be = min(X.shape[0],bs + batch_size)
			nx = nd.array(X.iloc[bs:be].values)
			df = pd.DataFrame(model.manifold(nx).asnumpy())
			manifold_X = manifold_X.append(df, ignore_index=True, sort=False)
			del be, df, nx
		del model, bs
		return manifold_X

def transform_X(transform_fn, X=X, suffix='', **transform_fn_args):
	transform_name = transform_fn.__name__
	arg_names = '_'.join([f'{k}_{v.__str__()}' for k,v in transform_fn_args.items() if not isinstance(v, list)])
	feather_name = f'cache/mx_{transform_name}_{arg_names}_X{suffix}_{X.shape[0]}_{X.shape[1]}.fth'
	print(transform_name, feather_name)
	if not os.path.exists(feather_name):
		transform = transform_fn(**transform_fn_args)
		transform_X = pd.DataFrame(transform.fit_transform(X))
		transform_X.columns = transform_X.columns.astype(str)
		transform_X.to_feather(feather_name)
	else:
		transform_X = pd.read_feather(feather_name)
	return transform_X

def get_meta():
	if False:
		meta_X = pd.DataFrame({
			'soz':(X[X==0]).fillna(1).sum(axis=1),
			'mean':X.mean(axis=1),
			'std' :X.std(axis=1),
			'nzm' :X[X!=0].mean(axis=1),
			'nzs' :X[X!=0].std(axis=1),
			'med' :X[X!=0].median(axis=1),
			'max' :X[X!=0].max(axis=1),
			'min' :X[X!=0].min(axis=1),
			'var' :X[X!=0].var(axis=1)})
	else:
		meta_X = transform_X(StatsTransformer,X = X.values, stat_funs=get_stat_funs(), verbose=2)

	return meta_X

if 'a' in decompositions:
	manifold_name = f'cache/mx_manifold_X.csv'
	if not os.path.exists(manifold_name):
		X_norm = np.log1p(X)
		X_norm = X_norm.div(X_norm.max(), axis='columns')
		manifold_X = get_manifold(X_norm)
		manifold_X.to_csv(manifold_name, index=False)
	else:
		manifold_X = pd.read_csv(manifold_name)

random_state = 17

X_all_type =''
if a.olivier:
	X = X[get_olivier_features()]
	X_all_type += 'olivier_'

if a.dropX:
	X_all = None
	X_all_type += 'dropX'
elif a.drop:
	X_all = X.drop(drop_cols, axis=1)
	X_all_type += 'drop'
else:
	X_all = X
	X_all_type += 'X'

if 'l' in decompositions: X_all = pd.concat([X_all, transform_X(LatentDirichletAllocation, n_components=num_decompose)], axis=1, sort=False)

X[X==0.0] = -1.0
if 'p' in decompositions: X_all = pd.concat([X_all, transform_X(PCA, n_components=num_decompose, random_state=random_state)], axis=1, sort=False)
if 't' in decompositions: X_all = pd.concat([X_all, transform_X(TruncatedSVD, n_components=num_decompose, random_state=random_state)], axis=1, sort=False)
if 'g' in decompositions: X_all = pd.concat([X_all, transform_X(GaussianRandomProjection, n_components=num_decompose, eps=0.1, random_state=random_state)], axis=1, sort=False)
if 'r' in decompositions: X_all = pd.concat([X_all, transform_X(SparseRandomProjection, n_components=num_decompose, dense_output=True, random_state=random_state)], axis=1, sort=False)
if 's' in decompositions: X_all = pd.concat([X_all, transform_X(SparsePCA, n_components=num_decompose, random_state=random_state)], axis=1, sort=False)
if 'i' in decompositions: X_all = pd.concat([X_all, transform_X(FastICA, n_components=num_decompose, random_state=random_state)], axis=1, sort=False)

if a.weighted:
	pca = transform_X(PCA, n_components=num_decompose, random_state=random_state)
	test_mean = np.mean(pca[Y.shape[0]:], axis=0)
	train_distances = np.linalg.norm(pca[:Y.shape[0]]- test_mean, axis=1)
	train_weights = MinMaxScaler(feature_range=(1, 100), copy=True).fit_transform(np.expand_dims(1. / np.log1p(train_distances), axis=1))

#Xnn = X.values.copy()
if a.meta_depth != 0:
	meta_X = None
	def op_mul(a,b): return 1 if b ==0 else a*b
	meta_op = op_mul if a.meta_mul else operator.pow
	for nn in range(a.meta_depth):
		for nnn, Xn in enumerate(np.array_split(X.values, meta_op(a.meta_base, nn), axis=1)):
			meta = transform_X(StatsTransformer,X = Xn, suffix=f'{a.meta_base}-{nn}-{nnn}', stat_funs=get_stat_funs(), verbose=2)
			meta_X = pd.concat([meta, meta_X], axis=1, sort=False) if meta_X is not None else meta
	X_all = pd.concat([X_all, meta_X], axis=1, sort=False)

print(X_all.values.shape)

X_all = X_all.T.drop_duplicates().T
#ut = UniqueTransformer().fit(X_all.values)
#X_all = ut.transform(X_all.values)

print(X_all.shape)

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

xgb_params = {
    #'tree_method' : 'gpu_hist',
    'n_estimators': 15000,
    'objective': 'reg:linear',
    'booster': 'gbtree',
    'learning_rate': 0.02,
    'max_depth': 32,
    'min_child_weight': 57,
    'gamma' : 1.5,
    'alpha': 0.0,
    'lambda': 0.0,
    'subsample': 0.8,
    'colsample_bytree': 0.055*2,
    'colsample_bylevel': 0.50*2,
    'n_jobs': multiprocessing.cpu_count(),
    'random_state': 456,
    }
xgb_fit_params = {
    'early_stopping_rounds': 200,
    'eval_metric': 'rmse',
    'verbose': False,
}

if a.select_k_best is not None:
	print(f"Selecting {a.select_k_best} best features")
	def score_features(X, y, estimator=None):
	    return clone(estimator).fit(X, y).feature_importances_

	xgb_regressor = xgb.XGBRegressor(**xgb_params)

	fs = SelectKBest(score_func=lambda X, y: score_features(X, y, estimator=xgb_regressor), k=a.select_k_best).fit(X_all[:Y.shape[0]], Y)

	X_all = X_all.iloc[:, fs.get_support(indices=True)]

print(X_all.shape)

print('start training...')

folds = a.folds
bootstrap_runs = a.bootstrap_runs

fold_scores = []
fold_predictions = []

for fold_id, (_IDX_train, IDX_test) in enumerate(KFold(n_splits=folds, random_state=random_state, shuffle=False).split(Y)):

	print(f'K-Fold run {fold_id+1}/{folds}:')
	train_idx_seen  = set()
	train_idx_total = set(_IDX_train)

	bootstrap_scores = []
	bootstrap_predictions = []

	for bootstrap_run in range(bootstrap_runs):

		print(f' Boostrapping run {bootstrap_run+1}/{bootstrap_runs}:')

		if bootstrap_runs > 1:
			IDX_train = resample(_IDX_train, n_samples=len(_IDX_train))
		else:
			IDX_train = _IDX_train

		train_idx_seen = train_idx_seen.union(set(IDX_train))

		# add those unseen items in the last run to make sure we use all training samples
		if (bootstrap_run == bootstrap_runs -1) and (bootstrap_runs > 1):
			IDX_train = np.hstack([IDX_train, list(train_idx_total.difference(train_idx_seen))])

		X_train = X_all.iloc[IDX_train].values
		X_test =  X_all.iloc[IDX_test].values

		Y_train = Y[IDX_train]
		Y_test =  Y[IDX_test]

		cb_clf = CatBoostRegressor(iterations=30000,
									learning_rate=0.005,
									depth=6,
									eval_metric='RMSE',
									random_seed = fold_id,
									bagging_temperature = 0.5,
									od_type='Iter',
									metric_period = 200,
									od_wait=200,
									l2_leaf_reg = None,
									)

		xgb_regressor = xgb.XGBRegressor(**xgb_params)

		svm_regressor = SVR(kernel='rbf', cache_size=20e3)

		regressor_scores = []
		regressor_predictions = []

		regressor_list = []
		if 'x' in regressors: regressor_list.append((xgb_regressor, xgb_fit_params))
		if 'c' in regressors: regressor_list.append((cb_clf, {}))
		assert (len(regressor_list) > 0) and (len(regressor_list) == len(regressors))

		for regressor, fit_params in regressor_list:
			if not isinstance(regressor, SVR):
				fit_params['eval_set']=[(X_test, Y_test)]
			if a.weighted:
				fit_params['sample_weight'] = train_weights

			regressor.fit(X_train, Y_train, **fit_params)
			if isinstance(regressor, xgb.XGBRegressor):
				score = regressor.best_score
			elif isinstance(regressor, SVR):
				score =  np.sqrt(mean_squared_error(regressor.predict(X_test), Y_test))
			else:
				score =  np.sqrt(mean_squared_error(regressor.get_test_eval(), Y_test))

			print(f" Score {score} @ k-fold {fold_id+1}/{folds} @ bootstrap {bootstrap_run+1}/{bootstrap_runs}")
			regressor_scores.append(score)
			T = regressor.predict(X_all[Y.shape[0]:].values)
			regressor_predictions.append(np.expm1(T))

		bootstrap_scores.append(np.mean(regressor_scores))
		bootstrap_predictions.append(np.mean(regressor_predictions, axis=0))

	fold_scores.append(np.mean(bootstrap_scores))
	fold_predictions.append(np.mean(bootstrap_predictions, axis=0))

del X_all

average_rmse = np.mean(fold_scores)
print(f'Average RMSE: {average_rmse}')

submissions = np.mean(fold_predictions, axis=0)
result = pd.DataFrame({'ID':ID
					,'target':submissions})
result.to_csv(f"csv/sub{'_w' if a.weighted else ''}_{X_all_type}_n{a.num_decompose}_d{''.join(decompositions)}_m{a.meta_base}{'mul' if a.meta_mul else 'pow'}{a.meta_depth}_s{a.select_k_best}_r{''.join(regressors)}_f{folds}_b{bootstrap_runs}_RMSE{average_rmse}.csv", index=False)

print('end')
