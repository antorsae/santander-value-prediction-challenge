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
from tqdm import tqdm

import xgboost as xgb
import lightgbm
from sklearn.utils import resample
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from copy import copy
import argparse
import itertools
import sys

from leak import get_column_groups

print(' '.join(sys.argv))

allowed_decompositions = set(list('plstigra'))
allowed_regressors     = set(list('cxlr'))

parser = argparse.ArgumentParser()
parser.add_argument('-d', nargs='*',                type=str, help='Decompositions to use, any of {" ".join(allowed_decompositions)}')
parser.add_argument('-r', nargs='+', default=['x'], type=str, help='Regressors to use, any of {" ".join(allowed_decompositions)}')

parser.add_argument('-i',  '--iters',               default=3000,        type=int,   help='Max iterations/estimators/etc. for regressors')
parser.add_argument('-lr', '--learning-rate',       default=None,        type=float, help='Learning rate')
parser.add_argument(       '--catboost-device',     default='GPU',       type=str,   help='GPU or CPU')
parser.add_argument('-des', '--disable-early-stop', action='store_true', )
parser.add_argument('-v',   '--verbose',            action='store_true', )

parser.add_argument('-o',   '--oof', nargs='+',                type=str, help='OOF predictions for L2 model')
parser.add_argument('-od',  '--oof-dir',       default=None, type=str, help='Path to save OOF predictions')

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
parser.add_argument('-cg', '--column-groups',  action='store_true',    help='Only use features in discovered leak')
parser.add_argument('-ds', '--drop-selected',  action='store_true',    help='Drop selected colums')
parser.add_argument('-w',  '--weighted',       action='store_true',    help='Weight training samples based on similarity vs test distribution')
parser.add_argument('-bh', '--bins',           default=0,    type=int, help='Bins for histogram')
parser.add_argument(       '--leak',           default='baseline_3865_7853_0.9981888745148771.csv',    help='Leak file')
parser.add_argument('-p',  '--pseudo',         action='store_true',    help='Pseudo label training (using leak)')
parser.add_argument('-du', '--dummify-ugly',    action='store_true',     help='Dummify fake/ugly test rows')

parser.add_argument(       '--debug',          action='store_true',    help='Wait for remote debugger to attach')

a = parser.parse_args()

if a.debug:
	import ptvsd

	# Allow other computers to attach to ptvsd at this IP address and port, using the secret
	ptvsd.enable_attach("", address = ('0.0.0.0', 3000))

	# Pause the program until a remote debugger is attached
	ptvsd.wait_for_attach()

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
# l LightgbmRegressor
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

	df_train = pd.read_feather('./train.csv.pd')
	df_test = pd.read_feather('./test.csv.pd')
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


leak_Y =  np.log1p(pd.read_csv(a.leak)['target'].values) if a.leak else None

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

def transform_X(transform_fn, X_fit=X, X_transform=X, suffix='', **transform_fn_args):
	transform_name = transform_fn.__name__
	arg_names = '_'.join([f'{k}_{v.__str__()}' for k,v in transform_fn_args.items() if not isinstance(v, list)])
	feather_name = f'cache/mx_{transform_name}_{arg_names}_X{suffix}_{X_fit.shape[0]}_{X_fit.shape[1]}.fth'
	print(transform_name, feather_name)
	if not os.path.exists(feather_name):
		transform = transform_fn(**transform_fn_args)
		transform.fit(X_fit)
		transform_X = pd.DataFrame(transform.transform(X_transform))
		transform_X.columns = transform_X.columns.astype(str)
		transform_X.to_feather(feather_name)
	else:
		transform_X = pd.read_feather(feather_name)
	return transform_X

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

X_all = pd.DataFrame(index=X.index)
X_all_type =''

if a.bins != 0:
	_X = np.log1p(X.values)
	X_min, X_max = _X[_X>0].min(), _X.max()
	print(X_min, X_max, a.bins)
	X_bins = [0] + np.linspace(X_min, X_max, a.bins).tolist()
	X_bins[-1] += 1
	X_hist = np.zeros((_X.shape[0], len(X_bins)-1), dtype=int)
	for i in tqdm(range(_X.shape[0])):
	    X_hist[i] = np.histogram(_X[i], X_bins)[0]

	X_all = pd.concat([X_all, pd.DataFrame(X_hist)], axis=1, sort=False)
	X_all_type += f'bh{a.bins}_'

selected = None
if a.drop_selected: X_all_type += 'dropped'
if a.olivier:
	selected = get_olivier_features()
	X_all_type += 'olivier_'
elif a.column_groups:
	selected = list(itertools.chain.from_iterable(get_column_groups()) )
	X_all_type += 'columngroups_'
if selected:
	print(f"Using {len(selected)} columns")
	X = X[selected] if not a.drop_selected else X.drop(selected, axis=1)

idx_X_zeros = X==0.0
Xnan = X.copy()
Xnan[idx_X_zeros] = np.nan
drop_cols = set(drop_cols).intersection(set(Xnan.columns))

if a.dropX:
	X_all_type += 'dropX'
	# no concat
elif a.drop:
	X_all = pd.concat([X_all,  Xnan.drop(drop_cols, axis=1)], axis=1, sort=False)
	X_all_type += 'drop'
else:
	X_all = pd.concat([X_all,  Xnan], axis=1, sort=False)
	X_all_type += 'X'

X_f = X_t = X
if a.dummify_ugly:
	def not_ugly(row):
		for v in row.values[1:][row.values[1:] > 0]:
			if str(v)[::-1].find('.') > 2:
				return False
		return True

	X_f = X[X.apply(not_ugly, axis=1)]

if 'l' in decompositions: X_all = pd.concat([X_all, transform_X(LatentDirichletAllocation, X_f, X_t, n_components=num_decompose)], axis=1, sort=False)

#X[idx_X_zeros] = -1.0
if 'p' in decompositions: X_all = pd.concat([X_all, transform_X(PCA, X_f, X_t, n_components=num_decompose, random_state=random_state)], axis=1, sort=False)
if 't' in decompositions: X_all = pd.concat([X_all, transform_X(TruncatedSVD, X_f, X_t, n_components=num_decompose, random_state=random_state)], axis=1, sort=False)
if 'g' in decompositions: X_all = pd.concat([X_all, transform_X(GaussianRandomProjection, X_f, X_t, n_components=num_decompose, eps=0.1, random_state=random_state)], axis=1, sort=False)
if 'r' in decompositions: X_all = pd.concat([X_all, transform_X(SparseRandomProjection, X_f, X_t, n_components=num_decompose, dense_output=True, random_state=random_state)], axis=1, sort=False)
if 's' in decompositions: X_all = pd.concat([X_all, transform_X(SparsePCA, X_f, X_t, n_components=num_decompose, random_state=random_state)], axis=1, sort=False)
if 'i' in decompositions: X_all = pd.concat([X_all, transform_X(FastICA, X_f, X_t, n_components=num_decompose, random_state=random_state)], axis=1, sort=False)

if a.weighted:
	pca = transform_X(PCA, n_components=num_decompose, random_state=random_state)
	test_mean = np.mean(pca[Y.shape[0]:], axis=0)
	train_distances = np.linalg.norm(pca[:Y.shape[0]]- test_mean, axis=1)
	train_weights = MinMaxScaler(feature_range=(1, 100), copy=True).fit_transform(np.expand_dims(1. / np.log1p(train_distances), axis=1))

if a.meta_depth != 0:
	meta_X = None
	def op_mul(a,b): return 1 if b ==0 else a*b
	meta_op = op_mul if a.meta_mul else operator.pow
	for nn in range(a.meta_depth):
		for nnn, Xn in enumerate(np.array_split(X.values, meta_op(a.meta_base, nn), axis=1)):
			meta = transform_X(StatsTransformer,Xn, Xn, suffix=f'{a.meta_base}-{nn}-{nnn}', stat_funs=get_stat_funs(), verbose=2)
			meta_X = pd.concat([meta, meta_X], axis=1, sort=False) if meta_X is not None else meta
	X_all = pd.concat([X_all, meta_X], axis=1, sort=False)

print(X_all.values.shape)

X_all = X_all.T.drop_duplicates().T

print(X_all.shape)

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

xgb_params = {
    #'tree_method' : 'gpu_hist',
    'n_estimators':  a.iters,
    'objective': 'reg:linear',
    'booster': 'gbtree',
    'learning_rate': a.learning_rate or 0.02,
    'max_depth': 32,
    'min_child_weight': 57,
    'gamma' : 1.5,
    'alpha': 0.0,
    'lambda': 0.0,
    'subsample': 0.8,
    'colsample_bytree': 0.055*2,
    'colsample_bylevel': 0.50*2,
    'n_jobs': multiprocessing.cpu_count() // 2,
    'random_state': 456,
    }
xgb_fit_params = {
    'eval_metric': 'rmse',
    'verbose': a.verbose,
}

cb_fit_params = {
	'verbose': a.verbose,
}
lgb_fit_params = {
	'eval_metric' : 'l2_root', 
	'verbose' : -1 if not a.verbose else 1
}

if not a.disable_early_stop:
	xgb_fit_params['early_stopping_rounds'] = 200
	cb_fit_params['early_stopping_rounds'] = 200
	lgb_fit_params['early_stopping_rounds'] = 200

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
oof_fold_predictions = []

n_leak = np.where(leak_Y !=0)[0].shape[0]
print(np.arange(len(Y)).shape, n_leak)
to_train_idx = np.arange(len(Y))
Y_all_to_train = Y
if a.pseudo:
	to_train_idx = np.hstack([to_train_idx, len(Y) + np.where(leak_Y !=0)[0]])
	Y_all_to_train = np.hstack([Y_all_to_train, leak_Y[leak_Y!=0]])

X_all_to_train = X_all.iloc[to_train_idx]

X_oofs = []
Y_oofs = []
if a.oof:
	for oof_filename in a.oof:
		print(oof_filename)
		X_oof, Y_oof = pickle.load(open(oof_filename, 'rb'))
		X_oof = [np.log1p(__o) for __o in X_oof ]
		Y_oof = [np.log1p(__o) for __o in Y_oof ]
		X_oofs.append(X_oof)
		Y_oofs.append(Y_oof)

for X_oof in X_oofs:
	assert len(X_oof) == folds

regressor_best_iterations = []
for fold_id, (_IDX_train, IDX_test) in enumerate(
	KFold(n_splits=folds, random_state=random_state, shuffle=False).split(Y_all_to_train) if folds > 1 else [(range(len(Y_all_to_train)), None)] ):

	#print(len(_IDX_train), len(IDX_test))
	print(f'K-Fold run {fold_id+1}/{folds}:')
	train_idx_seen  = set()
	train_idx_total = set(_IDX_train)

	bootstrap_scores = []
	bootstrap_predictions = []
	
	oof_bootstrap_predictions = []

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

		X_train = X_all_to_train.iloc[IDX_train].values
		Y_train = Y_all_to_train[IDX_train]

		if X_train is []:
			X_train = np.zeros((Y_train.shape[0],0))

		if IDX_test is not None:
			X_test =  X_all_to_train.iloc[IDX_test].values
			Y_test =  Y_all_to_train[IDX_test]

		if X_test is []:
			X_test = np.zeros((Y_test.shape[0],0))

		if X_oofs:
			for X_oof in X_oofs:
				X_train_oof = X_test_oof = None
				for oof_fold_id, X_oof_fold in enumerate(X_oof):
					X_oof_fold = np.expand_dims(X_oof_fold, axis=-1)
					if oof_fold_id != fold_id:
						X_train_oof = np.vstack([X_train_oof, X_oof_fold]) if X_train_oof is not None else X_oof_fold
					else:
						X_test_oof  = np.vstack([ X_test_oof, X_oof_fold]) if X_test_oof  is not None else X_oof_fold

				X_train = np.hstack([X_train, X_train_oof])
				X_test  = np.hstack([X_test,  X_test_oof])

		print(X_train.shape)

		cb_clf = CatBoostRegressor(iterations= a.iters,
									learning_rate=a.learning_rate or 0.02,
									depth=6,
									eval_metric='RMSE',
									random_seed = fold_id,
									bagging_temperature = 0.5,
									#od_type='Iter' if not a.disable_early_stop else None,
									metric_period = 10,
									#od_wait=200,
									use_best_model = True,
									l2_leaf_reg = None,
									task_type = a.catboost_device
									)

		xgb_regressor = xgb.XGBRegressor(**xgb_params)

		svm_regressor = SVR(kernel='rbf', cache_size=20e3)

		lr_regressor = LinearRegression(n_jobs=-1)
		rf_regressor = RandomForestRegressor(n_estimators = a.iters, n_jobs=-1)
		lgb_regressor = lightgbm.LGBMRegressor(
	    	n_estimators = a.iters,
        	boosting_type = 'gbdt',
        	learning_rate = a.learning_rate or 0.007,
        	num_leaves = 512,
        	max_depth = 32,
        	n_jobs=  multiprocessing.cpu_count() // 2,
			min_data_in_leaf = 500,
        	random_state= random_state,
			verbose=-1 if not a.verbose else 1,
		)

		regressor_scores = []
		regressor_predictions = []
		oof_regressor_predictions = []

		regressor_list = []
		if 'x' in regressors: regressor_list.append((xgb_regressor, xgb_fit_params))
		if 'c' in regressors: regressor_list.append((cb_clf, cb_fit_params))
		if 'l' in regressors: regressor_list.append((lgb_regressor, lgb_fit_params))
		if 'r' in regressors: regressor_list.append((rf_regressor, {}))
		assert (len(regressor_list) > 0) and (len(regressor_list) == len(regressors))

		for regressor, fit_params in regressor_list:
			if IDX_test is not None and (
				isinstance(regressor, CatBoostRegressor) or 
				isinstance(regressor, xgb.XGBRegressor) or 
				isinstance(regressor, lightgbm.LGBMRegressor) ):
				fit_params['eval_set']=[(X_test, Y_test)]
			if a.weighted:
				fit_params['sample_weight'] = train_weights

			regressor.fit(X_train, Y_train, **fit_params)
			#print(regressor.get_feature_importance())

			predict_params = {}
			if isinstance(regressor, xgb.XGBRegressor) and not a.disable_early_stop:
				predict_params['ntree_limit'] = regressor.best_iteration
				regressor_best_iterations.append(regressor.best_iteration)

			if isinstance(regressor, lightgbm.LGBMRegressor) and not a.disable_early_stop:
				regressor_best_iterations.append(regressor.best_iteration_)

			if isinstance(regressor, CatBoostRegressor) and not a.disable_early_stop:
				regressor_best_iterations.append(regressor.tree_count_)
 
			score =  np.sqrt(mean_squared_error(regressor.predict(X_test, **predict_params), Y_test))

			print(f" Score {score} @ k-fold {fold_id+1}/{folds} @ bootstrap {bootstrap_run+1}/{bootstrap_runs}")
			regressor_scores.append(score)
			X_y = X_all[Y.shape[0]:].values
			if X_y == []:
				X_y = np.zeros((Y.shape[0], 0))
			if a.oof:
				for Y_oof in Y_oofs:
					X_y = np.hstack([X_y, np.expand_dims(Y_oof[fold_id], axis=-1)])
			T = regressor.predict(X_y)
			print(T)
			regressor_predictions.append(np.expm1(T))
			oof_regressor_predictions.append(np.expm1(regressor.predict(X_test)))

		bootstrap_scores.append(np.mean(regressor_scores))
		bootstrap_predictions.append(np.mean(regressor_predictions, axis=0))
		oof_bootstrap_predictions.append(np.mean(oof_regressor_predictions, axis=0))

	fold_scores.append(np.mean(bootstrap_scores))
	fold_predictions.append(np.mean(bootstrap_predictions, axis=0))
	oof_fold_predictions.append(np.mean(oof_bootstrap_predictions, axis=0))

del X_all

average_rmse = np.mean(fold_scores)
print(f'Average RMSE: {average_rmse} +- {np.var(fold_scores)}')
if not a.disable_early_stop:
	print(f'Best iterations {regressor_best_iterations}, mean {int(np.mean(regressor_best_iterations))}')

submissions = np.mean(fold_predictions, axis=0)
submissions[leak_Y != 0] = np.expm1(leak_Y[leak_Y != 0])
result = pd.DataFrame({'ID':ID,'target':submissions})
basename = f"{'_du' if a.dummify_ugly else ''}{'_w' if a.weighted else ''}_{X_all_type}_n{a.num_decompose}" \
	f"_d{''.join(decompositions)}_m{a.meta_base}{'mul' if a.meta_mul else 'pow'}{a.meta_depth}_s{a.select_k_best}" \
	f"_r{''.join(regressors)}_f{folds}_b{bootstrap_runs}{f'_i{a.iters}' if a.disable_early_stop else ''}" \
	f"_l{n_leak}_{'p' if a.pseudo else 'np'}" \
	f"_RMSE{average_rmse}"
result.to_csv(f"csv/sub{basename}.csv", index=False)
if a.oof_dir or not a.oof:
	oof_dir = a.oof_dir or 'oof'
	os.makedirs(oof_dir, exist_ok=True)
	pickle.dump( (oof_fold_predictions, fold_predictions), open( f"{oof_dir}/{basename}.pkl", "wb" ) )

print('end')
