import pandas as pd
import numpy as np
from datetime import datetime
import os
import argparse
from sklearn.preprocessing import MinMaxScaler

ID = 'ID'
TARGET = 'target'

CSV_DIR = 'csv'
os.makedirs(CSV_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-c',  '--csvs',         nargs='+', default=[], help='Paths to CSVs')
parser.add_argument('-o',  '--output',       default=None, help='Paths to CSVs')
parser.add_argument('-d',  '--dry-run',      action='store_true', help='Dont store output file')
parser.add_argument('-g',  '--geometric',    action='store_true', help='Geometric mean instead of euclidean')
parser.add_argument('-w',  '--weights',      nargs='+', default=[], type=int, help='Weights of each CSV')
parser.add_argument('-aw', '--auto-weights', action='store_true', help='Assign weights based on pearson scores')

a = parser.parse_args()

a.output = a.output or '%s/avgensemble%s_%s.csv' % (CSV_DIR, '_g' if a.geometric else '', datetime.now().strftime("%Y-%m-%d-%H%M"))

test_predicts_list = []
if len(a.weights) > 0:
	assert len(a.csvs) == len(a.weights)

test_predicts_weights = []
for name in a.csvs:
    b1 = pd.read_csv(name)
    test_predicts_list.append(b1[TARGET].values)

pearson = np.corrcoef(test_predicts_list, rowvar=True)
print(pearson)
if a.auto_weights:
	assert len(a.weights) == 0
	a.weights = MinMaxScaler(feature_range=(1, 2), copy=True).fit_transform(
		np.expand_dims(-np.log(((np.sum(pearson, axis=1)-1) / (len(a.csvs) -1))), axis=1)).squeeze().tolist()
	print(f'Using auto weights: {a.weights}')

for _ in a.csvs:
    test_predicts_weights.append(1. if len(a.weights) == 0 else a.weights.pop(0))


weights_sum = np.sum(test_predicts_weights)
test_predicts_weights = [w / weights_sum for w in test_predicts_weights]

test_predicts = np.zeros(test_predicts_list[0].shape)

print(test_predicts_weights)

for fold_predict in test_predicts_list:
	if a.geometric:
		test_predicts += np.log1p(fold_predict) * test_predicts_weights.pop(0)
	else:
		test_predicts += fold_predict * test_predicts_weights.pop(0)


if a.geometric:
	test_predicts = np.expm1(test_predicts)

if not a.dry_run:
	subm = pd.read_csv('submission.csv', usecols=[ID])
	subm[TARGET] = test_predicts#np.clip(test_predicts, 0, 1)
	subm.to_csv(a.output, index=False)