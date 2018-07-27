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
parser.add_argument('-bh', '--bins',           default=0,    type=int, help='Bins for histogram')
parser.add_argument(       '--pseudo',         default='baseline_submission_with_leaks_all_1000.csv',    help='Pseudo-labeling')
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

def get_column_groups():
	return [ [
		'f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
		'15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
		'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b',
		'62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
		'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd',
		'703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
		'6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2',
		'190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'
	], [
		'4302b67ec', '75b663d7d', 'fc4a873e0', '1e9bdf471', '86875d9b0',
		'8f76eb6e5', '3d71c02f0', '05c9b6799', '26df61cc3', '27a7cc0ca',
		'9ff21281c', '3ce93a21b', '9f85ae566', '3eefaafea', 'afe8cb696',
		'72f9c4f40', 'be4729cb7', '8c94b6675', 'ae806420c', '63f493dba',
		'5374a601b', '5291be544', 'acff85649', '3690f6c26', '26c68cede',
		'12a00890f', 'dd84964c8', 'a208e54c7', 'fb06e8833', '7de39a7eb',
		'5fe3acd24', 'e53805953', '3de2a9e0d', '2954498ae', '6c3d38537',
		'86323e98a', 'b719c867c', '1f8a823f2', '9cc5d1d8f', 'd3fbad629'
	], [
		'51c141e64', '0e348d340', '64e010722', '55a763d90', '13b54db14',
		'01fdd93d3', '1ec48dbe9', 'cf3841208', 'd208491c8', '90b0ed912',
		'633e0d42e', '9236f7b22', '0824edecb', '71deb9468', '1b55f7f4d',
		'377a76530', 'c47821260', 'bf45d326d', '69f20fee2', 'd6d63dd07',
		'5ab3be3e1', '93a31829f', '121d8697e', 'f308f8d9d', '0e44d3981',
		'ecdef52b2', 'c69492ae6', '58939b6cc', '3132de0a3', 'a175a9aa4',
		'7166e3770', 'abbde281d', '23bedadb2', 'd4029c010', 'fd99222ee',
		'bd16de4ba', 'fb32c00dc', '12336717c', '2ea42a33b', '50108b5b5'
	], [
		'87ffda550', '63c094ba4', '2e103d632', '1c71183bb', 'd5fa73ead',
		'e078302ef', 'a6b6bc34a', 'f6eba969e', '0d51722ca', 'ce3d7595b',
		'6c5c8869c', 'dfd179071', '122c135ed', 'b4cfe861f', 'b7c931383',
		'44d5b820f', '4bcf15776', '51d4053c7', '1fe5d56b9', 'ea772e115',
		'ad009c8b9', '68a945b18', '62fb56487', 'c10f31664', 'cbb673163',
		'c8d582dd2', '8781e4b91', 'bd6da0cca', 'ca2b906e8', '11e12dbe8',
		'bb0ce54e9', 'c0d2348b7', '77deffdf0', 'f97d9431e', 'a09a238d0',
		'935ca66a9', '9de83dc23', '861076e21', 'f02ecb19c', '166008929'
	], [
		'ced6a7e91', '9df4daa99', '83c3779bf', 'edc84139a', 'f1e0ada11',
		'73687e512', 'aa164b93b', '342e7eb03', 'cd24eae8a', '8f3740670',
		'2b2a10857', 'a00adf70e', '3a48a2cd2', 'a396ceeb9', '9280f3d04',
		'fec5eaf1a', '5b943716b', '22ed6dba3', '5547d6e11', 'e222309b0',
		'5d3b81ef8', '1184df5c2', '2288333b4', 'f39074b55', 'a8b721722',
		'13ee58af1', 'fb387ea33', '4da206d28', 'ea4046b8d', 'ef30f6be5',
		'b85fa8b27', '2155f5e16', '794e93ca6', '070f95c99', '939f628a7',
		'7e814a30d', 'a6e871369', '0dc4d6c7d', 'bc70cbc26', 'aca228668'
	], [
		'5030aed26', 'b850c3e18', '212efda42', '9e7c6b515', '2d065b147',
		'49ca7ff2e', '37c85a274', 'ea5ed6ff7', 'deabe0f4c', 'bae4f747c',
		'ca96df1db', '05b0f3e9a', 'eb19e8d63', '235b8beac', '85fe78c6c',
		'cc507de6c', 'e0bb9cf0b', '80b14398e', '9ca0eee11', '4933f2e67',
		'fe33df1c4', 'e03733f56', '1d00f511a', 'e62cdafcf', '3aad48cda',
		'd36ded502', '92b13ebba', 'f30ee55dd', '1f8754c4e', 'db043a30f',
		'e75cfcc64', '5d8a55e6d', '6e29e9500', 'c5aa7c575', 'c2cabb902',
		'd251ee3b4', '73700eaa4', '8ab6f5695', '54b1c1bc0', 'cbd0256fb'
	], [
		'0f8d7b98e', 'c30ff7f31', 'ac0e2ebd0', '24b2da056', 'bd308fe52',
		'476d95ef1', '202acf9bd', 'dbc0c19ec', '06be6c2bb', 'd8296080a',
		'f977e99dc', '2191d0a24', '7db1be063', '1bc285a83', '9a3a1d59b',
		'c4d657c5b', 'a029667de', '21bd61954', '16bf5a9a2', '0e0f8504b',
		'5910a3154', 'ba852cc7a', '685059fcd', '21d6a4979', '78947b2ad',
		'1435ecf6b', '3839f8553', 'e9b5b8919', 'fa1dd6e8c', '632586103',
		'f016fd549', 'c25ea08ba', '7da54106c', 'b612f9b7e', 'e7c0a50e8',
		'29181e29a', '395dbfdac', '1beb0ce65', '04dc93c58', '733b3dc47'
	], [
		'ccc7609f4', 'ca7ea80a3', 'e509be270', '3b8114ab0', 'a355497ac',
		'27998d0f4', 'fa05fd36e', '81aafdb57', '4e22de94f', 'f0d5ffe06',
		'9af753e9d', 'f1b6cc03f', '567d2715c', '857020d0f', '99fe351ec',
		'3e5dab1e3', '001476ffa', '5a5eabaa7', 'cb5587baa', '32cab3140',
		'313237030', '0f6386200', 'b961b0d59', '9452f2c5f', 'bcfb439ee',
		'04a22f489', '7e58426a4', 'a4c9ea341', 'ffdc4bcf8', '1a6d866d7',
		'd7334935b', '298db341e', '08984f627', '8367dfc36', '5d9f43278',
		'7e3e026f8', '37c10d610', '5a88b7f01', '324e49f36', '99f466457'
	], [
		'a3e023f65', '9126049d8', '6eaea198c', '5244415dd', '0616154cc',
		'2165c4b94', 'fc436be29', '1834f29f5', '9d5af277d', 'c6850e7db',
		'6b241d083', '56f619761', '45319105a', 'fcda960ae', '07746dcda',
		'c906cd268', 'c24ea6548', '829fb34b8', '89ebc1b76', '22c019a2e',
		'1e16f11f3', '94072d7a3', '59dfc16da', '9886b4d22', '0b1741a7f',
		'a682ef110', 'e26299c3a', '5c220a143', 'ac0493670', '8d8bffbae',
		'68c7cf320', '3cea34020', 'e9a8d043d', 'afb6b8217', '5780e6ffa',
		'26628e8d8', '1de4d7d62', '4c53b206e', '99cc87fd7', '593cccdab'
	], [
		'b6daeae32', '3bdee45be', '3d6d38290', '5a1589f1a', '961b91fe7',
		'29c059dd2', 'cfc1ce276', '0a953f97e', '30b3daec2', 'fb5f5836e',
		'c7525612c', '6fa35fbba', '72d34a148', 'dcc269cfe', 'bdf773176',
		'469630e5c', '23db7d793', 'dc10234ae', '5ac278422', '6cf7866c1',
		'a39758dae', '45f6d00da', '251d1aa17', '84d9d1228', 'b98f3e0d7',
		'66146c12d', 'd6470c4ce', '3f4a39818', 'f16a196c6', 'b8f892930',
		'6f88afe65', 'ed8951a75', '371da7669', '4b9540ab3', '230a025ca',
		'f8cd9ae02', 'de4e75360', '540cc3cd1', '7623d805a', 'c2dae3a5a'
	], [
		'1d9078f84', '64e483341', 'a75d400b8', '4fe8154c8', '29ab304b9',
		'20604ed8f', 'bd8f989f1', 'c1b9f4e76', '4824c1e90', '4ead853dc',
		'b599b0064', 'd26279f1a', '58ed8fb53', 'ff65215db', '402bb0761',
		'74d7998d4', 'c7775aabf', '9884166a7', 'beb7f98fd', 'fd99c18b5',
		'd83a2b684', '18c35d2ea', '0c8063d63', '400e9303d', 'c976a87ad',
		'8a088af55', '5f341a818', '5dca793da', 'db147ffca', '762cbd0ab',
		'fb5a3097e', '8c0a1fa32', '01005e5de', '47cd6e6e4', 'f58fb412c',
		'a1db86e3b', '50e4f96cf', 'f514fdb2e', '7a7da3079', 'bb1113dbb'
	], [
		'df838756c', '2cb73ede7', '4dcf81d65', '61c1b7eb6', 'a9f61cf27',
		'1af4d24fa', 'e13b0c0aa', 'b9ba17eb6', '796c218e8', '37f57824c',
		'd1e0f571b', 'f9e3b03b7', 'a3ef69ad5', 'e16a20511', '04b88be38',
		'99e779ee0', '9f7b782ac', '1dd7bca9f', '2eeadde2b', '6df033973',
		'cdfc2b069', '031490e77', '5324862e4', '467bee277', 'a3fb07bfd',
		'64c6eb1cb', '8618bc1fd', '6b795a2bc', '956d228b9', '949ed0965',
		'a4511cb0b', 'b64425521', '2e3c96323', '191e21b5f', 'bee629024',
		'1977eaf08', '5e645a169', '1d04efde3', '8675bec0b', '8337d1adc'
	], [
		'e20edfcb8', '842415efb', '300d6c1f1', '720f83290', '069a2c70b',
		'87a91f998', '611151826', '74507e97f', '504e4b156', 'baa95693d',
		'cb4f34014', '5239ceb39', '81e02e0fa', 'dfdf4b580', 'fc9d04cd7',
		'fe5d62533', 'bb6260a44', '08d1f69ef', 'b4ced4b7a', '98d90a1d1',
		'b6d206324', '6456250f1', '96f5cf98a', 'f7c8c6ad3', 'cc73678bf',
		'5fb85905d', 'cb71f66af', '212e51bf6', 'd318bea95', 'b70c62d47',
		'11d86fa6a', '3988d0c5e', '42cf36d73', '9f494676e', '1c68ee044',
		'a728310c8', '612bf9b47', '105233ed9', 'c18cc7d3d', 'f08c20722'
	], [
		'2d60e2f7a', '11ad148bd', '54d3e247f', 'c25438f10', 'e6efe84eb',
		'964037597', '0196d5172', '47a8de42e', '6f460d92f', '0656586a4',
		'22eb11620', 'c3825b569', '6aa919e2e', '086328cc6', '9a33c5c8a',
		'f9c3438ef', 'c09edaf01', '85da130e3', '2f09a1edb', '76d34bbee',
		'04466547a', '3b52c73f5', '1cfb3f891', '704d68890', 'f45dd927f',
		'aba01a001', 'c9160c30b', '6a34d32d6', '3e3438f04', '038cca913',
		'504c22218', '56c679323', '002d634dc', '1938873fd', 'd37030d36',
		'162989a6d', 'e4dbe4822', 'ad13147bd', '4f45e06b3', 'ba480f343'
	], [
		'86cefbcc0', '717eff45b', '7d287013b', '8d7bfb911', 'aecaa2bc9',
		'193a81dce', '8dc7f1eb9', 'c5a83ecbc', '60307ab41', '3da5e42a7',
		'd8c61553b', '072ac3897', '1a382b105', 'f3a4246a1', '4e06e4849',
		'962424dd3', 'a3da2277a', '0a69cc2be', '408d191b3', '98082c8ef',
		'96b66294d', 'cc93bdf83', 'ffa6b80e2', '226e2b8ac', '678b3f377',
		'b56f52246', '4fa02e1a8', '2ef57c650', '9aeec78c5', '1477c751e',
		'a3c187bb0', '1ce516986', '080cd72ff', '7a12cc314', 'ead538d94',
		'480e78cb0', '737d43535', 'a960611d7', '4416cd92c', 'd5e6c18b0'
	], [
		'a1cd7b681', '9b490abb3', 'b10f15193', '05f54f417', 'a7ac690a8',
		'ed6c300c2', 'd0803e3a1', 'b1bb8eac3', 'bd1c19973', 'a34f8d443',
		'84ec1e3db', '24018f832', '82e01a220', '4c2064b00', '0397f7c9b',
		'ba42e41fa', '22d7ad48d', '9abffd22c', 'dbfa2b77f', '2c6c62b54',
		'9fa38def3', 'ecb354edf', '9c3154ae6', '2f26d70f4', '53102b93f',
		'a36b95f78', '1fa0f78d0', '19915a6d3', 'c944a48b5', '482b04cba',
		'2ce77a58f', '86558e595', 'c3f400e36', '20305585c', 'f8ccfa064',
		'dd771cb8e', '9aa27017e', 'cd7f0affd', '236cc1ff5', 'a3fc511cd'
	], [
		'3b843ae7e', 'c8438b12d', 'd1b9fc443', '19a45192a', '63509764f',
		'6b6cd5719', 'b219e3635', '4b1d463d7', '4baa9ff99', 'b0868a049',
		'3e3ea106e', '043e4971a', 'a2e5adf89', '25e2bcb45', '3ac0589c3',
		'413bbe772', 'e23508558', 'c1543c985', '2dfea2ff3', '9dcdc2e63',
		'1f1f641f1', '75795ea0a', 'dff08f7d5', '914d2a395', '00302fe51',
		'c0032d792', '9d709da93', 'cb72c1f0b', '5cf7ac69f', '6b1da7278',
		'47b5abbd6', '26163ffe1', '902c5cd15', '45bc3b302', '5c208a931',
		'e88913510', 'e1d6a5347', '38ec5d3bb', 'e3d64fcd7', '199d30938'
	], [
		'0d7692145', '62071f7bc', 'ab515bdeb', 'c30c6c467', 'eab76d815',
		'b6ee6dae6', '49063a8ed', '4cb2946ce', '6c27de664', '772288e75',
		'afd87035a', '44f2f419e', '754ace754', 'e803a2db0', 'c70f77ef2',
		'65119177e', '3a66c353a', '4c7768bff', '9e4765450', '24141fd90',
		'dc8b7d0a8', 'ba499c6d9', '8b1379b36', '5a3e3608f', '3be3c049e',
		'a0a3c0f1b', '4d2ca4d52', '457bd191d', '6620268ab', '9ad654461',
		'1a1962b67', '7f55b577c', '989d6e0f5', 'bc937f79a', 'e059a8594',
		'3b74ac37b', '555265925', 'aa37f9855', '32c8b9100', 'e71a0278c'
	], [
		'266525925', '4b6dfc880', '2cff4bf0c', 'a3382e205', '6488c8200',
		'547d3135b', 'b46191036', '453128993', '2599a7eb7', '2fc60d4d9',
		'009319104', 'de14e7687', 'aa31dd768', '2b54cddfd', 'a67d02050',
		'37aab1168', '939cc02f5', '31f72667c', '6f951302c', '54723be01',
		'4681de4fd', '8bd53906a', '435f27009', 'f82167572', 'd428161d9',
		'9015ac21d', 'ec4dc7883', '22c7b00ef', 'd4cc42c3d', '1351bf96e',
		'1e8801477', 'b7d59d3b5', 'a459b5f7d', '580f5ff06', '39b3c553a',
		'1eec37deb', '692c44993', 'ce8ce671e', '88ef1d9a8', 'bf042d928'
	], [
		'c13ee1dc9', 'abb30bd35', 'd2919256b', '66728cc11', 'eab8abf7a',
		'cc03b5217', '317ee395d', '38a92f707', '467c54d35', 'e8f065c9d',
		'2ac62cba5', '6495d8c77', '94cdda53f', '13f2607e4', '1c047a8ce',
		'28a5ad41a', '05cc08c11', 'b0cdc345e', '38f49406e', '773180cf6',
		'1906a5c7e', 'c104aeb2e', '8e028d2d2', '0dc333fa1', '28a785c08',
		'03ee30b8e', '8e5a41c43', '67102168f', '8b5c0fb4e', '14a22ab1a',
		'9fc776466', '4aafb7383', '8e1dfcb94', '55741d46d', '8f940cb1b',
		'758a9ab0e', 'fd812d7e0', '4ea447064', '6562e2a2c', '343922109'
	], [
		'5b465f819', 'a2aa0e4e9', '944e05d50', '4f8b27b6b', 'a498f253f',
		'c73c31769', '025dea3b3', '616c01612', 'f3316966c', '83ea288de',
		'2dbeac1de', '47b7b878f', 'b4d41b335', '686d60d8a', '6dcd9e752',
		'7210546b2', '78edb3f13', '7f9d59cb3', '30992dccd', '26144d11f',
		'a970277f9', '0aea1fd67', 'dc528471e', 'd51d10e38', 'efa99ed98',
		'48420ad48', '7f38dafa6', '1af4ab267', '3a13ed79a', '73445227e',
		'971631b2d', '57c4c03f6', '7f91dc936', '0784536d6', 'c3c3f66ff',
		'052a76b0f', 'ffb34b926', '9d4f88c7b', '442b180b6', '948e00a8d'
	], [
		'920a04ee2', '93efdb50f', '15ea45005', '78c57d7cd', '91570fb11',
		'c5dacc85b', '145c7b018', '590b24ab1', 'c283d4609', 'e8bd579ae',
		'7298ca1ef', 'ce53d1a35', 'a8f80f111', '2a9fed806', 'feb40ad9f',
		'cfd255ee3', '31015eaab', '303572ae2', 'cd15bb515', 'cb5161856',
		'a65b73c87', '71d64e3f7', 'ec5fb550f', '4af2493b6', '18b4fa3f5',
		'3d655b0ed', '5cc9b6615', '88c0ec0a6', '8722f33bb', '5ed0c24d0',
		'54f26ee08', '04ecdcbb3', 'ade8a5a19', 'd5efae759', 'ac7a97382',
		'e1b20c3a6', 'b0fcfeab8', '438b8b599', '43782ef36', 'df69cf626'
	], [
		'fec5644cf', 'caa9883f6', '9437d8b64', '68811ba58', 'ef4b87773',
		'ff558c2f2', '8d918c64f', '0b8e10df6', '2d6565ce2', '0fe78acfa',
		'b75aa754d', '2ab9356a0', '4e86dd8f3', '348aedc21', 'd7568383a',
		'856856d94', '69900c0d1', '02c21443c', '5190d6dca', '20551fa5b',
		'79cc300c7', '8d8276242', 'da22ed2b8', '89cebceab', 'f171b61af',
		'3a07a8939', '129fe0263', 'e5b2d137a', 'aa7223176', '5ac7e84c4',
		'9bd66acf6', '4c938629c', 'e62c5ac64', '57535b55a', 'a1a0084e3',
		'2a3763e18', '474a9ec54', '0741f3757', '4fe8b17c2', 'd5754aa08'
	]]

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


pseudo_Y =  np.log1p(pd.read_csv(a.pseudo)['target'].values) if a.pseudo else None

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

if a.olivier:

	X = X[get_olivier_features()]
	X_all_type += 'olivier_'

idx_X_zeros = X==0.0
Xnan = X.copy()
Xnan[idx_X_zeros] = np.nan

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

#Xnn = X.values.copy()
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
#ut = UniqueTransformer().fit(X_all.values)
#X_all = ut.transform(X_all.values)

print(X_all.shape)

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

xgb_params = {
    #'tree_method' : 'gpu_hist',
    'n_estimators': 1000,
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
    'verbose': True,
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

print(np.arange(len(Y)).shape, np.where(pseudo_Y !=0)[0].shape)
to_train_idx = np.hstack([np.arange(len(Y)), len(Y) + np.where(pseudo_Y !=0)[0]])

X_all_to_train = X_all.iloc[to_train_idx]
Y_all_to_train = np.hstack([Y, pseudo_Y[pseudo_Y!=0]])

for fold_id, (_IDX_train, IDX_test) in enumerate(
	KFold(n_splits=folds, random_state=random_state, shuffle=False).split(Y_all_to_train) if folds > 1 else [(range(len(Y_all_to_train)), None)] ):

	#print(len(_IDX_train), len(IDX_test))
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

		X_train = X_all_to_train.iloc[IDX_train].values
		Y_train = Y_all_to_train[IDX_train]

		if IDX_test is not None:
			X_test =  X_all_to_train.iloc[IDX_test].values
			Y_test =  Y_all_to_train[IDX_test]

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
			if IDX_test is not None and not isinstance(regressor, SVR):
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
submissions[pseudo_Y != 0] = np.expm1(pseudo_Y[pseudo_Y != 0])
result = pd.DataFrame({'ID':ID
					,'target':submissions})
result.to_csv(f"csv/sub{'_du' if a.dummify_ugly else ''}{'_w' if a.weighted else ''}_{X_all_type}_n{a.num_decompose}_d{''.join(decompositions)}_m{a.meta_base}{'mul' if a.meta_mul else 'pow'}{a.meta_depth}_s{a.select_k_best}_r{''.join(regressors)}_f{folds}_b{bootstrap_runs}_RMSE{average_rmse}.csv", index=False)

print('end')
