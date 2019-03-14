from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



class FeatureSelector():
	def __init__(self,method):
		
		self.data_dir = 'data/prep_data/'
		self.method = method
		self.verbose = 1

	def load_data(self,split):
		assert split in ('train','valid','test')
		X = pd.read_csv('%s%s_X.csv'%(self.data_dir,split))
		y = pd.read_csv('%s%s_y.csv'%(self.data_dir,split))
		if self.verbose > 0:
			print(X.shape,y.shape)
			print(np.unique(y,return_counts=True))
		return X,y

	def get_model(self):
		if self.method == 'variance': # Unsupervised
			p = .5
			selector = feature_selection.VarianceThreshold(threshold=(p*(1-p)))
		if self.method == 'rfe':
			estimator = LogisticRegression()
			# selector = feature_selection.RFE(estimator, n_features_to_select=10, step=1,verbose=0)
			selector = feature_selection.RFE(estimator, n_features_to_select=1, step=1,verbose=0)

		return selector

	def feature_selection(self):
		selector = self.get_model()
		train_X,train_y = self.load_data(split='train')
		valid_X,valid_y = self.load_data(split='valid')
		
		columns = train_X.columns

		if self.method == 'variance':
			train_X = selector.fit_transform(train_X)
		elif self.method == 'rfe':
			print(train_y.shape)
			selector.fit(train_X,train_y.values.ravel())
			ranking = selector.ranking_
			idx = np.argsort(ranking)
			feature_ranking = columns[idx]
			ranking = ranking[idx]
			print([x for x in zip(ranking,feature_ranking)])
			# for r,f in zip(ranking,feature_ranking):
				# print(r,f)
		

		feature_idx = selector.get_support()
		sel_columns = columns[feature_idx]
		removed_feats = np.setdiff1d(columns, sel_columns)


		# print('Total Features %d | %s'%(len(columns),columns))
		# print('Selected Features %d | %s'%(len(sel_columns),sel_columns))
		print('Removed Features %d | %s'%(len(removed_feats),removed_feats))

if __name__ == '__main__':
	# feat_sel = FeatureSelector(method='variance')
	feat_sel = FeatureSelector(method='rfe')
	feat_sel.feature_selection()
