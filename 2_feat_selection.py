from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import joblib


import pandas as pd
import numpy as np

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



class FeatureSelector():
	def __init__(self,method,feat_limit=False):
		
		self.data_dir = 'data/prep_orig_data/'
		self.method = method
		self.feat_limit = feat_limit
		self.verbose = 1

		self.name = 'feat_sel_%s'%(method)
		self.model_save_path = 'models/%s.pkl'%(self.name)

	def load_data(self,split):
		assert split in ('train','test')
		X = pd.read_csv('%s%s_X.csv'%(self.data_dir,split))
		y = pd.read_csv('%s%s_y.csv'%(self.data_dir,split))
		if self.verbose > 0:
			print(X.shape,y.shape)
			print(np.unique(y,return_counts=True))
		return X,y

	def get_model(self,resume=False):
		if not resume:
			if self.method == 'variance': # Unsupervised
				p = .5
				selector = feature_selection.VarianceThreshold(threshold=(p*(1-p)))
			elif self.method == 'rfe':
				estimator = LogisticRegression()
				selector = feature_selection.RFE(estimator, n_features_to_select=self.feat_limit, step=1,verbose=0)
			elif self.method == 'forward':
				estimator = ExtraTreesClassifier(n_estimators=100)
				selector = SelectFromModel(estimator)
			elif self.method == 'seq_bwd':
				estimator = LogisticRegression(solver='lbfgs')
				selector = SFS(estimator, k_features=self.feat_limit, forward=False, floating=False, scoring='roc_auc',cv=4, n_jobs=-1)
			elif self.method == 'seq_fwd':
				estimator = LogisticRegression(solver='lbfgs')
				selector = SFS(estimator, k_features=self.feat_limit, forward=True, floating=False, scoring='roc_auc',cv=4, n_jobs=-1)
		else:
			selector = joblib.load(self.model_save_path)
		
		if self.verbose > 2:
			print(selector)
		return selector

	def feature_selection(self,resume=False):
		selector = self.get_model(resume=resume)
		train_X,train_y = self.load_data(split='train')
		columns = train_X.columns

		if not resume:
			train_X = selector.fit_transform(train_X.values,train_y.values.ravel())
		else:
			train_X = selector.transform(train_X.values)
		
		if self.method == 'rfe':
			ranking = selector.ranking_
			idx = np.argsort(ranking)
			feature_ranking = columns[idx]
			ranking = ranking[idx]
			print('RANKING : ',[x for x in zip(ranking,feature_ranking)])
		
		if not resume:
			joblib.dump(selector,self.model_save_path)
		
		if self.method in ('seq_bwd','seq_fwd'):
			feature_idx = np.array([False for x in columns])
			idx = np.array(list(selector.k_feature_idx_))
			feature_idx[idx] = True
		else:
			feature_idx = selector.get_support()

		sel_columns = columns[feature_idx]
		removed_feats = np.setdiff1d(columns, sel_columns)


		# print('Total Features %d | %s'%(len(columns),columns))
		print('Selected Features %d | %s'%(len(sel_columns),sel_columns))
		print('Removed Features %d | %s'%(len(removed_feats),removed_feats))

if __name__ == '__main__':

	feat_sel = FeatureSelector(method='variance')
	feat_sel.feature_selection(resume=False)

	feat_sel = FeatureSelector(method='forward')
	feat_sel.feature_selection(resume=False)

	feat_sel = FeatureSelector(method='rfe',feat_limit=15)
	feat_sel.feature_selection(resume=False)

	feat_sel = FeatureSelector(method='seq_bwd',feat_limit=15)
	feat_sel.feature_selection(resume=False)

	feat_sel = FeatureSelector(method='seq_fwd',feat_limit=15)
	feat_sel.feature_selection(resume=False)