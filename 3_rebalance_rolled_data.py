import pandas as pd
import numpy as np
import helpers
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours

class DataBalancer():
	def __init__(self,data_src, dst_dir, seed=42):
		self.data_src = data_src
		self.dst = dst_dir
		self.verbose = 1

		self.columns = None
		self.seed = seed

	def load_data(self,split):
		assert split in ('train','test')
		df = pd.read_csv('%s%s.csv'%(self.data_src,split))
		X = df[[x for x in df.columns if x not in ('serial_number','failure')]]
		y = df['failure']
		self.columns = X.columns
		if self.verbose > 0:
			print(X.shape,y.shape)
			helpers.unique(y)
		return X,y

	def get_samplers(self,method):
		samplers = {
		'none':None,
		# oversampling
		'smote':SMOTE(random_state=self.seed,n_jobs=32),
		'adasyn':ADASYN(random_state=self.seed,n_jobs=32),
		# undersampling
		'cluser':ClusterCentroids(random_state=self.seed,n_jobs=32),
		'near_miss':NearMiss(random_state=self.seed,n_jobs=32),
		# combined
		'smoteenn':SMOTEENN(random_state=self.seed,smote=SMOTE(random_state=self.seed,n_jobs=16),enn=EditedNearestNeighbours(n_jobs=16)),
		'smotetomek':SMOTETomek(random_state=self.seed,smote=SMOTE(random_state=self.seed,n_jobs=16),tomek=TomekLinks(n_jobs=16)),
		}
		return samplers[method]

	def resample(self,method,split):
		X,y = self.load_data(split=split)
		sampler = self.get_samplers(method=method)
		if sampler is not None:
			X,y = sampler.fit_resample(X,y)
		if self.verbose > 0:
			print(method,split,X.shape,y.shape)
			helpers.unique(y)
		return X,y

	def save_data(self,X,y,name):
		helpers.create_folder(self.dst)
		y = y.reshape(-1,1)
		data = np.hstack((X,y))
		cols = np.hstack([self.columns,'failure'])
		df = pd.DataFrame(data=data,columns=cols)
		df.to_csv('%s%s.csv'%(self.dst,name),index=False)
		print(df.shape)
		helpers.unique(df['failure'])

	def rebalance_data(self,split,method):
		X,y = self.resample(method=method,split=split)
		self.save_data(X,y,name='%s_%s'%(method,split))

if __name__ == '__main__':
	bal = DataBalancer(data_src='data/windowed/',dst_dir='data/windowed_balanced/',seed=42)
	bal.rebalance_data(split='train',method='smotetomek')
	bal.rebalance_data(split='test',method='none')