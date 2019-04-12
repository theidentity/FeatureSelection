import joblib
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import svm
import helpers


class AnomalyDetection():
	def __init__(self, data_src, seed = 42, method='iso_forest', normal_class=0):
		assert method in ('iso_forest','ocsvm')

		self.data_src = data_src
		self.method = method
		self.seed = seed

		self.name = '%s_ano_detect'%(self.method)
		helpers.create_folder('models/')
		helpers.create_folder('logs/')
		self.save_path = 'models/%s.pkl'%(self.name)
		self.logs = 'logs/%s.txt'%(self.name)
		
		self.normal_class = normal_class
		self.verbose=1

	def get_data(self,split):
		assert split in ('train','test')
		df = pd.read_csv('%s%s.csv'%(self.data_src,split))
		X = df[[x for x in df.columns if x not in ('serial_number','failure')]]
		y = df['failure']
		self.columns = X.columns

		if split in ('valid'):
			X = X[y==self.normal_class]
			y = y[y==self.normal_class]

		if self.verbose > 0:
			print(X.shape,y.shape)
			helpers.unique(y)
		return X,y

	def get_model(self):
		if self.method == 'iso_forest':
			model = ensemble.IsolationForest(n_estimators=500, max_features=1.0, contamination= .1, bootstrap=False, n_jobs=32, behaviour='new', random_state=self.seed, verbose=1)
		elif self.method == 'ocsvm':
			model = svm.OneClassSVM(kernel='rbf', degree=3, tol=0.001, nu=0.5, verbose=1, max_iter=-1, random_state=self.seed)
		return model

	def train(self):
		model = self.get_model()
		X,y = self.get_data(split='train')
		model.fit(X,y)
		joblib.dump(model,self.save_path)

	def get_pred(self):
		X,y_true = self.get_data(split='test')
		model = joblib.load(self.save_path)
		y_pred = model.predict(X)
		helpers.unique(y_pred)
		y_pred = (y_pred == -1)*1
		y_scores = -model.decision_function(X)
		return y_scores,y_pred,y_true

	def evaluate(self):
		y_scores,y_pred,y_true = self.get_pred()
		cm = metrics.confusion_matrix(y_true,y_pred)
		rep = metrics.classification_report(y_true,y_pred)
		auc = metrics.roc_auc_score(y_true,y_scores)
		f1 = metrics.f1_score(y_true,y_pred)
		results = {'name':self.name,'cm':cm,'rep':rep,'auc':auc,'f1':f1}

		with open(self.logs,'w+') as file:
			for key in results.keys():
				res = results[key]
				print(key,res)
				print(key,res,file=file)

if __name__ == '__main__':
	ano = AnomalyDetection(data_src='data/orig/',method='iso_forest',normal_class=0, seed=42)
	ano.train()
	ano.evaluate()

	ano = AnomalyDetection(data_src='data/orig/',method='ocsvm',normal_class=0, seed=42)
	ano.train()
	ano.evaluate()


