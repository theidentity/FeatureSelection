import pandas as pd
from sklearn.model_selection import train_test_split
import helpers
import numpy as np


verbose = 1

def load_orig_data(split):
	assert split in ('train','test')
	df = pd.read_csv('data/orig/%s.csv'%(split))
	df = df.dropna()
	X = df.loc[:, df.columns != 'failure']
	y = df.loc[:, df.columns == 'failure']
	if verbose > 1:
		print(X.columns)
		print(X.shape,y.shape)
		print(np.unique(y,return_counts=True))
	return X,y

def split_train_valid():
	X,y = load_orig_data(split='train')
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)
	if verbose > 1:
		print(X_train.shape,y_train.shape)
		print(X_valid.shape,y_valid.shape)
	return X_train,y_train,X_valid,y_valid

def save_csvs(target_dir='data/prep_data/'):
	helpers.clear_folder(target_dir)
	X_train,y_train,X_valid,y_valid = split_train_valid()
	X_test, y_test = load_orig_data(split='test')
	data = {'train':(X_train,y_train),'valid':(X_valid,y_valid),'test':(X_test,y_test)}
	for split in ['train','valid','test']:
		X,y = data[split]
		X.to_csv('%s%s_X.csv'%(target_dir,split),index=False)
		y.to_csv('%s%s_y.csv'%(target_dir,split),index=False)
		if verbose > 0:
			print(split,X.shape,np.unique(y,return_counts=True))


if __name__ == '__main__':
	# for split in ['train','test']:
		# load_orig_data(split)
	save_csvs()