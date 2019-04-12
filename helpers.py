import os
import shutil
import itertools
from tqdm import tqdm
import numpy as np

from multiprocessing import Pool,cpu_count
from sklearn.utils import class_weight


def get_cart_prod(*lists):
	return list(itertools.product(*lists))

def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def shift_files(srcs,dsts,move=False):
	if move:
		for src,dst in tqdm(zip(srcs,dsts),total=len(srcs)):
			shutil.move(src,dst)
	else:
		for src,dst in tqdm(zip(srcs,dsts),total=len(srcs)):
			shutil.copy(src,dst)

def get_count(path):
	for root, dirs, files in os.walk(path):
		count = sum([len(f) for r, d, f in os.walk(root)])
		print(root, ':', count)

def calc_class_wts(y):
    return class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

def unique(arr):
    items,counts = np.unique(arr,return_counts=True)
    print(['%s : %d'%(x,y) for x,y in zip(items,counts)])
    return items,counts

def create_folder(path):
	if not os.path.exists(path):
		clear_folder(path)