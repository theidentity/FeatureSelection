import os
import shutil
import itertools
from tqdm import tqdm


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

