import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import numpy as np
plt.style.use("ggplot")
#%matplotlib inline

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

IMG_WIDTH = 768
IMG_HEIGHT = 512
TRAIN_PATH = os.path.join(os.getcwd()+ "/train/")
TEST_PATH = os.path.join(os.getcwd()+ "/test/")

def get_data_training(path, train=True):
	ids = next(os.walk(path + "images"))[2]
	ids = list(map(lambda item: item[:-4],ids))
	X = np.zeros((len(ids), IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.float32) # grayscale image
	if train:
		y = np.zeros((len(ids), IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.float32) # grayscale image
	print('Getting and resizing images ... ')
	for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):    
		# Load images
		img = load_img(path + '/images/' + id_ + '.jpg')
		mask = np.load(path + '/mask/' + id_ + '.npy')
		
		x_img = img_to_array(img)
		x_img = resize(x_img, (IMG_WIDTH, IMG_HEIGHT, 3), mode='constant', preserve_range=True)
		mask = resize(mask, (IMG_WIDTH, IMG_HEIGHT, 1), mode='constant', preserve_range=True)
		mask[mask > 0] = 1
		X[n] = x_img.squeeze() / 255
		if train:
			y[n] = mask
	print('Done!')
	if train:
		return X, y
	else:
		return X

def load_model(model_dir):
	yaml_file = open('%s/model_num.yaml' % model_dir, 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	model = tf.keras.models.model_from_yaml(loaded_model_yaml)
	return model
   	
def input_arg():
	parser = argparse.ArgumentParser(description='Input path for images, labels')
	parser.add_argument('--path_ds', help='Input path for dataset images/mask', required=True)
	args = parser.parse_args()
	args = vars(args)
	return args

if __name__ == "__main__":
	args = input_arg()
	X, y = get_data_training(args["path_ds"])
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2019)
	
	# Normalize interval [0,1] for image
	# Data Augmentation 
	# Rotation hue channel of HSV image (range 0 to 0.3)
	# Height and Width shifts (range: 0.2 of each dimension size)
	# Flipping image horizontal and vertical
	# Using Adam (lr = 0.001, beta1 = 0.9, beta2=0.999 , epoch=500)
	# Using BCE - Binary Cross Entropy Loss, Dice - Sorenson-Dice coefficientfor x in X_train:
	# input_img = Input((IMG_WIDTH, IMG_HEIGHT, 3), name='img')

	model = load_model('./')
	# model = get_speechnet(input_img, n_filters=64, dropout=0.05, batchnorm=True)
	optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
	model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
	results = model.fit(X_train, y_train, batch_size=4, epochs=1, validation_data=(X_valid, y_valid))

