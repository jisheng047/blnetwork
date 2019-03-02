import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
plt.style.use("ggplot")
#%matplotlib inline

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


IMG_WIDTH = 768
IMG_HEIGHT = 512
TRAIN_PATH = os.join(os.getcwd()+ "/train/")
TEST_PATH = os.join(os.getcwd()+ "/test/")

def get_data(path, train=True):
	ids = next(os.walk(path + "images"))[2]
	X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32) # grayscale image
	if train:
		y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32) # grayscale image
	print('Getting and resizing images ... ')
	for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):    
		# Load images
		img = load_img(path + '/images/' + id_, grayscale=True)
		x_img = img_to_array(img)
		x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)

        # Load masks
		if train:
			mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))
			mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)
		# Save images
		X[n, ..., 0] = x_img.squeeze() / 255
		if train:
			y[n] = mask / 255
	print('Done!')
	if train:
		return X, y
	else:
		return X

def input_arg():
	parser = argparse.ArgumentParser(description='Input path for images, labels')
	parser.add_argument('--images', help='Input path for images', required=True)
	parser.add_argument('--labels', help='Input path for labels', required=True)
	args = parser.parse_args()
	args = vars(args)
	return args

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
	

def show_data(X_train):
	# Check if training data looks all right
	ix = random.randint(0, len(X_train))
	has_mask = y_train[ix].max() > 0

	fig, ax = plt.subplots(1, 2, figsize=(20, 10))

	ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
	if has_mask:
		ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
	ax[0].set_title('Seismic')

	ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
	ax[1].set_title('Salt');

if __name__ == "__main__":
	args = input_arg()
	X, y = get_data(args[""])
	fX_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2019)
