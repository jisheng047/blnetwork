import os
import random
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import argparse
from PIL import Image
import numpy as np
from sklearn.utils import class_weight
#plt.style.use("ggplot")
#%matplotlib inline

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.backend.tensorflow_backend import _to_tensor


IMG_WIDTH = 768
IMG_HEIGHT = 512
TRAIN_PATH = os.path.join(os.getcwd()+ "/train/")
TEST_PATH = os.path.join(os.getcwd()+ "/test/")

def dice_coef(y_true, y_pred, smooth=1.0):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	coef = (2. * intersection + smooth ) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
	return coef

def dice_coef_loss(y_true, y_pred):
	return 1-dice_coef(y_true, y_pred)

def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type='hard', alpha=0.95):
	target_tensor = y_true
	prediction_tensor = y_pred
	_epsilon = _to_tensor(K.epsilon(), prediction_tensor.dtype.base_dtype)
	prediction_tensor = K.tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon)
	prediction_tensor = K.tf.log(prediction_tensor / (1 - prediction_tensor))

	if bootstrap_type == 'soft':
		bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.sigmoid(prediction_tensor)
	else:
		bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.cast(
		K.tf.sigmoid(prediction_tensor) > 0.5, K.tf.float32)
	return K.mean(K.tf.nn.sigmoid_cross_entropy_with_logits(labels=bootstrap_target_tensor, logits=prediction_tensor))


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5, bootstrapping='hard', alpha=1.):
	return bootstrapped_crossentropy(y_true, y_pred, bootstrapping, alpha) * bce + dice_coef_loss(y_true, y_pred) * dice


def get_data_training(path, train=True):
	ids = next(os.walk(path + "images"))[2]
	ids = list(map(lambda item: item[:-4],ids))
	X = np.zeros((len(ids), IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.float32)
	if train:
		y = np.zeros((len(ids), IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.float32)
	for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):    
		mask = np.load(path + '/mask/' + id_ + '.npy')
		x_img = imread(path + '/images/' + id_ + '.jpg')
		x_img = resize(x_img, (IMG_WIDTH, IMG_HEIGHT), mode='constant', preserve_range=True)
		mask = resize(mask, (IMG_WIDTH, IMG_HEIGHT,1), mode='constant', preserve_range=True)
		mask[mask >= 0.8] = 1.
		mask[mask < 0.8] = 0.
		X[n] = x_img / 255.
		if train:
			y[n] = mask
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
	parser.add_argument('--path_train', help='Input path for dataset training images/mask', required=True)
	parser.add_argument('--path_valid', help='Input path for dataset valid images/mask', required=True)
	args = parser.parse_args()
	args = vars(args)
	return args

def bce_dice_loss(dice=0.8, bce=0.2, bootstrapping='soft', alpha=1):
	def loss(y,p):
		return dice_coef_loss_bce(y, p, dice, bce, bootstrapping, alpha)
	return loss

if __name__ == "__main__":
	args = input_arg()
	X_train, y_train = get_data_training(args["path_train"])
	X_valid, y_valid = get_data_training(args["path_valid"])
	model = load_model('./')
	optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
	bce_dice_loss = bce_dice_loss(dice=0.8, bce=0.2, bootstrapping='soft', alpha=1)
	# model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[dice_coef])
	# model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coef])
	model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=["accuracy"])
	filepath = 'speech_balloon_v1.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	results = model.fit(X_train, y_train, batch_size=4, epochs=50, validation_data=(X_valid, y_valid),callbacks =callbacks_list)

