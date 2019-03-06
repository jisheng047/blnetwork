#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:13:55 2019

@author: DD

Script to load the balloon detection model and an image and save
a prediction as an image.
"""
from os.path import splitext

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from tensorflow.python.keras import models
# import tensorflow as tf

# img_path = './train/images/balloon_00002.jpg'
img_path = './train/images/balloon_00001.jpg'
# model_path = 'speech_balloon.h5'
model_path = 'speech_balloon.h5'
#save path for prediction is based on the path of the image amd will be
#saved in the same directory
prediction_save_path = splitext(img_path)[0] + '_pred' + splitext(img_path)[1]

#model = models.load_model(model_path)
model = models.load_model(model_path)

img = imread(img_path)
img_width, img_height, img_depth = img.shape
img = resize(img, (768,512,3), anti_aliasing=True, preserve_range=True)
img = np.expand_dims(img, axis=0)
img = img/255

p = model.predict(img)
# p = p[0]
print(p.shape)
# for lines in p:
# 	for line in lines:
# 		print(line)
# p = resize(p, (img_width, img_height, 1), anti_aliasing=True, preserve_range=True)
imsave(fname = prediction_save_path, arr = p[0,:,:,0])
