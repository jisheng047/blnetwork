#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import splitext
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from tensorflow.python.keras import models
from keras.backend.tensorflow_backend import _to_tensor
import keras.backend as K
import keras.losses
from keras.utils.generic_utils import get_custom_objects

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

def bce_dice_loss(dice=0.8, bce=0.2, bootstrapping='soft', alpha=1):
        def loss(y,p):
                return dice_coef_loss_bce(y, p, dice, bce, bootstrapping, alpha)
        return loss

def process_test_path(parent_path, file_path):
        img_path = os.path.join(parent_path, file_path)
        prediction_save_path = splitext(img_path)[0] + '_pred' + splitext(img_path)[1]
        return img_path, prediction_save_path

# Load modile
model_path = 'speech_balloon_ds2.h5'
bce_dice_loss = bce_dice_loss(dice=0.8, bce=0.2, bootstrapping='soft', alpha=1)
model = models.load_model(model_path, custom_objects={'loss':bce_dice_loss})

# Process test_path
imgs_path = os.listdir("./test")
imgs_path = list(map(lambda path: process_test_path("./test/",path), imgs_path))


for img_path,prediction_save_path in imgs_path:
        img = imread(img_path)
        img_width, img_height, img_depth = img.shape
        img = resize(img, (768,512,3), anti_aliasing=True, preserve_range=True)
        img = np.expand_dims(img, axis=0)
        img = img/255
        p = model.predict(img)
        mask = p[0]
        mask = resize(mask, (img_width, img_height), anti_aliasing=True, preserve_range=True)
        imsave(fname = prediction_save_path, arr = mask[:,:,0])
