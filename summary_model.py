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

def encoding_conv(input_tensor, n_filters, kernel_size=3, batchnorm=True):
	enc_1 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
	if batchnorm:
		enc_1 = BatchNormalization()(enc_1)
	enc_1 = Activation("relu")(enc_1)
	return enc_1

def get_speechnet(input_img, n_filters=64, dropout=0.5, batchnorm=True):
	
	enc_1 = encoding_conv(input_img, n_filters = n_filters*1, kernel_size=3, batchnorm=True)
	enc_1_p = MaxPooling2D((2, 2)) (enc_1)
	enc_1_d = Dropout(dropout*0.5)(enc_1_p)

	enc_2 = encoding_conv(enc_1, n_filters = n_filters*2, kernel_size=3, batchnorm=True)
	enc_2_p = MaxPooling2D((2, 2)) (enc_2)
	enc_2_d = Dropout(dropout*0.5)(enc_2_p)

	enc_3 = encoding_conv(enc_2, n_filters = n_filters*4, kernel_size=3, batchnorm=True)
	enc_3_p = MaxPooling2D((2, 2)) (enc_3)
	enc_3_d = Dropout(dropout*0.5)(enc_3_p)

	enc_4 = encoding_conv(enc_3, n_filters = n_filters*8, kernel_size=3, batchnorm=True)
	enc_4_p = MaxPooling2D((2, 2)) (enc_4)
	enc_4_d = Dropout(dropout*0.5)(enc_4_p)

	middle = encoding_conv(enc_3, n_filters = n_filters*8, kernel_size=3, batchnorm=True)
	middle = MaxPooling2D((2, 2)) (middle)
	middle = Dropout(dropout*0.5)(middle)


	dec_1 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (middle)
	dec_1_c = concatenate([dec_1, enc_4])
	dec_1_d = Dropout(dropout)(dec_1_c)

	dec_2 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (dec_1_d)
	dec_2_c = concatenate([dec_2, enc_3])
	dec_2_d = Dropout(dropout)(dec_2_c)

	dec_3 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (dec_2_d)
	dec_3_c = concatenate([dec_3, enc_2])
	dec_3_d = Dropout(dropout)(dec_3_c)

	dec_4 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (dec_3_d)
	dec_4_c = concatenate([dec_4, enc_1])
	dec_4_d = Dropout(dropout)(dec_4_c)

	outputs = Conv2D(3, (1, 1), activation='sigmoid') (dec_4_d)
	model = Model(inputs=[input_img], outputs=[outputs])
	return model
