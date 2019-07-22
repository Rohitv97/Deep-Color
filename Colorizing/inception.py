import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate, Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
from keras import backend as k
import numpy as np
import os
import random
import tensorflow as tf

#Load weights for all
#Load weights
def color_inception():
    print("--------LOADING InceptionResNetV2--------")
    inception = InceptionResNetV2(weights=None, include_top=True)
    inception.load_weights('inception_model.h5')
    inception.graph = tf.get_default_graph()

    embed_input = Input(shape=(1000,))

    print("-------CREATING MODEL-------")

#Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

#Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)

#Decoder
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

    model.load_weights("real_model.h5")

    print("-------MODEL CREATED------")

#give input to Classifier
    def create_inception_embedding(grayscaled_rgb):
        grayscaled_rgb_resized = []
        for i in grayscaled_rgb:
            i = resize(i, (299, 299, 3), mode='constant')
            grayscaled_rgb_resized.append(i)
        grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
        grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
        with inception.graph.as_default():
            embed = inception.predict(grayscaled_rgb_resized)
        return embed

#Make predictions on validation images
# Change to '/data/images/Test/' to use all the 500 test images

    print("-------loading images-----")
    color_me = []
    for filename in os.listdir('static/color/'):
        color_me.append(img_to_array(load_img('static/color/'+filename)))
    color_me = np.array(color_me, dtype=float)
    color_me = 1.0/255*color_me
    color_me = gray2rgb(rgb2gray(color_me))
    color_me_embed = create_inception_embedding(color_me)
    color_me = rgb2lab(color_me)[:,:,:,0]
    color_me = color_me.reshape(color_me.shape+(1,))

    print("-------ADDING SPLASHES HERE AND THERE---------")

# Test model
    output = model.predict([color_me, color_me_embed])
    output = output * 128

    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = color_me[i][:,:,0]
        cur[:,:,1:] = output[i]
        imsave("static/inc/img_inception_"+str(i)+".png", lab2rgb(cur))

    k.clear_session()
    print("------TADDAA------")
