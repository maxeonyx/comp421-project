import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, MaxPooling2D, AveragePooling2D, UpSampling2D


def classifier(image_shape, num_classes):
    inputs = Input(image_shape)
    outputs = inputs

    outputs = Conv2D(
        filters=4,
        kernel_size=[5, 5],
        strides=[1, 1],
        activation='relu',
        padding='same',
    )(outputs)
    outputs = AveragePooling2D(pool_size=[2, 2])(outputs)

    outputs = Conv2D(
        filters=8,
        kernel_size=[5, 5],
        strides=[1, 1],
        activation='relu',
        padding='same',
    )(outputs)
    outputs = AveragePooling2D(pool_size=[2, 2])(outputs)

    outputs = Conv2D(
        filters=16,
        kernel_size=[5, 5],
        strides=[1, 1],
        activation='relu',
        padding='same',
    )(outputs)
    outputs = AveragePooling2D(pool_size=[2, 2])(outputs)

    outputs = Conv2D(
        filters=32,
        kernel_size=[3, 3],
        strides=[1, 1],
        activation='relu',
        padding='same',
    )(outputs)
    outputs = AveragePooling2D(pool_size=[2, 2])(outputs)

    outputs = Conv2D(
        filters=64,
        kernel_size=[3, 3],
        strides=[1, 1],
        activation='relu',
        padding='same',
    )(outputs)


    outputs = Flatten()(outputs)

    outputs = Dense(num_classes, activation='relu')(outputs)
    outputs = Dense(num_classes)(outputs)
    outputs = keras.layers.Softmax()(outputs)

    return Model(inputs, outputs)

def autoencoder(image_size, latent_dims):

    enc_inputs = Input(shape=[image_size, image_size, 3])
    enc_outputs = enc_inputs
    
    enc_outputs = Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], activation="relu", padding="same")(enc_outputs)
    enc_outputs = AveragePooling2D(pool_size=(2, 2))(enc_outputs)
    
    enc_outputs = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], activation="relu", padding="same")(enc_outputs)
    enc_outputs = AveragePooling2D(pool_size=(2, 2))(enc_outputs)
    
    enc_outputs = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], activation="relu", padding="same")(enc_outputs)
    enc_outputs = AveragePooling2D(pool_size=(2, 2))(enc_outputs)
    
    enc_outputs = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], activation="relu", padding="same")(enc_outputs)
    enc_outputs = AveragePooling2D(pool_size=(2, 2))(enc_outputs)
    
    enc_outputs = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], activation="relu", padding="same")(enc_outputs)
    
    last_conv_layer = enc_outputs

    enc_outputs = Flatten()(enc_outputs)
    flat_layer = enc_outputs
#     enc_outputs = Dense(enc_outputs.shape[1], activation="relu")(enc_outputs)
    enc_outputs = Dense(enc_outputs.shape[1], activation="relu")(enc_outputs)
    enc_outputs = Dense(latent_dims, activation="linear")(enc_outputs)
    
    # limit the range of the latent space so that we know the boundaries when visualizing and sampling
#     enc_outputs = K.clip(enc_outputs, -4, 4)
    
    encoder = Model(inputs=enc_inputs, outputs=enc_outputs)
    encoder.summary()

    dec_inputs = Input(shape=[latent_dims])
    dec_outputs = dec_inputs

    dec_outputs = Dense(flat_layer.shape[1], activation="relu")(dec_outputs)
#     dec_outputs = Dense(flat_layer.shape[1], activation="relu")(dec_outputs)
#     dec_outputs = Dense(flat_layer.shape[1], activation="relu")(dec_outputs)i
    dec_outputs = Reshape(last_conv_layer.shape[1:])(dec_outputs)
                       
    dec_outputs = Conv2D(filters=64, kernel_size=[3, 3], activation="relu", padding="same")(dec_outputs)
    dec_outputs = UpSampling2D(2)(dec_outputs)
                       
    dec_outputs = Conv2D(filters=32, kernel_size=[3, 3], activation="relu", padding="same")(dec_outputs)
    dec_outputs = UpSampling2D(2)(dec_outputs)
                       
    dec_outputs = Conv2D(filters=16, kernel_size=[3, 3], activation="relu", padding="same")(dec_outputs)
    dec_outputs = UpSampling2D(2)(dec_outputs)
                       
    dec_outputs = Conv2D(filters=8, kernel_size=[3, 3], activation="relu", padding="same")(dec_outputs)
    dec_outputs = UpSampling2D(2)(dec_outputs)
    
    dec_outputs = Conv2D(filters=3, kernel_size=[3, 3], activation="relu", padding="same")(dec_outputs)
    
    # limit the range of the output so that they are always valid images
#     dec_outputs = keras.backend.clip(dec_outputs, 0, 1)

    decoder = Model(inputs=dec_inputs, outputs=dec_outputs)
    decoder.summary()

    latent = encoder(enc_inputs)
    generated = decoder(latent)

    full_model = keras.Model(inputs=enc_inputs, outputs=[generated])

    return full_model, encoder, decoder
