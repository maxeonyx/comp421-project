import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, MaxPooling2D, UpSampling2D


def classifier(image_shape, num_classes):
    input_shape = [*image_shape, 1] # add channel dim
    inputs = Input(input_shape)
    outputs = inputs

    outputs = Conv2D(
        filters=4,
        kernel_size=[3, 3],
        strides=[1, 1],
        activation='relu'
    )(outputs)
    outputs = MaxPooling2D(pool_size=[2, 2])(outputs)

    outputs = Conv2D(
        filters=8,
        kernel_size=[3, 3],
        strides=[1, 1],
        activation='relu'
    )(outputs)
    outputs = MaxPooling2D(pool_size=[2, 2])(outputs)

    outputs = Conv2D(
        filters=16,
        kernel_size=[3, 3],
        strides=[1, 1],
        activation='relu'
    )(outputs)
    outputs = MaxPooling2D(pool_size=[2, 2])(outputs)

    outputs = Conv2D(
        filters=16,
        kernel_size=[3, 3],
        strides=[1, 1],
        activation='relu'
    )(outputs)
    outputs = MaxPooling2D(pool_size=[2, 2])(outputs)


    outputs = Flatten()(outputs)

    outputs = Dense(num_classes, activation='relu')(outputs)
    outputs = Dense(num_classes)(outputs)
    outputs = keras.layers.Softmax()(outputs)

    return Model(inputs, outputs)

def autoencoder(image_size, latent_dims):

    enc_inputs = Input(shape=[image_size, image_size, 1])
    
    conv1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], activation="relu", padding="same")(enc_inputs)
    maxp1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], activation="relu", padding="same")(maxp1)
    maxp2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], activation="relu", padding="same")(maxp2)
    maxp3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    flat1 = Flatten()(maxp3)
    full1 = Dense(flat1.shape[1], activation="relu")(flat1)
    full2 = Dense(full1.shape[1], activation="relu")(full1)
    full3 = Dense(latent_dims, activation="relu")(full2)
    
    # limit the range of the latent space so that we know the boundaries for the visualization
    enc_output = K.clip(full3, -4, 4)
    
    encoder = Model(inputs=enc_inputs, outputs=enc_output)


    dec_inputs = Input(shape=[latent_dims])

    dfull1 = Dense(full2.shape[1], activation="relu")(dec_inputs)
    dfull2 = Dense(full1.shape[1], activation="relu")(dfull1)
    dfull3 = Dense(flat1.shape[1], activation="relu")(dfull2)
    dresh1 = Reshape(maxp3.shape[1:])(dfull3)
    dconv1 = Conv2D(filters=64, kernel_size=[3, 3], activation="relu", padding="same")(dresh1)
    dupsm1 = UpSampling2D(2)(dconv1)
    dconv2 = Conv2D(filters=32, kernel_size=[3, 3], activation="relu", padding="same")(dupsm1)
    dupsm2 = UpSampling2D(2)(dconv2)
    dconv3 = Conv2D(filters=16, kernel_size=[3, 3], activation="relu", padding="same")(dupsm2)
    dupsm3 = UpSampling2D(2)(dconv3)
    dconv4 = Conv2D(filters=1, kernel_size=[3, 3], activation="relu", padding="same")(dupsm3)
    
    # limit the range of the output so that they are always valid images
    dec_output = keras.backend.clip(dconv4, 0, 1)

    decoder = Model(inputs=dec_inputs, outputs=dec_output)

    latent = encoder(enc_inputs)
    generated = decoder(latent)

    full_model = keras.Model(inputs=enc_inputs, outputs=[generated])

    return full_model, encoder, decoder
