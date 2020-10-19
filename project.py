import importlib

import data as d
importlib.reload(d)
import util as u
importlib.reload(u)
import models as m
importlib.reload(m)

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


image_size = 48

def display_classes():
    image = np.zeros([1, image_size*3, image_size*3], dtype=np.uint8)
    d.line(image)
    u.display_uint8_image(image[0])

    image = np.zeros([1, image_size*3, image_size*3], dtype=np.uint8)
    d.rect(image)
    u.display_uint8_image(image[0])

    image = np.zeros([1, image_size*3, image_size*3], dtype=np.uint8)
    d.circle(image)
    u.display_uint8_image(image[0])

    image = np.zeros([1, image_size*3, image_size*3], dtype=np.uint8)
    d.triangle(image)
    u.display_uint8_image(image[0])

def display_dataset():
    latent_dims = 6
    data = d.make_image_dataset(n_x_data=100000, image_size=image_size)

    # random selection of images from each of the 4 classes
    indices = np.concatenate([
               np.random.randint(i * data["n_all"] // 4, (i+1) * data["n_all"] // 4, 100) for i in range(4)
    ]) 
    images = tf.gather(data["x_all"], indices)
    reshaped = tf.reshape(images, [20, 20, image_size, image_size])
    u.display_many_images(reshaped)

    return data

def make_infinite_dataset(n, sets, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(sets).repeat().shuffle(n).batch(batch_size)

    return dataset

def train_classifier(data):

    image_shape = [image_size, image_size]
    num_classes = 4
    training_steps = 10000
    steps_per_epoch = 100
    epochs = training_steps // steps_per_epoch

    model = m.classifier(image_shape, num_classes)

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    dataset = make_infinite_dataset(data["n_train"], (data["x_train"], data["y_train"]))

    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)

    return model

def reverse_model(model, targets, target_type="channel"):

    inputs = model.inputs

    if target_type == "conv_neuron":
        layer, x, y, channel = targets
        outputs = model.layers[layer].output[0, x, y, channel]
    elif target_type == "conv_channel":
        layer, channel = targets
        outputs = model.layers[layer].output[0, :, :, channel]
    elif target_type == "dense_neuron":
        layer, neuron = targets
        outputs = model.layers[layer].output[0, neuron]
    
    print("Model outputs:", outputs)

    return Model(inputs, outputs)

def layer_stuff(model, data):
    for i, layer in enumerate(model.layers):
        print("Layer", i, ":", layer.name)
        print("Layer", i, "shape", layer.output.shape)
    print()

    print("Neuron")
    print(model.layers[3].output[0, 1, 1, 0])
    print("Channel")
    print(model.layers[3].output[0, :, :, 0])
    print()

    print("Image")
    img = data["x_all"][0]
    u.display_float32_image(img)
    img = tf.expand_dims(img, 0)
    print()
    
    rev_model = reverse_model(model, [3, 1, 1, 0], "conv_neuron")
    # print(rev_model)
    print("Activation of neuron [3, 1, 1, 0]")
    print(rev_model(img))

    return rev_model

@tf.function
def loss_fn(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    activations = model(img_batch)

    return tf.reduce_mean(activations)

def viz_by_opt(model, options=[]):

    img_shape = tf.constant([image_size, image_size])
    iw, ih = img_shape

    regularizers = []

    if "translate" in options:

        translate_dist = 10

        img_shape += translate_dist

        def make_regularizer_step():

            tx, ty = tf.random.uniform([2], 0, translate_dist, dtype=tf.int32)
            # print(tx.numpy(), ty.numpy())

            def transform_img(img):
                return img[tx:tx+image_size, ty:ty+image_size]
        
            def inverse_transform_gradient(grad):
                before_x = tx
                after_x = translate_dist - tx - 1
                before_y = ty
                after_y = translate_dist - ty - 1
                grad = tf.pad(grad, [[before_x, after_x], [before_y, after_y], [0,0]])
                return grad
            return transform_img, inverse_transform_gradient

        regularizers.append(make_regularizer_step)
    
    img_paramaterization = tf.random.uniform(img_shape, 0, 1)
    
    for img_paramaterization in viz_loop(img_paramaterization, model, regularizers=regularizers):
        yield img_paramaterization


def viz_loop(img_paramaterization, model, regularizers=[]):

    step_size = 0.01
    n_training_steps = 2 ** 12 + 1

    for i in tf.range(n_training_steps):
        regularizer_steps = [r() for r in regularizers]
        with tf.GradientTape() as tape:
            tape.watch(img_paramaterization)
            img = img_paramaterization
            for before, after in regularizer_steps:
                img = before(img)
            # print("imageshape", img.shape)
            # print("imageparamshape", img_paramaterization.shape)
            loss = loss_fn(img, model)
        gradient = tape.gradient(loss, img_paramaterization)
        # print("gradientshape", gradient.shape)
        # for before, after in reversed(regularizer_steps):
        #     gradient = after(gradient)
        # normalize gradient
        gradient /= tf.math.reduce_std(gradient) + 1e-8
        img_paramaterization = img_paramaterization + gradient * step_size
        img_paramaterization = tf.clip_by_value(img_paramaterization, 0, 1)

        yield img_paramaterization

def train_autoencoder(data):

    image_shape = [image_size, image_size]
    training_steps = 10000
    steps_per_epoch = 100
    epochs = training_steps // steps_per_epoch

    model, encoder, decoder = m.autoencoder(image_size, latent_dims=6)

    model.summary()

    model.compile(optimizer='adam', loss='mse')

    dataset = make_infinite_dataset(data["n_train"], (data["x_train"], data["x_train"]))

    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)

    return model, encoder, decoder
