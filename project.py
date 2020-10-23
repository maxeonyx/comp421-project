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

import tensorflow_addons as tfa

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
    data = d.make_image_dataset(n_x_data=24000, image_size=image_size)

    u.display_many_images(tf.reshape(data["x_test"][:100], [5, 20, image_size, image_size, 3]), color=True)

    return data

def make_infinite_dataset(n, sets, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(sets).repeat().shuffle(n).batch(batch_size)

    return dataset

def train_classifier(data):

    image_shape = [image_size, image_size, 3]
    n_classes = data["n_classes"]
    training_steps = 10000
    steps_per_epoch = 100
    epochs = training_steps // steps_per_epoch

    model = m.classifier(image_shape, n_classes)

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])

    dataset = make_infinite_dataset(data["n_train"], (data["x_train"], data["y_train"]))
    val_dataset = make_infinite_dataset(data["n_val"], (data["x_val"], data["y_val"]))

    model.fit(dataset, validation_data=val_dataset, validation_steps=120, epochs=epochs, steps_per_epoch=steps_per_epoch)

    return model

def reverse_model(model, targets, target_type="channel", batch=False):

    inputs = model.inputs

    if target_type == "conv_neuron":
        layer, x, y, channel = targets
        if batch:
            outputs = model.layers[layer].output[:, x, y, channel]
        else:
            outputs = model.layers[layer].output[0, x, y, channel]
    elif target_type == "conv_channel":
        layer, channel = targets
        if batch:
            outputs = model.layers[layer].output[:, :, :, channel]
        else:
            outputs = model.layers[layer].output[0, :, :, channel]
    elif target_type == "dense_neuron":
        layer, neuron = targets
        if batch:
            outputs = model.layers[layer].output[:, neuron]
        else:
            outputs = model.layers[layer].output[0, neuron]
    
#     print("Model outputs:", outputs)

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
    u.display_float32_image(img, color=True)
    img = tf.expand_dims(img, 0)
    print()
    
    rev_model = reverse_model(model, [3, 1, 1, 0], "conv_neuron")
    # print(rev_model)
    print("Activation of neuron [3, 1, 1, 0]")
    print(rev_model(img))

    return rev_model


def loss_fn(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    activations = model(img_batch)

    return tf.reduce_mean(activations)

def viz_by_opt(model, options=[]):
    
    img_par_size = image_size

    transforms = []
    
    if "rotate" in options:
        
#         img_par_size = int(img_par_size * np.sqrt(2))
        
        def rotate_img(img):
            angle = tf.random.uniform(shape=[], minval=0, maxval=2*np.pi)
            return tfa.image.rotate(img, angle)

        transforms.append(rotate_img)

    if "translate" in options:

        translate_dist = 20

        img_par_size += translate_dist
        
        def translate_img(img):
            t = tf.random.uniform([2], 0, translate_dist, dtype=tf.int32)
            return img[t[0]:t[0]+image_size, t[1]:t[1]+image_size]

        transforms.append(translate_img)
    
    
    if "clip" in options:
        
        def clip(img):
            return tf.clip_by_value(img, 0, 1)

        transforms.append(clip)
    
    def transform(x):
        for t in transforms:
            x = t(x)
        return x
    
    img_par_shape = tf.constant([img_par_size, img_par_size, 3])
    img_paramaterization = tf.random.uniform(img_par_shape, 0, 1)
#     print(img_paramaterization.shape)
    
#     u.display_float32_image(img_paramaterization, color=True)
    
    for img_paramaterization in viz_loop(img_paramaterization, model, transform):
        yield img_paramaterization


def viz_loop(img_paramaterization, model, transform, steps=2049):
    
    img_paramaterization = tf.Variable(img_paramaterization)
    
    step_size = 0.01
    
    def viz_step(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            img = x
            img = transform(img)

            loss = loss_fn(img, model)
        gradient = tape.gradient(loss, x)
        gradient /= tf.math.reduce_std(gradient) + 1e-8
        x = x + gradient * step_size
        x = tf.clip_by_value(x, 0, 1)
        return x

    for i in tf.range(steps):
        
        img_paramaterization = viz_step(img_paramaterization)

        yield img_paramaterization
    
def viz_loop2(img_paramaterization, decoder, part_classifier, steps=1025):
    
    x = tf.Variable(img_paramaterization)

    step_size = 100
    
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    
    def viz_step(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            activations = part_classifier(decoder(x))
            loss = tf.reduce_mean(activations)
        gradient = tape.gradient(loss, x)
        gradient = gradient / (tf.norm(gradient) + 1e-18)
        return x + gradient * step_size
    
    for i in tf.range(steps):
        
        x = viz_step(x)

        yield x

def viz_search(over, part_classifier, k=4):
    
    def viz_search_inner(inps):
        activations = part_classifier(inps)
        reduction_dims = list(range(len(activations.shape)))[1:]
        losses = tf.reduce_mean(activations, axis=reduction_dims)
        return tf.math.top_k(losses, k)
       
    return viz_search_inner(over)

def viz_search2(over, decoder, part_classifier, k=4):
    
    
    def viz_search_inner(inps):
        imgs = decoder(inps)
        activations = part_classifier(imgs)
        reduction_dims = list(range(len(activations.shape)))[1:]
        losses = tf.reduce_mean(activations, axis=reduction_dims)
        return tf.math.top_k(losses, k)
       
    return viz_search_inner(over)

def train_autoencoder(data):

    image_shape = [image_size, image_size]
    training_steps = 100000
    steps_per_epoch = 1000
    epochs = training_steps // steps_per_epoch

    model, encoder, decoder = m.autoencoder(image_size, latent_dims=6)

    model.summary()

    model.compile(optimizer='adam', loss='mse')

    dataset = make_infinite_dataset(data["n_train"], (data["x_train"], data["x_train"]))

    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)

    return model, encoder, decoder
