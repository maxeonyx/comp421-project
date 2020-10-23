from PIL import Image, ImageDraw
from IPython.display import display
import numpy as np
import tensorflow as tf

def display_uint8_image(image, color=False):
    image = tf.squeeze(image)
    image = image.numpy()
    if color:
        display(Image.fromarray(image, "RGB"))
    else:
        display(Image.fromarray(image, "L"))

def display_float32_image(image, color=False):
    image = tf.squeeze(image)
    image = tf.clip_by_value(image, 0, 1)
    image *= 255
    display_uint8_image(tf.cast(image, tf.uint8), color)

def get_width_height(shape, d=0, color=False):
    width = 1
    height = 1
    color_dim = 1 if color else 0
    for dim in range(d, len(shape) - color_dim):
        if dim % 2 == 0:
            width *= shape[dim]
        else:
            height *= shape[dim]
    return width, height

def image_along_axis(buffer, generated, shape, d, color=False):
    if len(generated.shape) == 3 and color:
        buffer[:, :, :] = generated
    elif len(generated.shape) == 2:
        buffer[:, :] = generated
    else:
        slice_width, slice_height = get_width_height(shape, d+1, color)
        if d % 2 == 0:
            for i in range(shape[d]):
                if color:
                    image_along_axis(buffer[i*slice_width:(i+1)*slice_width, :, :], generated[i], shape, d+1, color)
                else:
                    image_along_axis(buffer[i*slice_width:(i+1)*slice_width, :], generated[i], shape, d+1, color)
        else:
            for i in range(shape[d]):
                if color:
                    image_along_axis(buffer[:, i*slice_height:(i+1)*slice_height, :], generated[i], shape, d+1, color)
                else:
                    image_along_axis(buffer[:, i*slice_height:(i+1)*slice_height], generated[i], shape, d+1, color)

def display_many_images(images, color=False):
    print("images size: ", images.shape)
    width, height = get_width_height(images.shape, color=color)
    color_dim = [3] if color else []
    full_image = np.zeros([width, height, *color_dim])
    print("combined image size: ", full_image.shape)
    image_along_axis(full_image, images, images.shape, 0, color)
    display_float32_image(full_image, color)

# Show n_vecs images along each dimension of the latent space
def display_latent_dims(decoder, latent_dims, range=[0, 1], n_vecs=3, color=False):
    n_dims = latent_dims
    vals = np.linspace(range[0], range[1], n_vecs)
    zs = np.transpose(np.meshgrid(*([vals] * n_dims)))
    flat_zs = zs.reshape([-1, latent_dims])
    flat_generated = decoder(flat_zs)
    if color:
        image_height = flat_generated.shape[-2]
        image_width = flat_generated.shape[-3]
    else:
        image_height = flat_generated.shape[-1]
        image_width = flat_generated.shape[-2]
    color_dim = [3] if color else []
    generated = tf.reshape(flat_generated, [*[n_vecs]*latent_dims, image_width, image_height, *color_dim])
    display_many_images(generated, color)

# display a selection of images and their reconstructions
def display_model_output(images, model, color=False):
    image_size = images.shape[1]
    
    generated = model(images)
#     indices = np.random.permutation(data["n_test"])[:120]
#     input_images = tf.reshape(tf.gather(images, indices), [-1, 15, image_size, image_size, 3])
    images = tf.reshape(images, [-1, 15, image_size, image_size, 3])
#     generated_images = tf.reshape(tf.gather(generated, indices), [-1, 20, image_size, image_size, 3])
    generated = tf.reshape(generated, [-1, 15, image_size, image_size, 3])
    display_many_images(images, color)
    display_many_images(generated, color)
