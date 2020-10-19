import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import itertools

from IPython.display import display

shape_types = [
    "line",
    "square",
    "circle",
    "tri",
]
line_types = [
    "single",
    "double",
    "filled",
]
colors = [
    "white",
    "blue",
    "green",
    "red",
    "rainbow",
]

def all_classes():
    return itertools.product(shape_types, line_types, colors)

def rainbow(shape, center_x, center_y, angle):
    rx = np.linspace(-1, 1, shape[0])
    ry = np.linspace(-1, 1, shape[1])
    coords = np.stack(np.meshgrid(rx, ry), axis=-1)
    angles = np.arctan2(coords[:, :, 0] - center_x, coords[:, :, 1] - center_y) + angle
    magnitudes = np.linalg.norm(coords, axis=-1)
    
    h = angles / (2*np.pi) + 0.5
    s = np.ones(h.shape)
    v = np.ones(h.shape)
    
    hsv = np.stack([h, s, v], axis=-1)
    hsv = (hsv * 255).astype(np.uint8)

    i = Image.fromarray(hsv, mode="HSV")
    i = i.convert(mode="RGB")

    rgb = np.asarray(i).astype(np.float) / 255.0

    return rgb

def square(d, center_x, center_y, radius, angle, fill):
    point1x = center_x + radius * np.cos(angle)
    point1y = center_y + radius * np.sin(angle)
    point2x = center_x + radius * np.cos(angle+np.pi/2)
    point2y = center_y + radius * np.sin(angle+np.pi/2)
    point3x = center_x + radius * np.cos(angle+np.pi)
    point3y = center_y + radius * np.sin(angle+np.pi)
    point4x = center_x + radius * np.cos(angle-np.pi/2)
    point4y = center_y + radius * np.sin(angle-np.pi/2)
    
    d.polygon([(point1x, point1y), (point2x,point2y), (point3x,point3y), (point4x,point4y)], fill=fill)

def tri(d, center_x, center_y, radius, angle, fill):
    point1x = center_x + radius * np.cos(angle)
    point1y = center_y + radius * np.sin(angle)
    point2x = center_x + radius * np.cos(angle+2*np.pi/3)
    point2y = center_y + radius * np.sin(angle+2*np.pi/3)
    point3x = center_x + radius * np.cos(angle-2*np.pi/3)
    point3y = center_y + radius * np.sin(angle-2*np.pi/3)
    
    d.polygon([(point1x, point1y), (point2x,point2y), (point3x,point3y)], fill=fill)

def circle(d, center_x, center_y, radius, angle, fill):
    d.ellipse([(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)], fill=fill)

# assumes 3 channels
def shapes(images, params, min_radius, max_radius, line_width):
    n_images = images.shape[0]
    image_width = images.shape[1]
    image_height = images.shape[2]

    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    black = (0, 0, 0)
    rbow = rainbow(images[0].shape, 0, 0, 0)

    for i in range(n_images):
        shape, line_type, color = params[i]
        radius = np.random.uniform(min_radius, max_radius)
        angle = np.random.uniform(-np.pi, np.pi)

        # leave 2 pixels at the edge
        center_x = np.random.uniform(0+radius+2, image_width-radius-2)
        center_y = np.random.uniform(0+radius+2, image_height-radius-2)
        angle = np.random.uniform(-np.pi, np.pi)
        
        img = Image.new("RGB", (image_width, image_height))
        d = ImageDraw.Draw(img)

        fill = white
        if color == "red":
            fill = red
        elif color == "green":
            fill = green
        elif color == "blue":
            fill = blue
        
        if shape == "tri":
            print("tri")
            tri(d, center_x, center_y, radius, angle, fill)
            if line_type != "filled":
                tri(d, center_x, center_y, radius-line_width, angle, black)
                if line_type == "double":
                    tri(d, center_x, center_y, radius-line_width*2-1, angle, fill)
                    tri(d, center_x, center_y, radius-line_width*3-1, angle, black)
        
        elif shape == "square":
            square(d, center_x, center_y, radius, angle, fill)
            if line_type != "filled":
                square(d, center_x, center_y, radius-line_width, angle, black)
                if line_type == "double":
                    square(d, center_x, center_y, radius-line_width*2-1, angle, fill)
                    square(d, center_x, center_y, radius-line_width*3-1, angle, black)
        
        elif shape == "circle":
            circle(d, center_x, center_y, radius, angle, fill)
            if line_type != "filled":
                circle(d, center_x, center_y, radius-line_width, angle, black)
                if line_type == "double":
                    circle(d, center_x, center_y, radius-line_width*2-1, angle, fill)
                    circle(d, center_x, center_y, radius-line_width*3-1, angle, black)

        display(img)

        imgdata = np.asarray(img)
        
        if color == "rainbow":
            x = imgdata.astype(np.float) / 255.0
            x *= rbow
            imgdata = (x * 255).astype(np.uint8)
        
        rgbx = imgdata.astype(np.float) / 255.0
        rgb = rgbx[:, :, :3]
        images[i] = rgb
            
def example_shapes():
    par = list(all_classes())
    images = np.zeros((len(par), 100, 100, 3), dtype=np.float)
    shapes(images, par, min_radius=30, max_radius=50, line_width=3)
    return images

def line(images, min_length=48, max_length=48):
    n_images = images.shape[0]
    image_width = images.shape[1]
    image_height = images.shape[2]
    lengths = np.random.uniform(min_length, max_length, n_images)
    angles = np.random.uniform(-np.pi, np.pi, n_images)
    widths = lengths * np.cos(angles)
    heights = lengths * np.sin(angles)

    x_lows = np.clip(-widths+1, 1, image_width-1)
    x_highs = np.clip(image_width-widths-1, 1, image_width-1)
    y_lows = np.clip(-heights+1, 1, image_height-1)
    y_highs = np.clip(image_height-heights-1, 1, image_height-1)

    starts = np.random.uniform(np.stack([x_lows, y_lows], axis=1), np.stack([x_highs, y_highs], axis=1), [n_images, 2])

    ends = starts + np.stack([widths, heights], axis=1)

    starts = starts.astype(np.uint32)
    ends = ends.astype(np.uint32)

    for i in range(n_images):
        imgdata = images[i]
        img = Image.frombuffer("L", imgdata.shape, imgdata)
        img.readonly = False
        d = ImageDraw.Draw(img)
        d.line([tuple(starts[i]), tuple(ends[i])], fill=255, width=6)


def rect(images, min_size=16, max_size=48):
    n_images = images.shape[0]
    image_width = images.shape[1]
    image_height = images.shape[2]
    widths = np.random.uniform(min_size, max_size, n_images)
    heights = np.random.uniform(min_size, max_size, n_images)

    x_lows = np.clip(-widths+1, 1, image_width-1)
    x_highs = np.clip(image_width-widths-1, 1, image_width-1)
    y_lows = np.clip(-heights+1, 1, image_height-1)
    y_highs = np.clip(image_height-heights-1, 1, image_height-1)

    starts = np.random.uniform(np.stack([x_lows, y_lows], axis=1), np.stack([x_highs, y_highs], axis=1), [n_images, 2])
    ends = starts + np.stack([widths, heights], axis=1)

    starts = starts.astype(np.uint32)
    ends = ends.astype(np.uint32)

    for i in range(n_images):
        imgdata = images[i]
        img = Image.frombuffer("L", imgdata.shape, imgdata)
        img.readonly = False
        d = ImageDraw.Draw(img)
        d.rectangle([tuple(starts[i]), tuple(ends[i])], fill=255)

def circleold(images, min_size=32, max_size=48):
    n_images = images.shape[0]
    image_width = images.shape[1]
    image_height = images.shape[2]
    diameters = np.random.uniform(min_size, max_size, n_images)

    x_lows = np.clip(-diameters+1, 1, image_width-1)
    x_highs = np.clip(image_width-diameters-1, 1, image_width-1)
    y_lows = np.clip(-diameters+1, 1, image_height-1)
    y_highs = np.clip(image_height-diameters-1, 1, image_height-1)

    starts = np.random.uniform(np.stack([x_lows, y_lows], axis=1), np.stack([x_highs, y_highs], axis=1), [n_images, 2])
    ends = starts + np.stack([diameters, diameters], axis=1)

    starts = starts.astype(np.uint32)
    ends = ends.astype(np.uint32)

    for i in range(n_images):
        imgdata = images[i]
        img = Image.frombuffer("L", imgdata.shape, imgdata)
        img.readonly = False
        d = ImageDraw.Draw(img)
        d.ellipse([tuple(starts[i]), tuple(ends[i])], fill=0, outline=255, width=6)

def triangle(images, min_size=48, max_size=48):
    n_images = images.shape[0]
    image_width = images.shape[1]
    image_height = images.shape[2]

    # two triangle sides
    lengths = np.random.uniform(min_size, max_size, [n_images, 2])
    # orientation
    directions = np.random.uniform(-np.pi, np.pi, n_images)
    # inner angle, narrow to equilateral
    inner_angles = np.random.uniform(2*np.pi* 1/6, 2*np.pi * 1/6, n_images)

    line1x = lengths[:, 0] * np.cos(directions)
    line1y = lengths[:, 0] * np.sin(directions)
    line2x = lengths[:, 1] * np.cos(directions + inner_angles)
    line2y = lengths[:, 1] * np.sin(directions + inner_angles)

    # bounding box relative to start point
    width_low = np.minimum(0, np.minimum(line1x, line2x))
    width_high = np.maximum(0, np.maximum(line1x, line2x))
    height_low = np.minimum(0, np.minimum(line1y, line2y))
    height_high = np.maximum(0, np.maximum(line1y, line2y))

    x_lows = np.clip(-width_low+1, 1, image_width-1)
    x_highs = np.clip(image_width-width_high-1, 1, image_width-1)
    y_lows = np.clip(-height_low+1, 1, image_height-1)
    y_highs = np.clip(image_height-height_high-1, 1, image_height-1)

    starts = np.random.uniform(np.stack([x_lows, y_lows], axis=1), np.stack([x_highs, y_highs], axis=1), [n_images, 2])

    point1 = starts + np.stack([line1x, line1y], axis=1)
    point2 = starts + np.stack([line2x, line2y], axis=1)

    starts = starts.astype(np.uint32)
    point1 = point1.astype(np.uint32)
    point2 = point2.astype(np.uint32)

    for i in range(n_images):
        imgdata = images[i]
        img = Image.frombuffer("L", imgdata.shape, imgdata)
        img.readonly = False
        d = ImageDraw.Draw(img)
        d.line([tuple(starts[i]), tuple(point1[i])], fill=255, width=6)
        d.line([tuple(point1[i]), tuple(point2[i])], fill=255, width=6)
        d.line([tuple(starts[i]), tuple(point2[i])], fill=255, width=6)

# make a convenient structure for our data
def create_dataset_obj(x_all, y_all, z_all):
    x_all = tf.convert_to_tensor(x_all)
    y_all = tf.convert_to_tensor(y_all)
    z_all = tf.convert_to_tensor(z_all)
    x_all = tf.expand_dims(x_all, -1)

    inds = np.indices([len(x_all)])
    # 80% train : 10% val : 10% test split
    train_indices = inds[inds % 10 >= 2]
    val_indices = inds[inds % 10 == 0]
    test_indices =  inds[inds % 10 == 1]

    return {
        "image_size": x_all.shape[1],

        "n_all": len(x_all),
        "x_all": x_all,
        "y_all": y_all,

        "n_z": len(z_all),
        "z_all": z_all,

        "n_train": len(train_indices),
        "x_train": tf.gather(x_all, train_indices),
        "y_train": tf.gather(y_all, train_indices),

        "n_val": len(val_indices),
        "x_val": tf.gather(x_all, val_indices),
        "y_val": tf.gather(y_all, val_indices),
        
        "n_test": len(test_indices),
        "x_test": tf.gather(x_all, test_indices),
        "y_test": tf.gather(y_all, test_indices),
    }


def make_image_dataset(n_x_data=10000, n_z_data=2000, image_size=24, latent_dims=6, pixel_dtype=np.uint8):

    fns = [
           line,
           rect,
           circleold,
           triangle,
    ]

    n_classes = len(fns)
    n_per_class = n_x_data // n_classes
    images = np.zeros([n_x_data, image_size*3, image_size*3], dtype=pixel_dtype)

    for i, fn in enumerate(fns):
        start = i*n_per_class
        end = (i+1)*n_per_class
        fn(images[start:end])

    class_labels = np.identity(n_classes)
    classes = class_labels[np.concatenate([np.repeat(i, n_per_class) for i in range(n_classes)])]

    gaussian_z = tf.random.normal([n_z_data, latent_dims])

    resized_images = np.zeros([n_x_data, image_size, image_size], dtype=np.float32)
    for i in range(len(images)):
        img = Image.frombuffer("L", images[i].shape, images[i])
        resized = img.resize((image_size, image_size))
        resized_images[i] = np.array(resized).astype(np.float32) / 255.

    return create_dataset_obj(resized_images, classes, gaussian_z)