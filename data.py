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
dont_include = [
    ("line", "filled", "green"),
    ("square", "filled", "green"),
    ("circle", "filled", "green"),
    
    ("line", "filled", "green"),
    ("square", "filled", "green"),
    ("circle", "filled", "green"),
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
    s = np.clip(magnitudes*2, 0, 1)
    v = np.ones_like(angles)
    
    hsv = np.stack([h, s, v], axis=-1)
    hsv = (hsv * 255).astype(np.uint8)

    i = Image.fromarray(hsv, mode="HSV")
    i = i.convert(mode="RGB")

    rgb = np.asarray(i).astype(np.float) / 255.0

    return rgb

def new_line(d, center_x, center_y, radius, angle, fill, line_width):
    point1x = center_x + radius * np.cos(angle)
    point1y = center_y + radius * np.sin(angle)
    point2x = center_x + radius * np.cos(angle+np.pi)
    point2y = center_y + radius * np.sin(angle+np.pi)
    
    d.line([(point1x, point1y), (point2x,point2y)], fill=fill, width=int(line_width))

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
def shapes(params, draw_size, resize_to, min_radius, max_radius, line_width):
    n_images = len(params)
    image_width = draw_size
    image_height = draw_size
    
    
#     images = np.zeros((len(params), resize_to, resize_to, 3), dtype=np.float)
    images = np.random.random([len(params), resize_to, resize_to, 3])
#     background = np.random.random([len(params), resize_to, resize_to, 3])
    background = np.zeros([len(params), resize_to, resize_to, 3])

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
        
#         img = Image.fromarray(noiseimages[i], "RGB")
        img = Image.new("RGB", (draw_size, draw_size))
        d = ImageDraw.Draw(img)

        fill = white
        
        if shape == "line":
            if line_type == "single":
                new_line(d, center_x, center_y, radius, angle, fill, line_width)
            elif line_type == "filled":
                new_line(d, center_x, center_y, radius, angle, fill, line_width * 4)
            elif line_type == "double":
                center_offset_x = line_width * np.cos(angle + np.pi/2)
                center_offset_y = line_width * np.sin(angle + np.pi/2)
                new_line(d, center_x + center_offset_x, center_y + center_offset_y, radius, angle, fill, line_width)
                new_line(d, center_x - center_offset_x, center_y - center_offset_y, radius, angle, fill, line_width)
                pass
        
        if shape == "tri":
            tri_line_width = line_width * 2
            tri(d, center_x, center_y, radius, angle, fill)
            if line_type != "filled":
                tri(d, center_x, center_y, radius-tri_line_width, angle, black)
                if line_type == "double":
                    tri(d, center_x, center_y, radius-tri_line_width*2-1, angle, fill)
                    tri(d, center_x, center_y, radius-tri_line_width*3-1, angle, black)
        
        elif shape == "square":
            sq_line_width = line_width * 1.41
            square(d, center_x, center_y, radius, angle, fill)
            if line_type != "filled":
                square(d, center_x, center_y, radius-sq_line_width, angle, black)
                if line_type == "double":
                    square(d, center_x, center_y, radius-sq_line_width*2-1, angle, fill)
                    square(d, center_x, center_y, radius-sq_line_width*3-1, angle, black)
        
        elif shape == "circle":
            circle(d, center_x, center_y, radius, angle, fill)
            if line_type != "filled":
                circle(d, center_x, center_y, radius-line_width, angle, black)
                if line_type == "double":
                    circle(d, center_x, center_y, radius-line_width*2-1, angle, fill)
                    circle(d, center_x, center_y, radius-line_width*3-1, angle, black)
        
        img = img.resize((resize_to, resize_to))
        mask = np.asarray(img).astype(np.float) / 255.0
        
        if color == "rainbow":
            images[i] = rbow * mask + background[i] * (1 - mask)
        elif color == "red":
            images[i] = np.array([1, 0, 0]) * mask + background[i] * (1 - mask)
        elif color == "green":
            images[i] = np.array([0, 1, 0]) * mask + background[i] * (1 - mask)
        elif color == "blue":
            images[i] = np.array([0, 0, 1]) * mask + background[i] * (1 - mask)
        elif color == "white":
            images[i] = mask + background[i] * (1 - mask)
        
    return images
            
def example_shapes():
    par = list(all_classes())
    draw_size = 200
    resize_to = 48
    line_width = draw_size / 25
    min_radius = line_width * 6
    max_radius = min_radius * 1.5
    images = shapes(par, draw_size, resize_to, min_radius=min_radius, max_radius=max_radius, line_width=line_width)
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
def create_dataset_obj(x_all, y_all, z_all, n_classes):
    x_all = tf.convert_to_tensor(x_all)
    y_all = tf.convert_to_tensor(y_all)
    z_all = tf.convert_to_tensor(z_all)

    inds = np.random.permutation(len(x_all))
    n_all = len(x_all)
    n_test = len(x_all) // 10
    n_val = len(x_all) // 10
    n_train = n_all - n_test - n_val
    # 80% train : 10% val : 10% test split
    train_indices = inds[:n_train]
    val_indices = inds[n_train:n_train+n_val]
    test_indices =  inds[n_train+n_val:n_train+n_val+n_test]

    return {
        "image_size": x_all.shape[1],
        "n_classes": n_classes,

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


def make_image_dataset(n_x_data, n_z_data=2000, image_size=24, latent_dims=6, pixel_dtype=np.uint8):

    
    n_classes = len(list(all_classes()))
    n_per_class = n_x_data // n_classes
    
    params = [par for par in all_classes()] * n_per_class
    
    class_labels = np.identity(n_classes)
    classes = [class_labels[i] for i, par in enumerate(all_classes())] * n_per_class
    
#     class1_labels = np.identity(len(shape_types))
#     class2_labels = np.identity(len(line_types))
#     class3_labels = np.identity(len(colors))
#     tclasses = [
#         (class1_labels[shape_types.index(shape_type)], class2_labels[line_types.index(line_type)], class3_labels[colors.index(color)]) for shape_type, line_type, color in all_classes()
#     ] * n_per_class
    
    draw_size = 200
    resize_to = image_size
    line_width = draw_size / 25
    min_radius = line_width * 6
    max_radius = min_radius * 1.5
    images = shapes(params, draw_size, resize_to, min_radius=min_radius, max_radius=max_radius, line_width=line_width)

    gaussian_z = tf.random.normal([n_z_data, latent_dims])

    return create_dataset_obj(images, classes, gaussian_z, n_classes)
