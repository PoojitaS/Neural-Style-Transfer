#Forward compatible imports
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

#General imports
import sys
import time
import argparse
import warnings
import numpy as np

#Processing & Optimization imports
import scipy.optimize as m
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imread, imresize, imsave, fromimage, toimage

#Deep Learning model imports
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras import backend as K
from keras.utils.data_utils import get_file
from tensorflow.python import debug as tf_debug
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D

"""
-----------------------------------------------------------------------------------
| Neural Style Transfer with Keras 2.0.5                                          |
|                                                                                 |
| Based on:                                                                       |
| https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py |
-----------------------------------------------------------------------------------
"""

"""GLOBAL VARIABLES"""
#Dimensions of the generated picture.
img_width = img_height = 0
img_WIDTH = img_HEIGHT = 0
aspect_ratio = 0

"""--------------------------------------------Provide & set conditions to user for execution-------------------------------------------"""

def str_to_bool(v):
    return v.lower() in ("true", "yes", "t", "1", "y")


#Commandline arguments
parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')

parser.add_argument('syle_image_paths', metavar='ref', nargs='+', type=str,
                    help='Path to the style reference image.')

parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

parser.add_argument("--style_masks", type=str, default=None, nargs='+',
                    help='Masks for style images')

parser.add_argument("--content_mask", type=str, default=None,
                    help='Masks for the content image')

parser.add_argument("--color_mask", type=str, default=None,
                    help='Mask for color preservation')

parser.add_argument("--image_size", dest="img_size", default=400, type=int,
                    help='Minimum image size')

parser.add_argument("--content_weight", dest="content_weight", default=0.025, type=float,
                    help="Weight of content")

parser.add_argument("--style_weight", dest="style_weight", nargs='+', default=[1], type=float,
                    help="Weight of style, can be multiple for multiple styles")

parser.add_argument("--style_scale", dest="style_scale", default=1.0, type=float,
                    help="Scale the weighing of the style")

parser.add_argument("--total_variation_weight", dest="tv_weight", default=8.5e-5, type=float,
                    help="Total Variation weight")

parser.add_argument("--num_iter", dest="num_iter", default=10, type=int,
                    help="Number of iterations")

parser.add_argument("--model", default="vgg16", type=str,
                    help="Choices are 'vgg16' and 'vgg19'")

parser.add_argument("--content_loss_type", default=0, type=int,
                    help='Can be one of 0, 1 or 2. Readme contains the required information of each mode.')

parser.add_argument("--rescale_image", dest="rescale_image", default="False", type=str,
                    help="Rescale image after execution to original dimentions")

parser.add_argument("--rescale_method", dest="rescale_method", default="bilinear", type=str,
                    help="Rescale image algorithm")

parser.add_argument("--maintain_aspect_ratio", dest="maintain_aspect_ratio", default="True", type=str,
                    help="Maintain aspect ratio of loaded images")

parser.add_argument("--content_layer", dest="content_layer", default="conv5_2", type=str,
                    help="Content layer used for content loss.")

parser.add_argument("--init_image", dest="init_image", default="content", type=str,
                    help="Initial image used to generate the final image. Options are 'content', 'noise', or 'gray'")

parser.add_argument("--pool_type", dest="pool", default="max", type=str,
                    help='Pooling type. Can be "ave" for average pooling or "max" for max pooling')

parser.add_argument('--preserve_color', dest='color', default="False", type=str,
                    help='Preserve original color in image')

parser.add_argument('--min_improvement', default=0.0, type=float,
                    help='Defines minimum improvement required to continue script')


''' Arguments '''
args = parser.parse_args()

#RGB or Grayscale?
read_mode = "gray" if args.init_image == "gray" else "color"

#Get tensor representations of our images
#Content, style & pastiche paths
base_image_path = args.base_image_path
style_reference_image_paths = args.syle_image_paths
result_prefix = args.result_prefix

#Option for using multiple styles
style_image_paths = []
for style_image_path in style_reference_image_paths:
    style_image_paths.append(style_image_path)

#Option for using style masks
style_masks_present = args.style_masks is not None
mask_paths = []

if style_masks_present:
    for mask_path in args.style_masks:
        mask_paths.append(mask_path)

if style_masks_present:
    assert len(style_image_paths) == len(mask_paths), "Wrong number of style masks provided.\n" \
                                                      "Number of style images = %d, \n" \
                                                      "Number of style mask paths = %d." % \
                                                      (len(style_image_paths), len(style_masks_present))

#Option for content masks
content_mask_present = args.content_mask is not None
content_mask_path = args.content_mask

#Option for color mask
color_mask_present = args.color_mask is not None

#Image resolution & style constraints
rescale_image = str_to_bool(args.rescale_image)
maintain_aspect_ratio = str_to_bool(args.maintain_aspect_ratio)
preserve_color = str_to_bool(args.color)

def image_component_weights():

    global style_image_paths
    #Weights of the different loss components
    content_weight = args.content_weight
    total_variation_weight = args.tv_weight
    style_weights = []
    if len(style_image_paths) != len(args.style_weight):
        # print("Mismatch in number of style images provided and number of style weights provided. \n"
        #       "Found %d style images and %d style weights. \n"
        #       "Equally distributing weights to all other styles." % (len(style_image_paths), len(args.style_weight)))

        weight_sum = sum(args.style_weight) * args.style_scale

        count = len(style_image_paths)

        for i in range(len(style_image_paths)):
            style_weights.append(weight_sum / count)
    else:
        for style_weight in args.style_weight:
            style_weights.append(style_weight * args.style_scale)

    return style_weights, content_weight, total_variation_weight

#Decide pooling function
pooltype = str(args.pool).lower()
assert pooltype in ["ave", "max"], 'Pooling argument is wrong. Needs to be either "ave" or "max".'

pooltype = 1 if pooltype == "ave" else 0

#Content loss definition
assert args.content_loss_type in [0, 1, 2], "Content Loss Type must be one of 0, 1 or 2"


"""-------------------------------------------------------Build VGG model for NST-------------------------------------------------------"""

def load_model(ip):
    #Weights for pre-trained VGG model with variations
    THEANO_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
    TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    TH_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels_notop.h5'
    TF_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


    #Build the VGG16 network with our 3 images as input
    x = Convolution2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(ip)
    x = Convolution2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(x)
    x = pooling_func(x)

    x = Convolution2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(x)
    x = Convolution2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(x)
    x = pooling_func(x)

    x = Convolution2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(x)
    x = Convolution2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(x)
    x = Convolution2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(x)
    if args.model == "vgg19":
        x = Convolution2D(256, (3, 3), activation='relu', name='conv3_4', padding='same')(x)
    x = pooling_func(x)

    x = Convolution2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(x)
    x = Convolution2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(x)
    x = Convolution2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(x)
    if args.model == "vgg19":
        x = Convolution2D(512, (3, 3), activation='relu', name='conv4_4', padding='same')(x)
    x = pooling_func(x)

    x = Convolution2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(x)
    x = Convolution2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(x)
    x = Convolution2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(x)
    if args.model == "vgg19":
        x = Convolution2D(512, (3, 3), activation='relu', name='conv5_4', padding='same')(x)
    x = pooling_func(x)

    model = Model(ip, x)

    if K.image_dim_ordering() == "th":
        if args.model == "vgg19":
            weights = get_file('vgg19_weights_th_dim_ordering_th_kernels_notop.h5', TH_19_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
        else:
            weights = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5', THEANO_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    else:
        if args.model == "vgg19":
            weights = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_19_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
        else:
            weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')

    model.load_weights(weights)

    if K.backend() == 'tensorflow' and K.image_dim_ordering() == "th":
        warnings.warn('You are using the TensorFlow backend, yet you '
                      'are using the Theano '
                      'image dimension ordering convention '
                      '(`image_dim_ordering="th"`). '
                      'For best performance, set '
                      '`image_dim_ordering="tf"` in '
                      'your Keras config '
                      'at ~/.keras/keras.json.')
        convert_all_kernels_in_model(model)

    print('Model loaded.')
    return model

"""-------------------------------------------------Pre-process inputs for model & NST-------------------------------------------------"""

#Util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, load_dims=False, read_mode="color"):
    global img_width, img_height, img_WIDTH, img_HEIGHT, aspect_ratio

    mode = "RGB" if read_mode == "color" else "L"
    img = imread(image_path, mode=mode)  # Prevents crashes due to PNG images (ARGB)

    if mode == "L":
        # Expand the 1 channel grayscale to 3 channel grayscale image
        temp = np.zeros(img.shape + (3,), dtype=np.uint8)
        temp[:, :, 0] = img
        temp[:, :, 1] = img.copy()
        temp[:, :, 2] = img.copy()

        img = temp

    if load_dims:
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = float(img_HEIGHT) / img_WIDTH

        img_width = args.img_size
        if maintain_aspect_ratio:
            img_height = int(img_width * aspect_ratio)
        else:
            img_height = args.img_size

    img = imresize(img, (img_width, img_height)).astype('float32')

    # RGB -> BGR --- Why?
    img = img[:, :, ::-1]

    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    if K.image_dim_ordering() == "th": #Must we keep changing it everytime?
        img = img.transpose((2, 0, 1)).astype('float32')

    img = np.expand_dims(img, axis=0)
    return img

#Util function to convert a tensor into a valid image
def deprocess_image(x):
    global img_width, img_height, img_WIDTH, img_HEIGHT, aspect_ratio

    if K.image_dim_ordering() == "th":
        x = x.reshape((3, img_width, img_height))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_width, img_height, 3))

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR -> RGB
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

#Util function to preserve image color
def original_color_transform(content, generated, mask=None):
    generated = fromimage(toimage(generated, mode='RGB'), mode='YCbCr')  # Convert to YCbCr color space

    if mask is None:
        generated[:, :, 1:] = content[:, :, 1:]  # Generated CbCr = Content CbCr
    else:
        width, height, channels = generated.shape

        for i in range(width):
            for j in range(height):
                if mask[i, j] == 1:
                    generated[i, j, 1:] = content[i, j, 1:]

    generated = fromimage(toimage(generated, mode='YCbCr'), mode='RGB')  # Convert to RGB color space
    return generated

#Util function to load masks
def load_mask(mask_path, shape, return_mask_img=False):
    if K.image_dim_ordering() == "th":
        _, channels, width, height = shape
    else:
        _, width, height, channels = shape

    mask = imread(mask_path, mode="L") # Grayscale mask load
    mask = imresize(mask, (width, height)).astype('float32')

    # Perform binarization of mask
    mask[mask <= 127] = 0
    mask[mask > 128] = 255

    max = np.amax(mask)
    mask /= max

    if return_mask_img: return mask

    mask_shape = shape[1:]

    mask_tensor = np.empty(mask_shape)

    for i in range(channels):
        if K.image_dim_ordering() == "th":
            mask_tensor[i, :, :] = mask
        else:
            mask_tensor[:, :, i] = mask

    return mask_tensor

#Util function to select pooling function
def pooling_func(x):
    if pooltype == 1:
        return AveragePooling2D((2, 2), strides=(2, 2))(x)
    else:
        return MaxPooling2D((2, 2), strides=(2, 2))(x)


def pre_nst():
    global img_width, img_height, img_WIDTH, img_HEIGHT, aspect_ratio

    #Content image
    base_image = K.variable(preprocess_image(base_image_path, True, read_mode=read_mode))

    #Style image(s)
    style_reference_images = []
    for style_path in style_image_paths:
        style_reference_images.append(K.variable(preprocess_image(style_path)))

    #Generated image
    if K.image_dim_ordering() == 'th':
        combination_image = K.placeholder((1, 3, img_width, img_height))
    else:
        combination_image = K.placeholder((1, img_width, img_height, 3))

    #List of all image tensors
    image_tensors = [base_image]
    for style_image_tensor in style_reference_images:
        image_tensors.append(style_image_tensor)
    image_tensors.append(combination_image)

    nb_tensors = len(image_tensors)
    nb_style_images = nb_tensors - 2 #Content and Output image not considered

    #Combine the various images into a single Keras tensor
    input_tensor = K.concatenate(image_tensors, axis=0)

    if K.image_dim_ordering() == "th":
        shape = (nb_tensors, 3, img_width, img_height)
    else:
        shape = (nb_tensors, img_width, img_height, 3)

    ip = Input(tensor=input_tensor, batch_shape=shape)

    return ip, nb_tensors, nb_style_images, combination_image

"""-------------------------------------------Compute the neural style loss & gradients-------------------------------------------------"""
#Util functions for loss computation

#Covariance function
def cov_matrix(x):
    assert K.ndim(x) == 3
    if K.image_dim_ordering() == "th":
        features = K.batch_flatten(x)
        num_points = (K.shape(features)[0]) - 1
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        num_points = (K.shape(features)[0]) - 1

    features -= K.mean(features, 0, keepdims=True)
    #TODO: 1. Check if it works with dividing by num_points. 2. Check if it works with subtracting mean alone 3. Check if it works with both
    cov = K.dot(features, K.transpose(features)) / K.cast(num_points, 'float32')
    return cov

#Gram matrix function (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_dim_ordering() == "th":
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))

    gram = K.dot(features, K.transpose(features))
    return gram

#Custom Style loss
def style_loss(style, combination, mask_path=None, nb_channels=None):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3

    if content_mask_path is not None:
        content_mask = K.variable(load_mask(content_mask_path, nb_channels))
        combination = combination * K.stop_gradient(content_mask)
        del content_mask

    if mask_path is not None:
        style_mask = K.variable(load_mask(mask_path, nb_channels))
        style = style * K.stop_gradient(style_mask)
        if content_mask_path is None:
            combination = combination * K.stop_gradient(style_mask)
        del style_mask

    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_width * img_height
    res = K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    return res

#Custom Content loss
def content_loss(base, combination):
    channel_dim = 0 if K.image_dim_ordering() == "th" else -1

    try:
        channels = K.int_shape(base)[channel_dim]
    except TypeError:
        channels = K.shape(base)[channel_dim]
    size = img_width * img_height

    if args.content_loss_type == 1:
        multiplier = 1. / (2. * (channels ** 0.5) * (size ** 0.5))
    elif args.content_loss_type == 2:
        multiplier = 1. / (channels * size)
    else:
        multiplier = 1.

    return multiplier * K.sum(K.square(combination - base))


#Total variation loss - designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_dim_ordering() == 'th':
        a = K.square(x[:, :, :img_width - 1, :img_height - 1] - x[:, :, 1:, :img_height - 1])
        b = K.square(x[:, :, :img_width - 1, :img_height - 1] - x[:, :, :img_width - 1, 1:])
    else:
        a = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])
        b = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def nst(style_weights, content_weight, total_variation_weight, outputs_dict, shape_dict, nb_tensors, nb_style_images, combination_image):

    nst.counter += 1
    print(nst.counter)
    global f_outputs

    print("W: ", style_weights)

    #Combine these loss functions into a single scalar
    loss = K.variable(0.)
    layer_features = outputs_dict[args.content_layer]  # 'conv5_2' or 'conv4_2'
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[nb_tensors - 1, :, :, :]
    loss = loss + content_weight * content_loss(base_image_features,
                                          combination_features)
    style_masks = []
    if style_masks_present:
        style_masks = mask_paths #If mask present, pass dictionary of masks to style loss
    else:
        style_masks = [None for _ in range(nb_style_images)] # If masks not present, pass None to the style loss

    channel_index = 1 if K.image_dim_ordering() == "th" else -1

    feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        shape = shape_dict[layer_name]
        combination_features = layer_features[nb_tensors - 1, :, :, :]

        style_reference_features = layer_features[1:nb_tensors - 1, :, :, :]
        sl = []
        for j in range(nb_style_images):
            sl.append(style_loss(style_reference_features[j], combination_features, style_masks[j], shape))

        for j in range(nb_style_images):
            loss = loss + (style_weights[j] / len(feature_layers)) * sl[j]

    loss = loss + total_variation_weight * total_variation_loss(combination_image)

    #Get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combination_image)
    # print("grads: ", grads)

    outputs = [loss]
    if type(grads) in {list, tuple}:
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([combination_image], outputs)
    # print("f_outputs: ", f_outputs)

    evaluator = Evaluator()

    if "content" in args.init_image or "gray" in args.init_image:
        x = preprocess_image(base_image_path, True, read_mode=read_mode)
    elif "noise" in args.init_image:
        x = np.random.uniform(0, 255, (1, img_width, img_height, 3)) - 128.

        if K.image_dim_ordering() == "th":
            x = x.transpose((0, 3, 1, 2))
    else:
        print("Using initial image : ", args.init_image)
        x = preprocess_image(args.init_image, read_mode=read_mode)

    # We require original image if we are to preserve color in YCbCr mode
    if preserve_color:
        content = imread(base_image_path, mode="YCbCr")
        content = imresize(content, (img_width, img_height))

        if color_mask_present:
            if K.image_dim_ordering() == "th":
                color_mask_shape = (None, None, img_width, img_height)
            else:
                color_mask_shape = (None, img_width, img_height, None)

            color_mask = load_mask(args.color_mask, color_mask_shape, return_mask_img=True)
        else:
            color_mask = None
    else:
        color_mask = None

    num_iter = args.num_iter
    prev_min_val = -1

    improvement_threshold = float(args.min_improvement)

    """-------------------------------------------------------NEURAL STYLE TRANSFER---------------------------------------------------------"""
    for i in range(num_iter):

        print("Starting iteration %d of %d" % ((i + 1), num_iter))
        start_time = time.time()

        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=50)

        if prev_min_val == -1:
            prev_min_val = min_val

        improvement = (prev_min_val - min_val) / prev_min_val * 100

        print('Current loss value:', min_val, " Improvement : %0.3f" % improvement, "%")

        prev_min_val = min_val
        # save current generated image
        img = deprocess_image(x.copy())

        if preserve_color and content is not None:
            img = original_color_transform(content, img, mask=color_mask)

        if not rescale_image:
            img_ht = int(img_width * aspect_ratio)
            print("Rescaling Image to (%d, %d)" % (img_width, img_ht))
            img = imresize(img, (img_width, img_ht), interp=args.rescale_method)

        if rescale_image:
            print("Rescaling Image to (%d, %d)" % (img_WIDTH, img_HEIGHT))
            img = imresize(img, (img_WIDTH, img_HEIGHT), interp=args.rescale_method)

        fname = result_prefix + '-' + str(nst.counter) + '_at_iteration_%d.png' % (i + 1)
        imsave(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i + 1, end_time - start_time))

        if improvement_threshold is not 0.0:
            if improvement < improvement_threshold and improvement is not 0.0:
                print("Improvement (%f) is less than improvement threshold (%f). Early stopping script." % (
                    improvement, improvement_threshold))
                break

    print("DONE", x.flatten().shape, min_val, info)

    print(min_val)
    return min_val



def eval_loss_and_grads(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((1, 3, img_width, img_height))
    else:
        x = x.reshape((1, img_width, img_height, 3))
    outs = f_outputs([x])

    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')

    return loss_value, grad_values

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
        self.w_loss = None
        self.w_grad = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        # print("in loss(): ")
        # print(loss_value, grad_values)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

    def wloss(self, style_weights, content_weight, total_variation_weight, outputs_dict, shape_dict, nb_tensors, nb_style_images, combination_image):

        min_val = nst(style_weights, content_weight, total_variation_weight, outputs_dict, shape_dict, nb_tensors, nb_style_images, combination_image)
        # grad = style_gradient(style_weights)
        self.w_loss = min_val
        # self.w_grad = grad
        return self.w_loss

    # def wgrads(self, style_weights, content_weight, total_variation_weight, outputs_dict, shape_dict, nb_tensors, nb_style_images, combination_image):
    #     assert self.w_grad is not None
    #     grad = np.copy(self.w_grad)
    #     self.w_loss = None
    #     self.w_grad = None
    #     print("GRAD: ",grad)
    #     return grad


def optimize_nst(content_weight, total_variation_weight, outputs_dict, shape_dict, nb_tensors, nb_style_images, combination_image):

    evaluator = Evaluator()
    length = nb_style_images
    style_weights = []
    bounds = []
    f = 1.0/length

    for i in range(0,length):
        style_weights.append(f)
        bounds.append((0,1))

    style_weights = np.asarray(style_weights)

    print("OPTIMIZING...")

    constraints = ({ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) })
    res = m.basinhopping(nst, style_weights, niter = 5, minimizer_kwargs = {'method': 'SLSQP', 'args': (content_weight, total_variation_weight, outputs_dict, shape_dict, nb_tensors, nb_style_images, combination_image), 'bounds': bounds, 'constraints': constraints, 'options': {'ftol': 1e-06, 'eps': 0.1, 'disp': True}}) #, 'jac': evaluator.wgrads
    # res = m.minimize(nst, style_weights, method = 'SLSQP', args = (content_weight, total_variation_weight, outputs_dict, shape_dict, nb_tensors, nb_style_images, combination_image), bounds = bounds, constraints = constraints, options={'ftol': 1e-06, 'eps': 0.2, 'disp': True})

    #Try basinhop with slsqp
    #Try without sum constraint
    #

    print("Res: ", res)
    res_allocs = res.x

    return res_allocs

if __name__ == '__main__':

    ip, nb_tensors, nb_style_images, combination_image = pre_nst()
    model = load_model(ip)

    #Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    shape_dict = dict([(layer.name, layer.output_shape) for layer in model.layers])

    style_weights, content_weight, total_variation_weight = image_component_weights()

    nst.counter = 0

    style_weights = optimize_nst(content_weight, total_variation_weight, outputs_dict, shape_dict, nb_tensors, nb_style_images, combination_image)

    print("Style weights used: ", style_weights)
    print("RUNNING NST: ")

    loss = nst(style_weights, content_weight, total_variation_weight, outputs_dict, shape_dict, nb_tensors, nb_style_images, combination_image)

    """
    JOURNAL:
    1. Dipping sun with the scream + wave_kanagawa - 0.5, 0.5
    2. Golden Gate with misty mood + frida kahlo - 0.5, 0.5


    Without gradient function, same style weights
    With gradient function, errors!!!!


    Break down into minute steps!

        python -W ignore optimize_nst.py images/inputs/content/Dipping-Sun.jpg images/inputs/style/the_scream.jpg images/inputs/style/wave_kanagawa.jpg Results/basinhop/bh

    """
