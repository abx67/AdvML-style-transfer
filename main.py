import time
import os
import numpy as np
import PIL.Image
from neural_network_my import NeuralNetwork
from vgg import VGG


# from code.neural_network import NeuralNetwork

PATH = '/Users/fanerror/GitHub/OtherGit/AdvML-fall17-project/'
VGG_MAT_PATH = PATH + 'data/imagenet-vgg-verydeep-19.mat'
CONTENT_IMAGE_PATH = PATH + 'input/content/new_york.jpg'
STYLE_IMAGE_PATH = PATH + 'input/style/van_gogh.jpg'
MIXED_IMAGE_PATH = PATH + '/figure'

CONTENT_WEIGHT = 5
STYLE_WEIGHT = 500
VARIATION_WEIGHT = 100
LEARNING_RATE = 10
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
MAX_ITERATION = 1000
POOLING = 'max'
CHECK_POINT = 1


def load_image(file_path, max_size=None, shape=None):
    # load image and define the factor used to tranfer the image size
    image = PIL.Image.open(file_path)
    # resize by max_size
    if max_size is not None:
        factor = float(max_size) / np.max(image.size)  # image.size = [height, width, 3]
        size = np.array(image.size) * factor
        size = size.astype(int)
        image = image.resize(size, PIL.Image.LANCZOS)  # image resize with filter LANCZOS
    # resize with shape
    if shape is not None:
        image = image.resize(shape, PIL.Image.LANCZOS)
    # return image values with float data type
    return np.float32(image)


def save_image(file_path, image):
    # ensure the pixel value is int between 0 and 255
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    # write to file
    PIL.Image.fromarray(image).save(file_path)
    return


def style_transfer(content_image_path, style_image_path, mixed_image_path,
                   content_weight, style_weight, variation_weight,
                   pooling, learning_rate, beta1, beta2, epsilon, max_iteration, check_point):
    # set the time point
    time_start = time.time()

    # load image
    content_image = load_image(content_image_path)
    style_image = load_image(style_image_path, shape=content_image.shape[:2])

    # initialize object
    vgg = VGG(VGG_MAT_PATH, pooling)
    nn = NeuralNetwork(content_image, style_image, vgg, content_weight, style_weight, variation_weight)

    # train the model
    for i, mixed_image in nn.train_model(learning_rate, beta1, beta2, epsilon, max_iteration, check_point):
        save_image(mixed_image_path + '\\v1_{}.jpeg'.format(i + 1), mixed_image)

    # print time
    time_end = time.time()
    print('Time elapsed: {} seconds'.format(round(time_end - time_start)))

    return


if __name__ == '__main__':
    style_transfer(content_image_path=CONTENT_IMAGE_PATH,
                   style_image_path=STYLE_IMAGE_PATH,
                   mixed_image_path=MIXED_IMAGE_PATH,
                   content_weight=CONTENT_WEIGHT,
                   style_weight=STYLE_WEIGHT,
                   variation_weight=VARIATION_WEIGHT,
                   pooling=POOLING,
                   learning_rate=LEARNING_RATE,
                   beta1=BETA1,
                   beta2=BETA2,
                   epsilon=EPSILON,
                   max_iteration=MAX_ITERATION,
                   check_point=CHECK_POINT)
