import time
import os
import numpy as np
import PIL.Image
from vgg import VGG
from neural_network_style_transfer import NeuralNetwork
import matplotlib.pyplot as plt
import shutil

PATH = '/'.join(os.getcwd().split('\\'))
VGG_MAT_PATH = PATH + r'/imagenet-vgg-verydeep-19.mat'
CONTENT_IMAGE_PATH = PATH + r'/Corner_towers_of_the_Forbidden_City_3335.jpg'
STYLE_IMAGE_PATH = PATH + r'/police-gazette.jpg'
MIXED_IMAGE_PATH = PATH + r'/figure'
MIXED_IMAGE_NAME = r'/mixed_image'

CONTENT_WEIGHT = 1e-2
STYLE_WEIGHT = 1
VARIATION_WEIGHT = 0
LEARNING_RATE = 10
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
MAX_ITERATION = 1000
POOLING = 'max'
CHECK_POINT = 1
INIT_IMAGE = 'content'  # random / content / black
STYLE_DEMEAN = False

CONTENT_LAYER_WEIGHTS = {
    'relu4_2': 1.0,
}
STYLE_LAYER_WEIGHTS = {
    'relu1_1': 0.2,
    'relu2_1': 0.2,
    'relu3_1': 0.2,
    'relu4_1': 0.2,
    'relu5_1': 0.2,
}


def load_image(file_path, max_size=None, shape=None):
    # load image and define the factor used to transfer the image size
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
                   pooling, learning_rate, beta1, beta2, epsilon, max_iteration, check_point,
                   init_image='random', mixed_image_name='mixed_image', style_demean=False):
    # set the time point
    time_start = time.time()

    # load image
    content_image = load_image(content_image_path)
    style_image = load_image(style_image_path, shape=content_image.shape[:2])
    print('Successfully loaded content image and style image.')

    # initialize object
    vgg = VGG(VGG_MAT_PATH, pooling)
    print('Successfully loaded the VGG-19 model.')
    nn = NeuralNetwork(content_image, style_image, vgg, content_weight, style_weight, variation_weight,
                       content_layer_weights=CONTENT_LAYER_WEIGHTS, style_layer_weights=STYLE_LAYER_WEIGHTS,
                       style_demean=style_demean)
    print('Successfully constructed the style transfer optimization model.')

    # train the model
    for mixed_image, losses in nn.train_model(learning_rate, beta1, beta2, epsilon, max_iteration, check_point,
                                              init_image):
        if len(losses) > 200:
            plt.plot(np.arange(start=len(losses) - 150, stop=len(losses), step=1),
                     losses[len(losses) - 150:len(losses)])
            plt.savefig(mixed_image_path + r'/' + mixed_image_name + '_loss.jpeg')
            plt.close()
        elif losses[len(losses) - 1] == 0:
            plt.plot(losses)
            plt.savefig(mixed_image_path + r'/' + mixed_image_name + '_loss.jpeg')
            plt.close()
            save_image(mixed_image_path + r'/' + mixed_image_name + '_image.jpeg', mixed_image)
            break
        else:
            plt.plot(losses)
            plt.savefig(mixed_image_path + r'/' + mixed_image_name + '_loss.jpeg')
            plt.close()
        save_image(mixed_image_path + r'/' + mixed_image_name + '_image.jpeg', mixed_image)

    # print time
    time_end = time.time()
    print('Time elapsed: {} seconds'.format(round(time_end - time_start)))

    return


def construct_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return


def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    return


if __name__ == '__main__':
    clear_dir(MIXED_IMAGE_PATH)
    construct_dir(MIXED_IMAGE_PATH)
    construct_dir(PATH + 'logs')
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
                   check_point=CHECK_POINT,
                   init_image=INIT_IMAGE,
                   mixed_image_name=MIXED_IMAGE_NAME,
                   style_demean=STYLE_DEMEAN)
