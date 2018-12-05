import time
import os
import numpy as np
import PIL.Image
from vgg import VGG
from neural_network_feature_visualization import NeuralNetwork
import matplotlib.pyplot as plt

PATH = '/'.join(os.getcwd().split('\\'))
VGG_MAT_PATH = PATH + r'/imagenet-vgg-verydeep-19.mat'
IMAGE_PATH = PATH + r'/police-gazette.jpg'
FEATURE_IMAGE_PATH = PATH + r'/figure'
FEATURE_IMAGE_NAME = 'feature_visualization'
TYPE = 'filter'

LEARNING_RATE = 10
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
MAX_ITERATION = 1000
POOLING = 'max'
CHECK_POINT = 1
INIT_IMAGE = 'random'  # random / content


def load_image(file_path, max_size=None, shape=None):
    # load image and define the factor used to tranfser the image size
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


def feature_visual(image_path, feature_image_path,
                   pooling, learning_rate, beta1, beta2, epsilon, max_iteration, check_point,
                   init_image='random', feature_image_name='feat_vis', type='filter'):
    # set the time point
    time_start = time.time()

    # load image
    image = load_image(image_path)

    # initialize object
    vgg = VGG(VGG_MAT_PATH, pooling)
    nn = NeuralNetwork(image, vgg, type=type)

    # train the model
    for feature_image, losses in nn.train_model(learning_rate, beta1, beta2, epsilon, max_iteration, check_point,
                                                init_image):
        if len(losses) > 200:
            plt.plot(np.arange(start=len(losses) - 150, stop=len(losses), step=1),
                     losses[len(losses) - 150:len(losses)])
            plt.savefig(feature_image_path + r'/' + feature_image_name + '_loss.jpeg')
            plt.close()
        elif losses[len(losses) - 1] == 0:
            plt.plot(losses)
            plt.savefig(feature_image_path + r'/' + feature_image_name + '_loss.jpeg')
            plt.close()
            save_image(feature_image_path + r'/' + feature_image_name + '_image.jpeg', feature_image)
            break
        else:
            plt.plot(losses)
            plt.savefig(feature_image_path + r'/' + feature_image_name + '_loss.jpeg')
            plt.close()
        save_image(feature_image_path + r'/' + feature_image_name + '_image.jpeg', feature_image)

    # print time
    time_end = time.time()
    print('Time elapsed: {} seconds'.format(round(time_end - time_start)))

    return


def construct_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    construct_dir(FEATURE_IMAGE_PATH)
    construct_dir(PATH + 'logs')
    feature_visual(image_path=IMAGE_PATH,
                   feature_image_path=FEATURE_IMAGE_PATH,
                   pooling=POOLING,
                   learning_rate=LEARNING_RATE,
                   beta1=BETA1,
                   beta2=BETA2,
                   epsilon=EPSILON,
                   max_iteration=MAX_ITERATION,
                   check_point=CHECK_POINT,
                   init_image=INIT_IMAGE,
                   feature_image_name=FEATURE_IMAGE_NAME,
                   type=TYPE)
