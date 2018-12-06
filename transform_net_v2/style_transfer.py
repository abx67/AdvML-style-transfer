import time
import os
import numpy as np
import PIL.Image
from vgg import VGG
from neural_network_transform import NeuralNetwork
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import shutil

PATH = '/'.join(os.getcwd().split('\\'))
VGG_MAT_PATH = PATH + r'/imagenet-vgg-verydeep-19.mat'
CONTENT_IMAGE_PATH = PATH + r'/Corner_towers_of_the_Forbidden_City_3335.jpg'
STYLE_IMAGE_PATH = PATH + r'/seated-nude.jpg'
MIXED_IMAGE_PATH = PATH + r'/figure'
MODEL_DIR = PATH + r'/model'
MODEL_STORE_NAME = r'transform_net'
MIXED_IMAGE_NAME = r'/mixed_image'

CONTENT_WEIGHT = 1e-2
STYLE_WEIGHT = 1.0
VARIATION_WEIGHT = 1e2
LEARNING_RATE = 10
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
MAX_ITERATION = 20000
CONTENT_REPRODUCE_ITERATION = 3000
POOLING = 'max'
CHECK_POINT = 100
STYLE_DEMEAN = False
VARIATION_LOSS_KERNEL_H = 100.0  # pixel value squared scale
SHAPE = (512, 512)

CONTENT_LAYER_WEIGHTS = {
    'conv4_2': 1.0,
}
STYLE_LAYER_WEIGHTS = {
    'relu1_1': 0.2,
    'relu2_1': 0.2,
    'relu3_1': 0.2,
    'relu4_1': 0.2,
    'relu5_1': 0.2,
}


def load_image(file_path, max_size=None, shape=None):
    '''
    load an image file
    :param file_path: str, path for the image file
    :param max_size: int, maximum value of height and width for an image, resize the image by a factor
    :param shape: tuple, (height, width), resize the loaded image to specific height and width
    :return: 3-D numpy array, RGB image
    '''

    # load image
    image = PIL.Image.open(file_path)
    # resize by max_size
    if max_size is not None:
        factor = float(max_size) / np.max(image.size)  # image.size = [height, width, 3]
        size = np.array(image.size) * factor
        size = size.astype(int)
        image = image.resize(size, PIL.Image.LANCZOS)  # image resize with filter LANCZOS
    # resize by shape
    if shape is not None:
        if image.size[0] / image.size[1] < shape[0] / shape[1]:
            left = 0
            upper = int(image.size[1] / 2) - int(image.size[0] / shape[0] * shape[1] / 2)
        else:
            upper = 0
            left = int(image.size[0] / 2) - int(image.size[1] / shape[1] * shape[0] / 2)
        image = image.crop(box=(left, upper, image.size[0] - left, image.size[1] - upper))
        image = image.resize(shape, PIL.Image.LANCZOS)
    # return image values with float data type
    return np.float32(image)


def save_image(file_path, image):
    '''
    function for saving the image
    :param file_path: str, path for the image to be saved
    :param image: 3-D numpy array, int or float
    :return: None
    '''
    # ensure the pixel value is int between 0 and 255 and data type is int
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    # write to file
    PIL.Image.fromarray(image).save(file_path)
    return


def style_transfer(content_image_path, style_image_path, mixed_image_path,
                   content_weight, style_weight, variation_weight,
                   pooling, learning_rate, beta1, beta2, epsilon, max_iteration, check_point,
                   mixed_image_name='mixed_image', shape=None, style_demean=False):
    '''
    function for training transformation model and generate the mixed image
    :param content_image_path: str, path for the content image file
    :param style_image_path: str, path for the style image file
    :param mixed_image_path: str, path for the output image file, the mixed image will be saved to this path
    :param content_weight: float, weight for the loss related to content image and mixed image
    :param style_weight: float, weight for the loss related to style image and mixed image
    :param variation_weight: float, weight for the loss related to mixed image (related to the smoothness of mixed image)
    :param pooling: 'avg' or 'max', pooling method for the vgg model
    :param learning_rate: float, default learning rate for AdamOptimizer
    :param beta1: float, parameter for AdamOptimizer
    :param beta2: float, parameter for AdamOptimizer
    :param epsilon: float, parameter for AdamOptimizer
    :param max_iteration: int, maximum iteration times
    :param check_point: int, after each check point number of iterations, the result will be printed and the trained network will be stored
    :param mixed_image_name: str, file name for the output mixed image
    :param shape: tuple, (height, width), resize the loaded image to specific height and width
    :param style_demean: bool, whether to centralize for style feature, default=False
    :return: None
    '''

    # set the time stamp
    time_start = time.time()

    # load image
    content_image = load_image(content_image_path, shape=shape)
    plt.imshow(np.uint8(content_image))
    plt.axis('off')
    plt.show()
    style_image = load_image(style_image_path,
                             shape=(content_image.shape[1], content_image.shape[0]))
    # the style image will have the same shape as content image
    plt.imshow(np.uint8(style_image))
    plt.axis('off')
    plt.show()
    print('Successfully loaded content image and style image.')

    # initialize object
    vgg = VGG(VGG_MAT_PATH, pooling)
    print('Successfully loaded the VGG-19 model.')
    nn = NeuralNetwork(content_image, style_image, vgg, content_weight, style_weight, variation_weight,
                       content_layer_weights=CONTENT_LAYER_WEIGHTS, style_layer_weights=STYLE_LAYER_WEIGHTS,
                       model_dir=MODEL_DIR + '/' + MODEL_STORE_NAME, style_demean=style_demean,
                       variation_loss_kernel_h=VARIATION_LOSS_KERNEL_H)
    print('Successfully constructed the style transfer optimization model.')

    # train the model
    print('Started training the model.')
    for mixed_image, losses in nn.train_model(learning_rate, beta1, beta2, epsilon, max_iteration,
                                              CONTENT_REPRODUCE_ITERATION, check_point):
        if len(losses) > 200:
            plt.plot(np.arange(start=len(losses) - 150, stop=len(losses), step=1),
                     losses[len(losses) - 150:len(losses)])
            plt.savefig(mixed_image_path + r'/' + mixed_image_name + '_loss.jpeg')
            plt.close()
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
    clear_dir(MODEL_DIR)
    construct_dir(MODEL_DIR)
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
                   mixed_image_name=MIXED_IMAGE_NAME,
                   shape=SHAPE,
                   style_demean=STYLE_DEMEAN)

    # load an image to be transformed
    image_input = load_image(CONTENT_IMAGE_PATH, shape=SHAPE)
    image_input = np.reshape(image_input, (1,) + image_input.shape)

    # using the trained model to transform the image
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            tf.saved_model.loader.load(sess, [tag_constants.SERVING], MODEL_DIR + '/' + MODEL_STORE_NAME)
            input_image = graph.get_tensor_by_name('input_image:0')
            mixed_image = graph.get_tensor_by_name('mixed_image:0')
            image_output = sess.run(mixed_image, feed_dict={input_image: image_input})
            image_output = np.reshape(image_output, image_output.shape[1:])

    # plot the transformed image
    plt.axis('off')
    plt.imshow(np.uint8(image_output))
    plt.show()
