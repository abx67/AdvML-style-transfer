import numpy as np
import os
import scipy.io
import tensorflow as tf

PATH = os.getcwd()

vgg_mat_path = PATH + '\\imagenet-vgg-verydeep-19.mat'


class VGG(object):
    # VGG - pre-trained neural network object

    def __init__(self, vgg_mat_path, pooling):
        # load the mat file for vgg19 and load variables
        self.network = scipy.io.loadmat(vgg_mat_path)
        self.layers = self.network['layers'].reshape(-1)  # length = 43
        self.layer_names = [self.layers[i][0, 0][0][0] for i in range(self.layers.shape[0])]
        idx = min([i for i in range(len(self.layer_names)) if self.layer_names[i][:2] == 'fc'])
        self.layer_names = self.layer_names[:idx]  # only keep layers before the fully connected layer
        self.mean_pix = self.network['meta'][0, 0]['normalization'][0][0][2][0][0]  # length = 3
        self.pooling = pooling
        return

    def load_net(self, input_image):
        # construct layers for an input image
        # input_image - [1, height, width, 3(channel)]

        # feed forward to get the feature values of each layer
        net = {}
        cur_layer = input_image
        for i, layer_name in enumerate(self.layer_names):
            type = layer_name[:4]
            if type == 'conv':
                cur_layer = self.get_conv_layer(cur_layer, self.layers[i])
            elif type == 'relu':
                cur_layer = self.get_relu_layer(cur_layer)
            elif type == 'pool':
                cur_layer = self.get_pool_layer(cur_layer)
            net[layer_name] = cur_layer
        # return constructed neural network
        return net

    def get_conv_layer(self, input, layer_wb):
        w, b = layer_wb[0][0][2][0]
        # matconvnet: weights are [width, height, in_channels, out_channels]
        # tensorflow: weights are [height, width, in_channels, out_channels]
        w = np.transpose(w, (1, 0, 2, 3))
        b = b.reshape(-1)
        layer = tf.nn.bias_add(tf.nn.conv2d(input, tf.constant(w), strides=(1, 1, 1, 1), padding='SAME'), b)
        return layer

    def get_relu_layer(self, input):
        layer = tf.nn.relu(input)
        return layer

    def get_pool_layer(self, input):
        if self.pooling == 'avg':
            layer = tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        elif self.pooling == 'max':
            layer = tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        return layer
