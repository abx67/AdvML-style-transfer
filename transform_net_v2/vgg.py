import numpy as np
import os
import scipy.io
import tensorflow as tf

PATH = os.getcwd()

vgg_mat_path = PATH + '\\imagenet-vgg-verydeep-19.mat'


class VGG(object):
    '''
    VGG model - pre-trained neural network object, originally used for classification
              - here we use this model for extracting features of style image
    '''

    def __init__(self, vgg_mat_path, pooling):
        '''
        initialization of VGG object
        :param vgg_mat_path: the path of vgg model, stored as a ***.mat file
        :param pooling: method of pooling in vgg model, 'max' or 'avg'
        '''

        # load the mat file for vgg19 and load variables
        self.network = scipy.io.loadmat(vgg_mat_path)
        # get all of the layer weights from the loaded data
        self.layers = self.network['layers'].reshape(-1)  # length = 43, 43 layers in total
        # get all of the layer names, like conv1_1, relu1_1, etc
        self.layer_names = [self.layers[i][0, 0][0][0] for i in range(self.layers.shape[0])]
        # only keep convolution and relu layers, excludes the fully connected layers
        idx = min([i for i in range(len(self.layer_names)) if self.layer_names[i][:2] == 'fc'])
        self.layer_names = self.layer_names[:idx]  # only keep layers before the fully connected layer
        # mean pixels used to center the input image
        self.mean_pix = self.network['meta'][0, 0]['normalization'][0][0][2][0][0]  # length = 3
        # pooling method
        self.pooling = pooling
        return

    def load_net(self, input_image):
        '''
        construct a feed forward network, input an image and get the value of each units in the vgg network
        :param input_image: input image, shape = (1, height, width, 3(channel)]
        :return: dict object, key - layer name (e.g. 'conv1_1'), value - 4-D tensor [1, height, width, channel]
        '''

        # generate the output dict type data
        net = {}
        # feed forward before the fully connected layer
        cur_layer = input_image
        for i, layer_name in enumerate(self.layer_names):
            layer_type = layer_name[:4]
            if layer_type == 'conv':
                cur_layer = self.get_conv_layer(cur_layer, self.layers[i])
            elif layer_type == 'relu':
                cur_layer = self.get_relu_layer(cur_layer)
            elif layer_type == 'pool':
                cur_layer = self.get_pool_layer(cur_layer)
            net[layer_name] = cur_layer
        # return constructed neural network
        return net

    def get_conv_layer(self, input_tensor, layer_wb):
        '''
        conpute the value for a convolution layer
        :param input_tensor: 4-D tensor of numpy array object, [1, height, width, channel]
        :param layer_wb: weight and bias for a specific layer
        :return: 4-D tensor, [1, height, width, channel]
        '''

        # get weight and bias
        w, b = layer_wb[0][0][2][0]
        # matconvnet: weights are [width, height, in_channels, out_channels]
        # tensorflow: weights are [height, width, in_channels, out_channels]
        w = np.transpose(w, (1, 0, 2, 3))
        b = b.reshape(-1)
        # compute the value of tensor at this layer
        output_tensor = tf.nn.bias_add(tf.nn.conv2d(input_tensor, tf.constant(w), strides=(1, 1, 1, 1), padding='SAME'),
                                       b)
        # return the output tensor
        return output_tensor

    def get_relu_layer(self, input_tensor):
        '''
        compute the value after a relu layer
        :param input_tensor: 4-D tensor of numpy array object, [1, height, width, channel]
        :return: 4-D tensor, [1, height, width, channel]
        '''

        # using standard relu function from tensorflow
        output_tensor = tf.nn.relu(input_tensor)
        # return the output tensor
        return output_tensor

    def get_pool_layer(self, input_tensor):
        '''
        compute the value after a pooling layer
        :param input_tensor: 4-D tensor of numpy array object, [1, height, width, channel]
        :return: 4-D tensor, [1, height, width, channel]
        '''
        # pooling strategy - 2 * 2 size with strides of 2
        if self.pooling == 'avg':
            output_tensor = tf.nn.avg_pool(input_tensor, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        elif self.pooling == 'max':
            output_tensor = tf.nn.max_pool(input_tensor, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # return the output tensor
        return output_tensor
