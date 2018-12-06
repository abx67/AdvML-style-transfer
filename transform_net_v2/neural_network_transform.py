import numpy as np
import tensorflow as tf
import os
import time
import shutil

PATH = '/'.join(os.getcwd().split('\\'))


# LAYER_NAMES = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
#                'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
#                'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4',
#                'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4',
#                'pool5']

def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    return


class NeuralNetwork(object):
    '''
    neural network used to train the style transfer model and do transformation for an input image,
    which includes the definition of loss function, optimization function, etc.
    '''

    def __init__(self, content, style, vgg, content_weight, style_weight, variation_weight,
                 content_layer_weights, style_layer_weights, model_dir, style_demean=False,
                 variation_loss_kernel_h=255 * 1e5):
        '''
        initialization for the NeuralNetwork object
        :param content: 3-D numpy array, content image, [height, width, 3(channel)]
        :param style: 3-D numpy array, style image, [height, width, 3(channel)]
        :param vgg: VGG object, VGG model, which can be founded in vgg.py file
        :param content_weight: float, weight for the loss related to content image and mixed image
        :param style_weight: float, weight for the loss related to style image and mixed image
        :param variation_weight: float, weight for the loss related to mixed image (related to the smoothness of mixed image)
        :param content_layer_weights: dict, weight of loss at each layer in vgg network related to content image and mixed image
        :param style_layer_weights: dict, weight of loss at each layer in vgg network related to style image and mixed image
        :param model_dir: str, directory where the trained transformation model is saved
        :param style_demean: bool, whether to centralize for style feature, default=False
        :param variation_loss_kernel_h: float, Gaussian kernel parameter, the higher this value, the more smooth
        '''

        # save content image and style image, expand the dimension to 4-D, [1, height, width, 3(channel)]
        self.content = np.reshape(content, (1,) + content.shape)
        self.style = np.reshape(style, (1,) + content.shape)

        # save vgg model object
        self.vgg = vgg

        # save variables related to weights
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.variation_weight = variation_weight
        self.content_layer_weights = content_layer_weights
        self.style_layer_weights = style_layer_weights

        # save the bool value for whether to de-mean for style image's gram matrix
        self.style_demean = style_demean

        # get the shape of the image (both content and style)
        self.shape = self.content.shape
        if self.shape != self.style.shape:
            print('The shapes of content image and style image are different. Cannot train the model.')
            print('    content image shape: {}, style image shape: {}'.format(self.content.shape, self.style.shape))

        # compute the value of units at each layer for content image
        self.content_features = self.get_content_features()

        # compute the value of gram matrix at each layer for style image
        self.style_features = self.get_style_features()

        # save the iteration numbers for the
        # save directory where the trained transformation model is saved
        self.model_dir = model_dir

        # save the parameter for kernel function of the variation loss
        self.variation_loss_kernel_h = variation_loss_kernel_h
        return

    def get_content_features(self):
        '''
        compute the feature for the content image (compute the value of units at each layer for content image)
        :return: dict, key - layer name (e.g. 'conv1_1'), value - 4-D numpy array [1, height, width, channel]
        '''

        # construct the output dict object
        content_features = {}

        # compute the value of units at each layer for the content image
        graph = tf.Graph()
        with graph.as_default(), graph.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=self.shape)  # placeholder for the input image
            net = self.vgg.load_net(image)  # go through the vgg model
            mean_pix = np.reshape(self.vgg.mean_pix, (1, 1, 1, 3))
            content_image = self.content - mean_pix  # subtract the mean pix before put into vgg model
            for layer_name in self.content_layer_weights:
                # get the value of each specified layer with respect to content image
                content_features[layer_name] = net[layer_name].eval(feed_dict={image: content_image})

        # return the dict object
        return content_features

    def get_style_features(self):
        '''
        compute the feature for the style image (compute the value of gram matrix at each layer for style image)
        :return: dict, key - layer name (e.g. 'conv1_1'), value - 2-D numpy array [channel, channel]
        '''

        # construct the output dict object
        style_features = {}

        # compute the value of gram matrix at each layer for the style image
        graph = tf.Graph()
        with graph.as_default(), graph.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=self.shape)
            net = self.vgg.load_net(image)
            mean_pix = np.reshape(self.vgg.mean_pix, (1, 1, 1, 3))
            style_image = self.style - mean_pix  # subtract the mean pix before put into vgg model
            for layer_name in self.style_layer_weights:
                features = net[layer_name].eval(
                    feed_dict={image: style_image})  # original feature (value of units in a layer), numpy array
                _, height, width, channel = features.shape  # get the shape of a specific layer
                if self.style_demean:
                    # subtract the mean for each activation map
                    mean_vec = np.reshape(features.mean(axis=(0, 1, 2)), (1, 1, 1, channel))
                    features = features - mean_vec
                features = np.reshape(features, (-1, features.shape[3]))  # flatten the units at each activation map
                # features.shape = [height * width, channel]
                gram = features.T.dot(features) / (height * width * channel)  # compute the gram matrix
                # gram.shape = [channel, channel]
                # gram computation - 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'
                style_features[layer_name] = gram  # store the gram matrix for a specific layer

        # return the dict object
        return style_features

    def batch_norm(self, input_tensor, bn_name='1'):
        '''
        function for constructing batch normalization layer
        reference to https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/resnet.py
        :param input_tensor: 4-D tensor, [1, height, width, channel]
        :param bn_name: str, batch normalization name, used for naming variables under this batch normalization layer
        :return: 4-D tensor (after being normalized), [1, height, width, channel]
        '''

        # get the number of channels of input tensor
        channel = input_tensor.get_shape().as_list()[-1]

        # compute the mean and variance of each channel
        mean, variance = tf.nn.moments(input_tensor, axes=[0, 1, 2])

        # define beta and gamma, these two variable are trainable
        beta = tf.get_variable('beta_' + bn_name, channel, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma_' + bn_name, channel, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))

        # construct the batch normalization layer
        bn_layer = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, variance_epsilon=0.001)

        # return the tensor after the batch normalization
        return bn_layer

    def res_block(self, input_tensor, res_block_name):
        '''
        function for constructing a residual block
        :param input_tensor: 4-D tensor, [1, height, width, channel]
        :param res_block_name: str, residual block name. There might be multiple residual block, each one needs a name
        :return: 4-D tensor (after going through the residual block), [1, height, width, channel]
        '''

        # restore the input, which will directly go to the last step
        shortcut = input_tensor

        # get the number of channels of input tensor
        channel = input_tensor.get_shape().as_list()[-1]

        # first convolution layer
        w_conv1 = tf.get_variable(res_block_name + '_w_conv1',
                                  shape=[3, 3, channel, channel])  # weights for first convolution layer
        conv1 = tf.nn.conv2d(input_tensor, w_conv1, strides=[1, 1, 1, 1], padding='SAME')  # convolution layer

        # first batch normalization layer
        bn1 = self.batch_norm(conv1, bn_name=res_block_name + '_bn1')

        # first relu layer
        relu1 = tf.nn.relu(bn1)

        # second convolution layer
        w_conv2 = tf.get_variable(res_block_name + '_w_conv2',
                                  shape=[3, 3, channel, channel])  # weights for second convolution layer
        conv2 = tf.nn.conv2d(relu1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')

        # second batch normalization layer
        bn2 = self.batch_norm(conv2, bn_name=res_block_name + '_bn2')

        # add the residual and the input tensor
        output_tensor = shortcut + bn2

        # return the output tensor
        return output_tensor

    def down_sample(self, input_tensor):
        '''
        down sampling, the input image will go through a few convolution layers,
        which will reduce height & width and increase channels
        :param input_tensor: 4-D tensor, [1, height, width, 3(channel)]
        :return: 4-D tensor, [1, height_reduced, width_reduced, channel_increased]
        '''

        # set parameters - channels
        channel_input = input_tensor.get_shape().as_list()[-1]
        channel_conv1 = 32
        channel_conv2 = 64
        channel_conv3 = 128

        # first convolution layer
        w_conv1 = tf.get_variable('down_sample_w_conv1', shape=[9, 9, channel_input, channel_conv1])
        conv1 = tf.nn.conv2d(input_tensor, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        bn1 = self.batch_norm(conv1, bn_name='down_sample_bn1')
        relu1 = tf.nn.relu(bn1)
        output1 = relu1

        # second convolution layer
        w_conv2 = tf.get_variable('down_sample_w_conv2', shape=[3, 3, channel_conv1, channel_conv2])
        conv2 = tf.nn.conv2d(output1, w_conv2, strides=[1, 2, 2, 1], padding='SAME')
        bn2 = self.batch_norm(conv2, bn_name='down_sample_bn2')
        relu2 = tf.nn.relu(bn2)
        output2 = relu2

        # third convolution layer
        w_conv3 = tf.get_variable('down_sample_w_conv3', shape=[3, 3, channel_conv2, channel_conv3])
        conv3 = tf.nn.conv2d(output2, w_conv3, strides=[1, 2, 2, 1], padding='SAME')
        bn3 = self.batch_norm(conv3, bn_name='down_sample_bn3')
        relu3 = tf.nn.relu(bn3)
        output3 = relu3

        # output tensor
        output = output3
        return output

    def up_sample(self, input_tensor):
        '''
        up sampling, the input tensor will be expanded to a image like tensor,
        which is the transpose of down_sample function
        The output of this function should our mixed image.
        :param input_tensor: 4-D tensor, [1(batch), height, width, channel]
        :return: 4-D tensor, [1, height, width, 3(channel)]
        '''

        # set parameters
        batch_input, height_input, width_input, channel_input = input_tensor.get_shape().as_list()
        channel_conv1 = 64
        channel_conv2 = 32
        channel_conv3 = 3

        # first convolution layer
        w_conv1 = tf.get_variable('up_sample_w_conv1', shape=[3, 3, channel_conv1, channel_input])
        conv1 = tf.nn.conv2d_transpose(input_tensor, w_conv1,
                                       output_shape=[batch_input, height_input * 2, width_input * 2, channel_conv1],
                                       strides=[1, 2, 2, 1], padding='SAME')
        bn1 = self.batch_norm(conv1, bn_name='up_sample_bn1')
        relu1 = tf.nn.relu(bn1)
        output1 = relu1

        # second convolution layer
        w_conv2 = tf.get_variable('up_sample_w_conv2', shape=[3, 3, channel_conv2, channel_conv1])
        conv2 = tf.nn.conv2d_transpose(output1, w_conv2,
                                       output_shape=[batch_input, height_input * 4, width_input * 4, channel_conv2],
                                       strides=[1, 2, 2, 1], padding='SAME')
        bn2 = self.batch_norm(conv2, bn_name='up_sample_bn2')
        relu2 = tf.nn.relu(bn2)
        output2 = relu2

        # third convolution layer
        w_conv3 = tf.get_variable('up_sample_w_conv3', shape=[9, 9, channel_conv3, channel_conv2])
        conv3 = tf.nn.conv2d_transpose(output2, w_conv3,
                                       output_shape=[batch_input, height_input * 4, width_input * 4, channel_conv3],
                                       strides=[1, 1, 1, 1], padding='SAME')
        bn3 = self.batch_norm(conv3, bn_name='up_sample_bn3')
        # tanh3 = (tf.nn.tanh(bn3) + 1.0) / 2.0 * 255.0  # scaled tanh

        # output tensor
        output = bn3
        output = tf.clip_by_value(output, 0.0, 255.0)
        return output

    def transform_net(self, input_tensor):
        '''
        transform network, the image go through will be transformed into a mixed image
        :param input_tensor: 4-D tensor, [1, height, width, 3(channel)]
        :return: 4-D tensor, [1, height, width, 3(channel)], same shape as input_tensor
        '''

        # down sampling part
        down_sampled = self.down_sample(input_tensor)

        # residual block part
        residual_block_1 = self.res_block(down_sampled, res_block_name='res_block1')
        residual_block_2 = self.res_block(residual_block_1, res_block_name='res_block2')
        residual_block_3 = self.res_block(residual_block_2, res_block_name='res_block3')
        residual_block_4 = self.res_block(residual_block_3, res_block_name='res_block4')
        residual_block_5 = self.res_block(residual_block_4, res_block_name='res_block5')

        # up sampling part
        up_sampled = self.up_sample(residual_block_5)

        # return the output tensor
        output = up_sampled
        return output

    def train_model(self, learning_rate, beta1, beta2, epsilon, max_iteration, content_reproduce_iteration,
                    check_point):
        '''
        This is the function for training the transform network and restore the trained network
        :param learning_rate: float, base learning rate
        :param beta1: float, parameter for AdamOptimizer
        :param beta2: float, parameter for AdamOptimizer
        :param epsilon: float, parameter for AdamOptimizer
        :param max_iteration: int, maximum iteration times
        :param content_reproduce_iteration: int, training steps of only taking content image as mixed image (target image)
        :param check_point: int, after each check point number of iterations, the result will be printed and the trained network will be stored
        :return: None
        '''

        tf.reset_default_graph()
        with tf.Graph().as_default():
            # placeholders
            input_image = tf.placeholder(tf.float32, shape=self.shape, name='input_image')
            content_loss_weight = tf.placeholder(tf.float32, shape=(), name='content_loss_weight')
            style_loss_weight = tf.placeholder(tf.float32, shape=(), name='style_loss_weight')
            variation_loss_weight = tf.placeholder(tf.float32, shape=(), name='variation_loss_weight')

            # go through transform net
            mixed_image = self.transform_net(input_image)
            mixed_image = tf.identity(mixed_image, name='mixed_image')  # name mixed_image as 'mixed_image'

            # go through the vgg model
            mixed_image_demean = mixed_image - tf.reshape(tf.cast(self.vgg.mean_pix, tf.float32),
                                                          shape=(1, 1, 1, self.shape[3]))
            mixed_net = self.vgg.load_net(mixed_image_demean)

            # calculate loss
            loss_content = self.calculate_loss_content(mixed_net)
            loss_style = self.calculate_loss_style(mixed_net)
            loss_variation = self.calculate_loss_variation(mixed_image)
            loss_total = content_loss_weight * loss_content + style_loss_weight * loss_style + variation_loss_weight * loss_variation

            # summary statistics
            tf.summary.scalar('loss_content', loss_content)
            tf.summary.scalar('loss_style', loss_style)
            tf.summary.scalar('loss_variation', loss_variation)
            tf.summary.scalar('loss_total', loss_total)
            summary_loss = tf.summary.merge_all()

            # initialize optimizer
            lr = tf.placeholder(tf.float32, shape=())
            train_op = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(loss_total)

            # generate the variable used for record the loss data
            losses = []

            # variable used as current learning rate
            cur_lr = learning_rate

            # set random seed
            tf.set_random_seed(1)

            # train the model
            with tf.Session() as sess:
                # define the summary writer
                summary_writer = tf.summary.FileWriter(PATH + '/logs', sess.graph)

                # initialization of variables
                sess.run(tf.global_variables_initializer())

                # last i is used to decide when to adjust learning rate
                last_i = 0

                # time stamp
                starttime = time.time()

                # start training
                for i in range(max_iteration):
                    # training operation
                    if i < content_reproduce_iteration:
                        train_op.run(feed_dict={input_image: self.content,
                                                lr: cur_lr,
                                                content_loss_weight: self.content_weight,
                                                style_loss_weight: 0.,
                                                variation_loss_weight: 0.})
                    else:
                        train_op.run(feed_dict={input_image: self.content,
                                                lr: cur_lr,
                                                content_loss_weight: self.content_weight,
                                                style_loss_weight: self.style_weight,
                                                variation_loss_weight: self.variation_weight})

                    # summary
                    summary = sess.run(summary_loss, feed_dict={input_image: self.content,
                                                                content_loss_weight: self.content_weight,
                                                                style_loss_weight: self.style_weight,
                                                                variation_loss_weight: self.variation_weight})
                    summary_writer.add_summary(summary, i)

                    # output the loss
                    cur_loss_content = loss_content.eval(feed_dict={input_image: self.content}) * self.content_weight
                    if i < content_reproduce_iteration:
                        cur_loss_style = 0.
                        cur_loss_variation = 0.
                        cur_loss_total = loss_total.eval(feed_dict={input_image: self.content,
                                                                    content_loss_weight: self.content_weight,
                                                                    style_loss_weight: 0.,
                                                                    variation_loss_weight: 0., })
                    else:
                        cur_loss_style = loss_style.eval(feed_dict={input_image: self.content}) * self.style_weight
                        cur_loss_variation = loss_variation.eval(
                            feed_dict={input_image: self.content}) * self.variation_weight
                        cur_loss_total = loss_total.eval(feed_dict={input_image: self.content,
                                                                    content_loss_weight: self.content_weight,
                                                                    style_loss_weight: self.style_weight,
                                                                    variation_loss_weight: self.variation_weight})

                    # print result for current training step
                    if (check_point and ((i + 1) % check_point) == 0) or i == max_iteration - 1:
                        print('iter: {}, time: {} s, learning rate: {}'.format(
                            i + 1, round(time.time() - starttime), cur_lr))
                        print('    loss total: {}, loss content: {}, loss style: {}, loss variation: {}'.format(
                            round(cur_loss_total), round(cur_loss_content), round(cur_loss_style),
                            round(cur_loss_variation)))
                        starttime = time.time()

                    # record the loss data
                    losses.append(cur_loss_total)

                    # update the learning rate
                    if i < content_reproduce_iteration:
                        cur_lr = learning_rate * 2.0 ** (0 - i / content_reproduce_iteration * 5)
                        cur_lr = max(0.1, cur_lr)
                    else:
                        cur_lr = learning_rate * 2.0 ** (- i / 1000)
                        cur_lr = max(0.1, cur_lr)

                    # yield the mixed image and loss data
                    if (check_point and ((i + 1) % check_point) == 0) or i == max_iteration - 1:
                        image_out = mixed_image.eval(feed_dict={input_image: self.content})
                        image_out = image_out.reshape(self.shape[1:])
                        yield image_out, losses

                    # save trained model
                    if (check_point and ((i + 1) % check_point) == 0) or i == max_iteration - 1:
                        clear_dir(self.model_dir)
                        inputs = {'input_image': input_image}
                        outputs = {'mixed_image': mixed_image}
                        tf.saved_model.simple_save(sess, self.model_dir, inputs, outputs)

    def calculate_loss_content(self, mixed_net):
        '''
        compute the loss related to the content image and mixed image
        :param mixed_net: dict object, key - layer name (e.g. 'conv1_1'), value - 4-D tensor [1, height, width, channel]
        :return: 1-D tensor, float
        '''

        loss = 0.
        for layer_name in self.content_layer_weights:
            # reference to 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'
            _, height, width, channel = self.content_features[layer_name].shape
            size = height * width * channel
            loss += self.content_layer_weights[layer_name] * tf.reduce_sum(
                tf.pow(mixed_net[layer_name] - self.content_features[layer_name], 2)) / (size ** 0.5)
            # loss += tf.reduce_sum(tf.pow(mixed_net[layer_name] - self.content_features[layer_name], 2))
        return loss

    def calculate_loss_style(self, mixed_net):
        '''
        compute the loss related to the style image and mixed image (gram matrix)
        :param mixed_net: dict object, key - layer name (e.g. 'conv1_1'), value - 4-D tensor [1, height, width, channel]
        :return: 1-D tensor, float
        '''
        loss = 0.
        for layer_name in self.style_layer_weights:
            temp_features = mixed_net[layer_name]
            _, height, width, channel = temp_features.get_shape().as_list()
            size = height * width * channel
            if self.style_demean:
                mean_vec = tf.reshape(tf.reduce_mean(temp_features, axis=(0, 1, 2)), (1, 1, 1, channel))
                mixed_features = temp_features - mean_vec
            else:
                mixed_features = temp_features
            mixed_features = tf.reshape(mixed_features, (-1, channel))
            # reference to 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'
            mixed_gram = tf.matmul(tf.transpose(mixed_features),
                                   mixed_features) / size
            loss += self.style_layer_weights[layer_name] * tf.reduce_sum(
                tf.pow((mixed_gram - self.style_features[layer_name]), 2)) / 4.
        return loss

    def calculate_loss_variation(self, mixed_image):
        '''
        compute the loss related to the smoothness of the mixed_image
        :param mixed_image: 4-D tensor, [1, height, width, 3(channel)]
        :return: 1-D tensor, float
        '''
        # compute the size of difference point along the height dimension and width dimension
        height_size = np.prod([dim.value for dim in mixed_image[:, 1:, :, :].get_shape()])
        width_size = np.prod([dim.value for dim in mixed_image[:, :, 1:, :].get_shape()])

        # compute the difference along the height dimension and width dimension
        height_diff_squared = tf.square(mixed_image[:, 1:, :, :] - mixed_image[:, :mixed_image.shape[1] - 1, :, :])
        width_diff_squared = tf.square(mixed_image[:, :, 1:, :] - mixed_image[:, :, :mixed_image.shape[2] - 1, :])

        # compute the kernel for the difference
        height_kernel = tf.math.exp(-height_diff_squared / self.variation_loss_kernel_h)
        width_kernel = tf.math.exp(-width_diff_squared / self.variation_loss_kernel_h)

        # compute the loss along the height dimension and width dimension
        height_loss = tf.reduce_sum(tf.multiply(height_diff_squared, height_kernel)) / height_size
        width_loss = tf.reduce_sum(tf.multiply(width_diff_squared, width_kernel)) / width_size

        # compute the variation loss
        loss = height_loss + width_loss
        return loss
