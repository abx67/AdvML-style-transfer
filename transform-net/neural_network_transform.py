import numpy as np
import tensorflow as tf
import os
from functools import reduce
import time

PATH = os.getcwd()


# LAYER_NAMES = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
#                'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
#                'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4',
#                'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4',
#                'pool5']


class NeuralNetwork(object):
    # neural network used for style transfer, which includes the definition of loss function, optimization function, etc.
    def __init__(self, content, style, vgg, content_weight, style_weight, variation_weight,
                 content_layer_weights, style_layer_weights, style_demean=False):
        # content - image, shape = (height, width, 3)
        # style - image, shape = (height, width, 3)
        # vgg - vgg object, definition see vgg.py
        # content_weight - scalar, weight for the loss of the content image
        # style_weight - scalar, weight for the loss of the style image
        # variation_weight - scalar, weight for the loss of variation of the mixed image

        self.content = content
        self.style = style
        self.vgg = vgg

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.variation_weight = variation_weight

        self.style_demean = style_demean

        self.content_shape = (1,) + self.content.shape
        self.style_shape = (1,) + self.style.shape

        self.content_layer_weights = content_layer_weights
        self.style_layer_weights = style_layer_weights

        self.content_features = self.get_content_features()
        self.style_features = self.get_style_features()
        self.mixed_image = None
        return

    def batch_norm(self, input, bn_name='1'):
        '''
        function for constructing batch normalziation layer
        :param input: 4-D tensor
        :return: the 4-D tensor after being normalized
        '''

        channel = input.get_shape().as_list()[-1]
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2])
        beta = tf.get_variable('beta_' + bn_name, channel, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma_' + bn_name, channel, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input, mean, variance, beta, gamma, variance_epsilon=0.001)

        return bn_layer

    def res_block(self, input, res_block_name, batch_norm=True):
        # residual block network
        # input - tensor with 4-D, [batch, height, width, channel]

        shortcut = input
        _, height, width, channel = input.get_shape()
        w_conv1 = tf.get_variable(res_block_name + '_w_conv1', shape=[3, 3, channel.value, channel.value])
        conv1 = tf.nn.conv2d(input, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        bn1 = self.batch_norm(conv1, bn_name=res_block_name + '_bn1')
        relu1 = tf.nn.relu(bn1)
        w_conv2 = tf.get_variable(res_block_name + '_w_conv2', shape=[3, 3, channel.value, channel.value])
        conv2 = tf.nn.conv2d(relu1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        bn2 = self.batch_norm(conv2, bn_name=res_block_name + '_bn2')
        add1 = shortcut + bn2
        # relu2 = tf.nn.relu(add1)
        output = add1
        return output

    def down_sample(self, input):

        # set parameters
        _, height, width, channel = input.get_shape()
        channel_conv1 = 32
        channel_conv2 = 64
        channel_conv3 = 128

        # first convolution layer
        w_conv1 = tf.get_variable('down_sample_w_conv1', shape=[9, 9, channel.value, 32])
        conv1 = tf.nn.conv2d(input, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        bn1 = self.batch_norm(conv1, bn_name='down_sample_bn1')
        relu1 = tf.nn.relu(bn1)

        # second convolution layer
        w_conv2 = tf.get_variable('down_sample_w_conv2', shape=[3, 3, channel_conv1, channel_conv2])
        conv2 = tf.nn.conv2d(relu1, w_conv2, strides=[1, 2, 2, 1], padding='SAME')
        bn2 = self.batch_norm(conv2, bn_name='down_sample_bn2')
        relu2 = tf.nn.relu(bn2)

        # third convolution layer
        w_conv3 = tf.get_variable('down_sample_w_conv3', shape=[3, 3, channel_conv2, channel_conv3])
        conv3 = tf.nn.conv2d(relu2, w_conv3, strides=[1, 2, 2, 1], padding='SAME')
        bn3 = self.batch_norm(conv3, bn_name='down_sample_bn3')
        relu3 = tf.nn.relu(bn3)

        # output tensor
        output = relu3
        return output

    def up_sample(self, input):
        # set parameters
        batch, height, width, channel = input.get_shape()
        channel_conv1 = 64
        channel_conv2 = 32
        channel_conv3 = 3

        # first convolution layer
        w_conv1 = tf.get_variable('up_sample_w_conv1', shape=[3, 3, channel_conv1, channel.value])
        conv1 = tf.nn.conv2d_transpose(input, w_conv1,
                                       output_shape=[batch.value, height.value * 2, width.value * 2, channel_conv1],
                                       strides=[1, 2, 2, 1], padding='SAME')
        bn1 = self.batch_norm(conv1, bn_name='up_sample_bn1')
        relu1 = tf.nn.relu(bn1)

        # second convolution layer
        w_conv2 = tf.get_variable('up_sample_w_conv2', shape=[3, 3, channel_conv2, channel_conv1])
        conv2 = tf.nn.conv2d_transpose(relu1, w_conv2,
                                       output_shape=[batch.value, height.value * 4, width.value * 4, channel_conv2],
                                       strides=[1, 2, 2, 1], padding='SAME')
        bn2 = self.batch_norm(conv2, bn_name='up_sample_bn2')
        relu2 = tf.nn.relu(bn2)

        # third convolution layer
        w_conv3 = tf.get_variable('up_sample_w_conv3', shape=[9, 9, channel_conv3, channel_conv2])
        conv3 = tf.nn.conv2d_transpose(relu2, w_conv3,
                                       output_shape=[batch.value, height.value * 4, width.value * 4, channel_conv3],
                                       strides=[1, 1, 1, 1], padding='SAME')
        bn3 = self.batch_norm(conv3, bn_name='up_sample_bn3')

        # output tensor
        output = tf.clip_by_value(tf.nn.relu(bn3), 0.0, 255.0)
        return output

    def get_content_features(self):
        content_features = {}
        graph = tf.Graph()
        with graph.as_default(), graph.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=self.content_shape)
            net = self.vgg.load_net(image)
            content = np.array(self.content - self.vgg.mean_pix)  # de-mean
            content = np.reshape(content, (1,) + content.shape)
            for layer_name in self.content_layer_weights:
                content_features[layer_name] = net[layer_name].eval(feed_dict={image: content})
        return content_features

    def get_style_features(self):
        style_features = {}
        graph = tf.Graph()
        with graph.as_default(), graph.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=self.style_shape)
            net = self.vgg.load_net(image)
            style = np.array(self.style - self.vgg.mean_pix)  # de-mean
            style = np.reshape(style, (1,) + style.shape)
            for layer_name in self.style_layer_weights:
                features = net[layer_name].eval(feed_dict={image: style})
                _, height, width, channel = features.shape
                if self.style_demean:
                    mean_vec = np.reshape(features.mean(axis=(0, 1, 2)), (1, 1, 1, channel))
                    features = features - mean_vec
                features = np.reshape(features, (-1, features.shape[3]))
                gram = features.T.dot(features) / (height * width * channel)  # TODO: find out why divide by the size
                # gram = features.T.dot(features)
                style_features[layer_name] = gram
        return style_features

    def train_model(self, learning_rate, beta1, beta2, epsilon, max_iteration, check_point, init_image='content'):
        with tf.Graph().as_default():
            # initial image with random guess
            input_image = tf.placeholder(tf.float32, shape=self.content_shape)
            content_image = np.reshape(np.array(self.content - self.vgg.mean_pix), self.content_shape)

            # go through transform net
            down_sample = self.down_sample(input_image)
            residual_block_1 = self.res_block(down_sample, res_block_name='res_block1')
            residual_block_2 = self.res_block(residual_block_1, res_block_name='res_block2')
            residual_block_3 = self.res_block(residual_block_2, res_block_name='res_block3')
            residual_block_4 = self.res_block(residual_block_3, res_block_name='res_block4')
            residual_block_5 = self.res_block(residual_block_4, res_block_name='res_block5')
            mixed_image = self.up_sample(residual_block_5)
            mixed_image = mixed_image - tf.reshape(tf.cast(self.vgg.mean_pix, tf.float32),
                                                   shape=(1, 1, 1, mixed_image.shape[3]))
            mixed_net = self.vgg.load_net(mixed_image)

            # calculate loss
            loss_content = self.calculate_loss_content(mixed_net)
            loss_style = self.calculate_loss_style(mixed_net)
            loss_variation = self.calculate_loss_variation(mixed_image)
            loss_total = loss_content + loss_style + loss_variation

            # summary statistics
            tf.summary.scalar('loss_content', loss_content)
            tf.summary.scalar('loss_style', loss_style)
            tf.summary.scalar('loss_variation', loss_variation)
            tf.summary.scalar('loss_total', loss_total)
            summary_loss = tf.summary.merge_all()

            # initialize optimization
            lr = tf.placeholder(tf.float32, shape=())
            train_step = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(loss_total)

            with tf.Session() as sess:
                summary_writer = tf.summary.FileWriter(PATH + '\\logs', sess.graph)
                sess.run(tf.global_variables_initializer())

                losses = []
                cur_lr = learning_rate
                for i in range(max_iteration):
                    starttime = time.time()
                    train_step.run(feed_dict={input_image: content_image, lr: cur_lr})

                    summary = sess.run(summary_loss, feed_dict={input_image: content_image})
                    summary_writer.add_summary(summary, i)
                    cur_loss_content = loss_content.eval(feed_dict={input_image: content_image})
                    cur_loss_style = loss_style.eval(feed_dict={input_image: content_image})
                    cur_loss_variation = loss_variation.eval(feed_dict={input_image: content_image})
                    cur_loss_total = loss_total.eval(feed_dict={input_image: content_image})
                    print('iter: {}, time: {} s, learning rate: {}'.format(
                        i + 1, round(time.time() - starttime), cur_lr))
                    print('    loss total: {}, loss content: {}, loss style: {}, loss variation: {}'.format(
                        round(cur_loss_total), round(cur_loss_content), round(cur_loss_style),
                        round(cur_loss_variation)))
                    cur_lr = max(0.01,
                                 min(max(cur_loss_content / cur_loss_style - 1, cur_loss_style / cur_loss_content - 1),
                                     10))
                    if cur_loss_total > 1e-6:
                        cur_lr = 10.0
                    losses.append(cur_loss_total)
                    self.mixed_image = mixed_image.eval(feed_dict={input_image: content_image})
                    # save image
                    if i == 0 or (check_point and ((i + 1) % check_point) == 0) or i == max_iteration - 1:
                        image_out = mixed_image.eval(feed_dict={input_image: content_image})
                        image_out = image_out.reshape(self.content_shape[1:]) + self.vgg.mean_pix
                        yield image_out, losses

    def calculate_loss_content(self, mixed_net):
        losses = []
        for layer_name in self.content_layer_weights:
            # losses += [self.content_layer_weights[layer_name] * 2 * tf.nn.l2_loss(
            #     mixed_net[layer_name] - self.content_features[layer_name]) / self.content_features[
            #                layer_name].size]  # TODO: find out why divide by the size
            losses += [self.content_layer_weights[layer_name] * 2 * tf.nn.l2_loss(
                mixed_net[layer_name] - self.content_features[layer_name])]
        return self.content_weight * reduce(tf.add, losses)
        # loss = 0.0
        # for layer_name in self.content_layer_weights:
        #     loss += tf.reduce_sum(tf.square(mixed_net[layer_name] - self.content_features[layer_name])) * \
        #             self.content_layer_weights[layer_name]
        # return self.content_weight * loss

    def calculate_loss_style(self, mixed_net):
        losses = []
        for layer_name in self.style_layer_weights:
            temp_features = mixed_net[layer_name]
            _, height, width, channel = temp_features.get_shape()
            size = height.value * width.value * channel.value
            if self.style_demean:
                mean_vec = tf.reshape(tf.reduce_mean(temp_features, axis=(0, 1, 2)), (1, 1, 1, channel.value))
                mixed_features = temp_features - mean_vec
            else:
                mixed_features = temp_features
            mixed_features = tf.reshape(mixed_features, (-1, channel.value))
            mixed_gram = tf.matmul(tf.transpose(mixed_features),
                                   mixed_features) / size  # TODO: find out why divide by the size
            losses += [self.style_layer_weights[layer_name] * 2 * tf.nn.l2_loss(
                mixed_gram - self.style_features[layer_name])]
        return self.style_weight * reduce(tf.add, losses)
        # loss = 0.0
        # for layer_name in self.style_layer_weights:
        #     _, height, width, channel = mixed_net[layer_name].get_shape()
        #     size = height.value * width.value * channel.value
        #     mixed_feature = tf.reshape(mixed_net[layer_name], (-1, channel))
        #     mixed_gram = tf.matmul(tf.transpose(mixed_feature), mixed_feature)
        #     loss += self.style_layer_weights[layer_name] / (4 * size ** 2) * tf.reduce_sum(
        #         tf.square(mixed_gram - self.style_features[layer_name]))
        # return self.style_weight * loss

    def calculate_loss_variation(self, mixed_image):
        height_size = np.prod([dim.value for dim in mixed_image[:, 1:, :, :].get_shape()])
        width_size = np.prod([dim.value for dim in mixed_image[:, :, 1:, :].get_shape()])
        loss = 2 * (tf.nn.l2_loss(
            mixed_image[:, 1:, :, :] - mixed_image[:, :mixed_image.shape[1] - 1, :, :]) / height_size + tf.nn.l2_loss(
            mixed_image[:, :, 1:, :] - mixed_image[:, :, :mixed_image.shape[2] - 1, :]) / width_size)
        return self.variation_weight * loss
