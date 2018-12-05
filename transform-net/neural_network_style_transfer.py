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
        return

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
                gram = features.T.dot(features) / (height * width)  # TODO: find out why divide by the size
                # gram = features.T.dot(features)
                style_features[layer_name] = gram
        return style_features

    def train_model(self, learning_rate, beta1, beta2, epsilon, max_iteration, check_point, init_image='content'):
        with tf.Graph().as_default():
            # initial image with random guess
            noise = np.random.normal(size=self.content_shape, scale=np.std(self.content) * 0.1)  # useless
            if init_image == 'random':
                init_image = tf.random_normal(self.content_shape) * 100.0
            elif init_image == 'content':
                init_image = np.reshape(np.array(self.content - self.vgg.mean_pix), self.content_shape)
            elif init_image == 'black':
                init_image = np.reshape(np.zeros(shape=self.content_shape[1:], dtype=float) - self.vgg.mean_pix,
                                        self.content_shape)
            mixed_image = tf.Variable(init_image, dtype=tf.float32)
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
            train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss_total)

            with tf.Session() as sess:
                summary_writer = tf.summary.FileWriter(PATH + '\\logs', sess.graph)
                sess.run(tf.global_variables_initializer())

                losses = []
                for i in range(max_iteration):
                    starttime = time.time()
                    train_step.run()
                    summary = sess.run(summary_loss)
                    summary_writer.add_summary(summary, i)
                    print(
                        'iter: {}, time: {} s, loss total: {}, loss content: {}, loss style: {}, loss variation: {}'.format(
                            i + 1, round(time.time() - starttime), round(loss_total.eval()), round(loss_content.eval()),
                            round(loss_style.eval()), round(loss_variation.eval())
                        ))
                    losses.append(loss_total.eval())
                    # save image
                    if i == 0 or (check_point and (
                            (i + 1) % check_point) == 0) or i == max_iteration - 1 or loss_total.eval() == 0.0:
                        image_out = mixed_image.eval()
                        image_out = image_out.reshape(self.content_shape[1:]) + self.vgg.mean_pix
                        yield image_out, losses
                    if loss_total.eval() == 0.0:
                        break

    def calculate_loss_content(self, mixed_net):
        losses = []
        for layer_name in self.content_layer_weights:
            losses += [self.content_layer_weights[layer_name] * 2 * tf.nn.l2_loss(
                mixed_net[layer_name] - self.content_features[layer_name]) / self.content_features[
                           layer_name].size]  # TODO: find out why divide by the size
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
            size = height.value * width.value
            if self.style_demean:
                mean_vec = tf.reshape(tf.reduce_mean(temp_features, axis=(0, 1, 2)), (1, 1, 1, channel.value))
                mixed_features = temp_features - mean_vec
            else:
                mixed_features = temp_features
            mixed_features = tf.reshape(mixed_features, (-1, channel.value))
            mixed_gram = tf.matmul(tf.transpose(mixed_features),
                                   mixed_features) / size  # TODO: find out why divide by the size
            losses += [self.style_layer_weights[layer_name] * 2 * tf.nn.l2_loss(
                mixed_gram - self.style_features[
                    layer_name]) / self.style_features[layer_name].size]  # TODO: find out why divide by the size
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
