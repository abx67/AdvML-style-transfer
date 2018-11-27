import numpy as np
import tensorflow as tf
import os
from functools import reduce

PATH = '../'

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


class NeuralNetwork(object):
    # neural network used for style transfer, which includes the definition of loss function, optimization function, etc.
    def __init__(self, content, style, vgg, content_weight, style_weight, variation_weight):
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

        self.content_shape = (1,) + self.content.shape
        self.style_shape = (1,) + self.style.shape

        self.content_layer_weights = CONTENT_LAYER_WEIGHTS
        self.style_layer_weights = STYLE_LAYER_WEIGHTS

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
                features = np.reshape(features, (-1, features.shape[3]))
                gram = features.T.dot(features) / features.size  # TODO: find out why divide by the size
                style_features[layer_name] = gram
        return style_features

    def train_model(self, learning_rate, beta1, beta2, epsilon, max_iteration, check_point, init_image='content'):
        with tf.Graph().as_default():
            # initial image with random guess
            noise = np.random.normal(size=self.content_shape, scale=np.std(self.content) * 0.1)  # useless
            if init_image == 'random':
                init_image = tf.random_normal(self.content_shape)
            elif init_image == 'content':
                init_image = np.reshape(np.array(self.content - self.vgg.mean_pix), self.content_shape)
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
                summary_writer = tf.summary.FileWriter(PATH + 'logs', sess.graph)
                sess.run(tf.global_variables_initializer())

                for i in range(max_iteration):
                    train_step.run()
                    summary = sess.run(summary_loss)
                    summary_writer.add_summary(summary, i)
                    # save image
                    if (check_point and ((i + 1) % check_point) == 0) or i == max_iteration - 1:
                        image_out = mixed_image.eval()
                        image_out = image_out.reshape(self.content_shape[1:]) + self.vgg.mean_pix
                        print('iter: {}, loss total: {}, loss content: {}, loss style: {}, loss variation: {}'.format(
                            i + 1, loss_total.eval(), loss_content.eval(), loss_style.eval(), loss_variation.eval()
                        ))
                        yield i, image_out

    def calculate_loss_content(self, mixed_net):
        losses = []
        for layer_name in self.content_layer_weights:
            losses += [self.content_layer_weights[layer_name] * 2 * tf.nn.l2_loss(
                mixed_net[layer_name] - self.content_features[layer_name]) / self.content_features[
                           layer_name].size]  # TODO: find out why divide by the size
        return self.content_weight * reduce(tf.add, losses)

    def calculate_loss_style(self, mixed_net):
        losses = []
        for layer_name in self.style_layer_weights:
            _, height, width, channel = mixed_net[layer_name].get_shape()
            size = height.value * width.value * channel.value
            mixed_features = tf.reshape(mixed_net[layer_name], (-1, channel.value))
            mixed_gram = tf.matmul(tf.transpose(mixed_features),
                                   mixed_features) / size  # TODO: find out why divide by the size
            losses += [self.style_layer_weights[layer_name] * 2 * tf.nn.l2_loss(
                mixed_gram - self.style_features[
                    layer_name]) / self.style_features[layer_name].size]  # TODO: find out why divide by the size
        return self.style_weight * reduce(tf.add, losses)

    def calculate_loss_variation(self, mixed_image):
        height_size = np.prod([dim.value for dim in mixed_image[:, 1:, :, :].get_shape()])
        width_size = np.prod([dim.value for dim in mixed_image[:, :, 1:, :].get_shape()])
        loss = 2 * (tf.nn.l2_loss(
            mixed_image[:, 1:, :, :] - mixed_image[:, :mixed_image.shape[1] - 1, :, :]) / height_size + tf.nn.l2_loss(
            mixed_image[:, :, 1:, :] - mixed_image[:, :, :mixed_image.shape[2] - 1, :]) / width_size)
        return self.variation_weight * loss
