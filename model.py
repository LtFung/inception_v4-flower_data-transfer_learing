from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception
import os
from tensorflow.python.platform import gfile
import argparse
import sys
import os
class TrainCofig():
    img_size = 299
    img_channel = 3
    num_classes = 5
    lr = 1e-4


class modelconfig(object):
    def __init__(self,config,iterator):
        self.iterator=iterator
        self.config=config
        self.is_training = tf.placeholder(tf.bool, name='MODE')
        self.keep = tf.placeholder(tf.float32)
        self.build_inputs()
        self.network()

    def build_inputs(self):
        img_size=229
        img_channel=3
        self.num_classes=5

        x, self.y_ = self.iterator.get_next()
        self.x = x/255
        #self.x = tf.placeholder(tf.float32, [None, img_size, img_size, img_channel], name='x')
        #self.y_ = tf.placeholder(tf.float32, [None, self.num_classes], 'y_')

    def print_activations(self, t):
        print(t.op.name, " ", t.shape.as_list())

    def network(self):
        with slim.arg_scope(inception.inception_v4_arg_scope()):
            end_points = inception.inception_v4(inputs=self.x, num_classes=1001, is_training=self.is_training,
                                                dropout_keep_prob=self.keep)
            # print(end_points[1])
        net = end_points[1]['PreLogitsFlatten']
        print(net)
        self.net = tf.stop_gradient(net)  # 这层与之前的层都不进行梯度更新

        print(net.shape)
        with tf.variable_scope('D'):
            fc1 = slim.fully_connected(self.net, 512, activation_fn=tf.nn.elu,
                                       scope='fc1')
            fc = slim.fully_connected(fc1, 48, activation_fn=tf.nn.sigmoid,
                                      scope='coding_layer')
            self.y = slim.fully_connected(fc, self.num_classes, activation_fn=tf.nn.softmax,
                                     scope='output')
        tvars = tf.trainable_variables()  # 获取所有可以更新的变量
        d_params = [v for v in tvars if v.name.startswith('D/')]
        # 只更新fc1,coding_layer与output的参数，其他的参数不更新
        self.cost = tf.losses.softmax_cross_entropy(onehot_labels=self.y_, logits=self.y)
        tf.summary.scalar('loss', self.cost)
        self.global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(self.config.lr, self.global_step,
                                        self.config.total_nums / self.config.batch_size, self.config.decay_rate)

        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss=self.cost, global_step=self.global_step,
                                                                                   var_list=d_params)
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
#config = TrainCofig()
#model = modelconfig(config)