from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _train_parse_function(example_proto):
    #解析TFrecord数据集
    features = {'data': tf.FixedLenFeature([299* 299 * 3], dtype=tf.float32),
                'label': tf.FixedLenFeature([5], dtype=tf.float32)}
    features = tf.parse_single_example(example_proto, features=features)
    data_raw = features['data']
    data = tf.reshape(data_raw, (299, 299, 3))
    print(data.shape)
    labels = features['label']
    return data, labels

def build_traindataset(filename,batch_size=100):
    #建立训练数据集
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_train_parse_function,num_parallel_calls=12)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    return dataset

def build_valdataset(filename,batch_size):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_train_parse_function,num_parallel_calls=12)
    dataset = dataset.shuffle(buffer_size=10001)
    dataset = dataset.batch(batch_size)
    return dataset
