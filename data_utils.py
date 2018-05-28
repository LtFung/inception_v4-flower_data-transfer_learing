import tensorflow as tf
import os



def _train_parse_function(example_proto):
    features = {'data': tf.FixedLenFeature([299* 299 * 3], dtype=tf.float32),
                'label': tf.FixedLenFeature([5], dtype=tf.float32)}
    features = tf.parse_single_example(example_proto, features=features)
    frames = features['data']


    frames = tf.reshape(frames, (299, 299, 3))
    print(frames.shape)
    labels = features['label']
    print(labels)
    # labels = tf.reshape(labels,(1000))
    return frames, labels



def build_traindataset(filename,batch_size=100):

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
