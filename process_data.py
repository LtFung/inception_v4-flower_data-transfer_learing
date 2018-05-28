import glob

import numpy as np
import os
import tensorflow as tf
import multiprocessing as mp
def make_example(key1,key2):
	example = tf.train.Example(features=tf.train.Features(
        feature={
            'data':tf.train.Feature(float_list=tf.train.FloatList(value=key1)),
            'label':tf.train.Feature(float_list=tf.train.FloatList(value=key2))
        }
    ))
	return example
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.') and not f.endswith('.txt'):
            yield f

def get_class_names(path):
    class_names = list(listdir_nohidden(path))
    return class_names

def process_data(inpath = './data/flower_photos/',out_path='./data/process/',validation_percentage=10):
    class_names = get_class_names(inpath)#获取data的class_name
    print (class_names)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    #建立训练测试集
    train_path = out_path+'train'
    test_path = out_path + 'test'
    if not os.path.exists(train_path):
        os.mkdir(train_path)
        os.mkdir(test_path)
    for class_index, class_name in enumerate(class_names):
        print('processing dataset is ',class_name)
        dirname = inpath + class_name+'/'
        jpg_files = list(listdir_nohidden(dirname))
        total_name = len(jpg_files)
        file_indices = tuple(range(total_name))
        tfrecordtrain = os.path.join(train_path+'/',class_name+'_train.rfrecords')
        tfrecordtest = os.path.join(test_path+'/',class_name + '_test.rfrecords')
        train_writer = tf.python_io.TFRecordWriter(tfrecordtrain)
        test_writer = tf.python_io.TFRecordWriter(tfrecordtest)
        for file_index in file_indices:
            chance = np.random.randint(100)  # 随机产生100个数代表百分比
            if chance > validation_percentage:
                convert_one_file(dirname,jpg_files,class_index,train_writer,file_index)
            else:
                convert_one_file(dirname, jpg_files,class_index,test_writer,file_index)



        train_writer.close()
        test_writer.close()
        print('finished in ',class_name)



def convert_one_file(dirname,jpg_files,class_index,writer,file_index):
    file_name=dirname+jpg_files[file_index]
    image_raw_data = tf.gfile.FastGFile(file_name,'rb').read()
    label = np.zeros(5)
    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        resized = tf.image.resize_images(images = img_data,size=[299,299])
        data = sess.run(resized)
        data = data.reshape(-1)
        label[class_index]=1
        example = make_example(data, label)
        writer.write(example.SerializeToString())


process_data()
