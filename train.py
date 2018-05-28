from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data_utils import *

from model import *
import tensorflow as tf
import glob


FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string('logdir', './graph/', 'tensorboard_dir')
#tf.flags.DEFINE_string('modeldir', '../model/model.ckpt-10001', 'modelckpt_dir')
tf.flags.DEFINE_string('TFdataset', './data/process/train/', 'TFRecordDataset')
tf.flags.DEFINE_string('savedir', './model/', 'save running model')

class TrainConfig():
    img_size = 299
    img_channel = 3
    num_classes = 10
    total_nums = 3400
    lr = 0.01
    batch_size = 100
    decay_rate = 0.99
    train_data_list = glob.glob(os.path.join(FLAGS.TFdataset, '*rfrecords*'))

def make_dir(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path)

def main(unused_arg):
    #bulid train_dataset
    model_name = 'model.ckpt'
    config = TrainConfig()
    print('------------------------')
    print(config.train_data_list)
    train_dataset = build_traindataset(config.train_data_list, batch_size=config.batch_size)
    #train_dataset = build_traindataset(FLAGS.TFdataset, batch_size=config.batch_size)
    iterator_train = tf.contrib.data.Iterator.from_structure(train_dataset.output_types,
                                                             train_dataset.output_shapes)

    train_init_op = iterator_train.make_initializer(train_dataset)
    next_train = iterator_train.get_next()

    #统计数据集内容
    """
    with tf.Session() as sess:
        count = 0
        for i in range(1):
            sess.run(train_init_op)
            count = 0
            while True:
                try:
                    print(count)
                    data , label = sess.run(next_train)
                    print(data.shape)
                    #if count%10==0:
                    #  print(label)
                    count+=1
                    #print(count)
                    #if count%100 ==0:
                    #    print(data.shape)



                except:
                    break




    """
    model = modelconfig(config,iterator_train)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.intra_op_parallelism_threads = 24
    sess_config.inter_op_parallelism_threads = 24
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False


    merged = tf.summary.merge_all()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())



    with tf.Session(config=sess_config) as sess:
        print(FLAGS.logdir)
        LOGDIR = make_dir(FLAGS.logdir)
        sess.run(init)
        """
        ckpt = tf.train.get_checkpoint_state(FLAGS.savedir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
        """
        var_list = tf.global_variables()
        var_list_1 = []
        for var in var_list:  # 不加载 最后两层的参数，即重新训练
            if 'fc1' in var.name or 'coding_layer' in var.name or 'output' in var.name:
                # var_list_1.remove(var)
                continue
            var_list_1.append(var)
        var_list = None
        var_list_1.pop()

        saver = tf.train.Saver(var_list=var_list_1)
        saver.restore(sess, './model/inception_v4.ckpt')
        writer = tf.summary.FileWriter(FLAGS.logdir, graph=sess.graph)

        for i in range(10001):
            sess.run(train_init_op)
            count = 0
            while True:
                try:


                    count += 1

                    loss, acc, _, step, merge = sess.run([model.cost, model.accuracy,
                                                          model.train_op,
                                                          model.global_step, merged],feed_dict={model.is_training:False,model.keep:0.5})


                    writer.add_summary(merge, step)

                except:
                    break

            print('steps %d/5,epoch %d,loss is %f,acc is %f'
                          % (i, count, loss, acc))
            if i%100==0:
                saver.save(sess, os.path.join(FLAGS.savedir, model_name),
                           global_step=step)





if __name__ == '__main__':
    tf.app.run()

