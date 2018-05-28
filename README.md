# inception_v4-flower_data-transfer_learing
inception_v4-flower_data-transferleaning
1.下载flower_data原始数据集：[http://download.tensorflow.org/example_images/flower_photos.tgz](url)
包含5种类别

2.数据集处理：生成每个class-label的TFrecord格式数据集，具体见process_data.py
部分代码：
image_raw_data = tf.gfile.FastGFile(file_name,'rb').read()#tensorflow中包含图片处理的API，可参照《tensoflow实战google学习框架》第七章
    label = np.zeros(5)
    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        resized = tf.image.resize_images(images = img_data,size=[299,299])
        data = sess.run(resized)
        data = data.reshape(-1)
        label[class_index]=1
        example = make_example(data, label)
        writer.write(example.SerializeToString())
        
3.inception_v4网络迁移：从v4网络的最后的PreLogitsFlatten层开始迁移,该层shape大小(None,1536)后面再接两层全连接实现5种flower分类。具体见model.py
   
   with slim.arg_scope(inception.inception_v4_arg_scope()):#引入tensorflow中inception_v4.py的模型，该模型采用的是slim轻量级框架，其中           end_points记录了v4中每层op_name,根据op_name定位到每一层
        end_points = inception.inception_v4(inputs=self.x, num_classes=1001, is_training=self.is_training,
                                                dropout_keep_prob=self.keep)
            # print(end_points[1])
        net = end_points[1]['PreLogitsFlatten']
        print(net)
        self.net = tf.stop_gradient(net)  # 这层与之前的层都不进行梯度更新

再设置后面两层的全连接：
   with tf.variable_scope('D'):
            fc1 = slim.fully_connected(self.net, 512, activation_fn=tf.nn.elu,
                                       scope='fc1')
            fc = slim.fully_connected(fc1, 48, activation_fn=tf.nn.sigmoid,
                                      scope='coding_layer')
            self.y = slim.fully_connected(fc, self.num_classes, activation_fn=tf.nn.softmax,
                                     scope='output')
   tvars = tf.trainable_variables()  
   
4.模型训练，首先从生成的TFrecord数据格式中解析为tf.data.Dataset形式生成迭代器,进行后面的数据迭代训练，TFrecord数据集解析具体见datd_utils.py。训练部分主要为迭代器初始化，和tf.ConfigProto()参数设置。训练了100个step，大约40分钟


5.总结：代码后续继续完善，有不足的地方还请指出～谢谢=.=
