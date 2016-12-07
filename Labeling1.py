import tensorflow as tf
import numpy as np

def labeling1(modeldir,rawdir,start_time,end_time_pair):
    SENSORS = 6
    WINDOW = 100
    CLASSES = 9

    x = tf.placeholder(tf.float32, shape=(None, SENSORS, WINDOW, 1))
    y_ = tf.placeholder(tf.float32, shape=(None, CLASSES))

    conv1 = tf.contrib.layers.convolution2d(inputs=x,
                                            num_outputs=32,
                                            kernel_size=[3, 3],
                                            stride=[1, 1],
                                            padding='VALID',
                                            normalizer_fn=tf.contrib.layers.batch_norm)

    pool1_ = tf.contrib.layers.max_pool2d(inputs=conv1,
                                          kernel_size=[1, 2],
                                          stride=[1, 2],
                                          padding='VALID')

    pool1 = tf.nn.dropout(pool1_, 1.0)

    conv2 = tf.contrib.layers.convolution2d(inputs=pool1,
                                            num_outputs=64,
                                            kernel_size=[3, 3],
                                            stride=[1, 1],
                                            padding='VALID',
                                            normalizer_fn=tf.contrib.layers.batch_norm)

    pool2_ = tf.contrib.layers.max_pool2d(inputs=conv2,
                                          kernel_size=[1, 2],
                                          stride=[1, 2],
                                          padding='VALID')

    pool2 = tf.nn.dropout(pool2_, 1.0)

    flat = tf.contrib.layers.flatten(inputs=pool2)

    fc_ = tf.contrib.layers.fully_connected(inputs=flat,
                                             num_outputs=100,
                                             normalizer_fn=tf.contrib.layers.batch_norm)

    fc = tf.nn.dropout(fc_, 1.0)

    y = tf.contrib.layers.fully_connected(inputs=fc,
                                          num_outputs=CLASSES,
                                          normalizer_fn=tf.contrib.layers.batch_norm)

    sess1 = tf.Session()
    sess1.run(tf.initialize_all_variables())
    tf.train.Saver().restore(sess1, modeldir)

    unlabeled_data = []
    labeled_data = []
    raw_data = (np.loadtxt(rawdir) + 32768) / 65536

    y_sliding = tf.argmax(y, 1)

    for i in range(len(start_time)):
        unlabeled_data.append(raw_data[start_time[i]:end_time_pair[i]])

    for j in range(len(start_time)):
        x_sliding = []

        for i in range(len(unlabeled_data[j]) - WINDOW):
            x_sliding.append(unlabeled_data[j][i:i + WINDOW])

        x_sliding = np.reshape(x_sliding, [-1, SENSORS, WINDOW, 1])
        y_labeling = sess1.run(y_sliding, feed_dict={x: x_sliding})

        labeled_data.append(y_labeling)

    return labeled_data

