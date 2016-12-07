import tensorflow as tf
import numpy as np
import Butterworth as bw
import matplotlib.pyplot as plt

def labeling2(modeldir,rawdir):
    SENSORS = 1
    WINDOW = 50
    CLASSES = 2

    x=tf.placeholder(tf.float32, shape=(None, SENSORS, WINDOW, 1))
    y_=tf.placeholder(tf.float32, shape=(None, CLASSES))
    global_step=tf.Variable(0)

    conv1 = tf.contrib.layers.convolution2d(inputs=x,
                                                num_outputs=32,
                                                kernel_size=[1, 3],
                                                stride=[1, 1],
                                                padding='VALID',
                                                normalizer_fn=tf.contrib.layers.batch_norm)

    pool1 = tf.contrib.layers.max_pool2d(inputs=conv1,
                                             kernel_size=[1, 2],
                                             stride=[1, 2],
                                             padding='VALID')

    conv2 = tf.contrib.layers.convolution2d(inputs=pool1,
                                                num_outputs=64,
                                                kernel_size=[1, 3],
                                                stride=[1, 1],
                                                padding='VALID',
                                                normalizer_fn=tf.contrib.layers.batch_norm)

    pool2 = tf.contrib.layers.max_pool2d(inputs=conv2,
                                             kernel_size=[1, 2],
                                             stride=[1, 2],
                                             padding='VALID')

    flat = tf.contrib.layers.flatten(inputs=pool2)

    fc1 = tf.contrib.layers.fully_connected(inputs=flat,
                                                num_outputs=128,
                                                normalizer_fn=tf.contrib.layers.batch_norm)

    y = tf.contrib.layers.fully_connected(inputs=fc1,
                                              num_outputs=CLASSES,
                                              normalizer_fn=tf.contrib.layers.batch_norm)

    sess2=tf.Session()
    tf.train.Saver().restore(sess2,modeldir)

    raw_data=(np.loadtxt(rawdir) + 32768) / 65536
    raw_data_x=bw.butterworth(6,40,0.8,raw_data[:,0])

    x_sliding = []
    y_sliding = tf.argmax(y,1)

    for i in range(len(raw_data_x)-WINDOW):
        x_sliding.append(raw_data_x[i:i+WINDOW])

    x_sliding = np.reshape(x_sliding,[-1, SENSORS, WINDOW, 1])
    y_sliding = sess2.run(y_sliding,feed_dict={x:x_sliding})

    return y_sliding
"""
    plt.subplot(211)
    plt.plot(range(len(y_sliding)),y_sliding, 'b.')
    plt.axis([0,len(y_sliding),-1,2])

    plt.subplot(212)
    plt.plot(range(len(raw_data_x)),raw_data_x, 'r.')
    plt.axis([0,len(raw_data_x)-WINDOW,0,1])

    plt.show()
"""
