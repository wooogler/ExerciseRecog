import Batching1
import tensorflow as tf

SENSORS = 6
WINDOW = 100
CLASSES = 9
LEARNINGRATE = 0.001

x=tf.placeholder(tf.float32, shape=(None, SENSORS, WINDOW, 1))
y_=tf.placeholder(tf.float32, shape=(None, CLASSES))
global_step=tf.Variable(0)

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

pool1 = tf.nn.dropout(pool1_,0.7)

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

pool2 = tf.nn.dropout(pool2_,0.7)

flat = tf.contrib.layers.flatten(inputs=pool2)

fc_ = tf.contrib.layers.fully_connected(inputs=flat,
                                            num_outputs=100,
                                            normalizer_fn=tf.contrib.layers.batch_norm)

fc = tf.nn.dropout(fc_,0.8)

y = tf.contrib.layers.fully_connected(inputs=fc,
                                          num_outputs=CLASSES,
                                          normalizer_fn=tf.contrib.layers.batch_norm)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train = tf.contrib.layers.optimize_loss(loss,
                                            global_step,
                                            learning_rate=LEARNINGRATE,
                                            optimizer='Adam')


sess1=tf.Session()
sess1.run(tf.initialize_all_variables())

dir="training_data/"
data_set=["data_set1","data_set2","data_set3","data_set4","data_set5"]
ex=["ex1.txt","ex2.txt","ex3.txt","ex4.txt","ex5.txt","ex6.txt","ex7.txt","ex8.txt","ex9.txt","rest.txt"]

for i in range(50):
    x_batch, y_batch = Batching1.batching1(SENSORS,WINDOW,CLASSES,dir,data_set,ex)
    sess1.run(train, feed_dict={x:x_batch, y_:y_batch})
    if i%10==0:
        print(sess1.run(loss, feed_dict={x:x_batch, y_:y_batch}))

tf.train.Saver().save(sess1,'CNN1_Models/Model_Ex.ckpt')







