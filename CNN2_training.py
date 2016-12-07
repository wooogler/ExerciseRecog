import Batching2
import tensorflow as tf

SENSORS = 1
WINDOW = 50
CLASSES = 2
LEARNINGRATE = 0.001

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

pool1 = tf.nn.dropout(pool1,0.7)

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

pool2 = tf.nn.dropout(pool2,0.7)

flat = tf.contrib.layers.flatten(inputs=pool2)

fc1 = tf.contrib.layers.fully_connected(inputs=flat,
                                            num_outputs=128,
                                            normalizer_fn=tf.contrib.layers.batch_norm)

fc1 = tf.nn.dropout(fc1,0.8)

y = tf.contrib.layers.fully_connected(inputs=fc1,
                                          num_outputs=CLASSES,
                                          normalizer_fn=tf.contrib.layers.batch_norm)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train = tf.contrib.layers.optimize_loss(loss,
                                            global_step,
                                            learning_rate=LEARNINGRATE,
                                            optimizer='Adam')




sess2=tf.Session()
sess2.run(tf.initialize_all_variables())

dir="training_data2/"
no_ex=["no_ex1.txt", "no_ex2.txt"]
ex=["ex1.txt","ex2.txt","ex3.txt","ex4.txt","ex5.txt","ex6.txt","ex7.txt","ex8.txt","ex9.txt",
    "ex10.txt","ex11.txt","ex12.txt","ex13.txt","ex14.txt","ex15.txt","ex16.txt","ex17.txt","ex18.txt"]

for i in range(100):
    x_batch, y_batch = Batching2.batching2(SENSORS,WINDOW,CLASSES,dir,no_ex,ex)
    sess2.run(train, feed_dict={x:x_batch, y_:y_batch})
    if i%10==0:
        print(sess2.run(loss, feed_dict={x:x_batch, y_:y_batch}))
        tf.train.Saver().save(sess2,'CNN2_Models/Model_NoEx.ckpt')









