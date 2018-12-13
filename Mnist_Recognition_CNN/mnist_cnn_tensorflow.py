from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

y = tf.matmul(x,w) + b
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def weight_variable(shape):
    # 随机产生一个形状为shape的服从截断正态分布（均值为mean，标准差为stddev）的tensor。
    # 截断的方法根据官方API的定义为:
    # 如果单次随机生成的值偏离均值2倍标准差之外，就丢弃并重新随机生成一个新的数。
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # strides是指滑动窗口（卷积核）的滑动规则，包含4个维度，分别对应input的4个维度，即每次在input
    # tensor上滑动时的步长。其中batch和in_channels维度一般都设置为1，所以形状为[1, stride, stride, 1]。
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # ksize：滑动窗口（pool）的大小尺寸。batch和in_channels维度多设置为1
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# input x 是一个形状为[batch, in_height, in_width, in_channels]的tensor
# batch:每次batch数据的数量.in_channels:输入通道数量。
# in_height，in_width:输入矩阵的高和宽，如输入层的图片是28*28，则in_height和in_width就都为28。
x_image = tf.reshape(x, [-1, 28, 28,1])
# filter卷积核是一个形状为[filter_height, filter_width, in_channels, out_channels]的tensor
# 卷积核的高与宽,输入通道数量，输出通道的数量
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])


h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
Saver = tf.train.Saver()
try:
    Saver.restore(sess, tf.train.latest_checkpoint("E://code_of_ocr/MNIST_CNN_TENSORFLOW/network_model"))
    print('success add the model')
except:
    sess.run(tf.global_variables_initializer())
    print('error of add the model')

for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict= {
            x: batch[0], y_: batch[1], keep_prob:1.0})
        print('step %d , training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
Saver.save(sess, "E://code_of_ocr/MNIST_CNN_TENSORFLOW/network_model/crack_capcha.model")
print('test accuracy %g ' %accuracy.eval(feed_dict= {
    x:mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))
