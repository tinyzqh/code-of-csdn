# https://blog.csdn.net/Jerr__y/article/details/61195257
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
tf.set_random_seed(1)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
lr = 0.001
training_iters = 100000
batch_size = 128
n_input = 28 #列 输入的图片数据为28*28
# 时序持续长度为28，即每做一次预测，需要先输入28行
n_step = 28 #行
n_hidden_unites = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None,n_step,n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
# weights = {
#     # (28, 128)
#     'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
#     # (128, 10)
#     'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))

weights = {
    'in' : tf.Variable(tf.random_normal([n_input, n_hidden_unites])),
    'out' : tf.Variable(tf.random_normal([n_hidden_unites, n_classes]))
}
biases = {
    'in' : tf.Variable(tf.constant(0.1, shape=[n_hidden_unites,])),
    'out' : tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    X_ = tf.reshape(X, [-1,n_input]) # 将数据转化为只有28列，不管多少行（其实会有128*28行）
    X_in = tf.matmul(X_, weights['in']) + biases['in'] #weights['in']是28*128的矩阵
    #上面的矩阵相乘相当于[128*28, 28]的矩阵乘以[28, 128的矩阵]，得到[128*28, 128]的矩阵
    # 将其转化回三维结构，如下代码所示，将变为[128,28,128]的数据。第一个128表示batch
    X_in_ = tf.reshape(X_in, [-1, n_step, n_hidden_unites])

    # basic LSTM Cell
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        # forget_bias:初始的遗忘值，最开始我们不希望他遗忘。
        # state_is_tuple=True：生成的state是一个元祖(c_state, h_state)
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unites, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_unites)
        # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # outputs是一个list，每一步的output都存在其中
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in_, initial_state=init_state, time_major=False)
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)
    return results
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
Saver = tf.train.Saver()

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    try:
        Saver.restore(sess, tf.train.latest_checkpoint("E://code_of_ocr/code_of_rnn/code_of_rnn_mnist/network_model"))
        print('success add the model')
    except:
        sess.run(tf.global_variables_initializer())
        print('error of add the model')
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_step, n_input])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1
    Saver.save(sess, "E://code_of_ocr/code_of_rnn/code_of_rnn_mnist/network_model/crack_capcha.model")