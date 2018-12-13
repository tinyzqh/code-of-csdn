from gen_captcha import gen_captcha_lable_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL
from PIL import Image, ImageDraw, ImageFont
batch_size_real = 128
keep_prob_real = 0.75
lable, image = gen_captcha_lable_and_image()
print('验证码图像channel：',image.shape)  #(60,160,3)
Image_Height = 60
Image_Width = 160
MAX_Captcha = len(lable)
print('验证码文本最长字符数',MAX_Captcha)
#把彩色图像转化为灰度图像（彩色对验证码识别没有用）
def convert2gray(img):
    if len(img.shape)>2:
        gray = np.mean(img,-1)
        #上面转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1],img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行

"""
# 文本转向量
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
char_set_len = len(char_set)
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_Captcha:
        raise ValueError('验证码最长4个字符')
    vector = np.zeros(MAX_Captcha*char_set_len)
    def char2pos(c):
        if c == '_':
             k = 62
             return k
        k = ord(c)-48
        if k > 9:
            k = ord(c)-55
            if k > 35:
                k = ord(c)-61
                if k > 61:
                    raise ValueError('No Map')
        return k
    for i,c in enumerate(text):
        idx = i * char_set_len + char2pos(c)
        vector[idx] = 1
    return vector
#向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  #c/63
        char_idx = c % char_set_len
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)
"""
向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""
# 生成一个训练batch
def get_next_batch(batch_size=batch_size_real):
    batch_x = np.zeros([batch_size,Image_Height*Image_Width])
    batch_y = np.zeros([batch_size, MAX_Captcha*char_set_len])
    # 有时生成图像大小不是（60，160，3）
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_lable_and_image()
            if image.shape == (60,160,3):
                return text,image
    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
        batch_x[i,:] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i,:] = text2vec(text)
    return batch_x,batch_y

X = tf.placeholder(tf.float32,[None,Image_Height*Image_Width])
Y = tf.placeholder(tf.float32,[None, MAX_Captcha*char_set_len])
keep_prob = tf.placeholder(tf.float32)
# 定义CNN
def crack_captcha_cnn(w_alpha=0.01,b_alpha=0.1):
    x = tf.reshape(X,shape=[-1,Image_Height,Image_Width,1])
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,1,32]))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer

    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_Captcha * char_set_len]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_Captcha * char_set_len]))

    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out
# 训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1,MAX_Captcha, char_set_len])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_Captcha, char_set_len]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        acc_max = 0.0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: keep_prob_real})
            # print(step,loss_)
            print("The step is  %d and the loss is %f " % (step, loss_))
            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X:batch_x_test, Y:batch_y_test, keep_prob : 1})
                # print(step, acc)
                print('The step is %d and the accuracy is %f and the max accuracy is %f ' % (step, acc, acc_max))
                if acc > acc_max:
                    saver.save(sess, "./network_model/crack_capcha.model", global_step=step)

                    acc_max = acc
                    # break
            step = step + 1



# 使用训练的模型识别验证码：
def crack_captcha(captcha_image):
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # saver.restore(sess, "./network_model/crack_capcha.model")
        saver.restore(sess, tf.train.latest_checkpoint("E://code_of_ocr/network_model"))
        predict = tf.argmax(tf.reshape(output, [-1, MAX_Captcha, char_set_len]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_Captcha * char_set_len)
        i = 0
        for n in text:
            vector[i * char_set_len + n] = 1
            i += 1
        return vec2text(vector)
text, image_ = gen_captcha_lable_and_image()
image = convert2gray(image_)
image = image.flatten() / 255
predict_text = crack_captcha(image)
print("正确: {}  预测: {}".format(text, predict_text))

f = plt.figure()
ax = f.add_subplot(111)
ax.text(0.1,0.9,predict_text,ha='center',va='center',transform=ax.transAxes)
plt.imshow(image_)
plt.show()


