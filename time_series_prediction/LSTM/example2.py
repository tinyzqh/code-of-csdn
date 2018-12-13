import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
def normalise_windows(window_data): # 数据全部除以最开始的数据再减一
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read() # 读取文件中的数据
    data = f.split('\n') # split() 方法用于把一个字符串分割成字符串数组，这里就是换行分割
    sequence_lenghth = seq_len + 1 # #得到长度为seq_len+1的向量，最后一个作为label
    result = []
    for index in range(len(data)-sequence_lenghth):
        result.append(data[index : index+sequence_lenghth]) # 制作数据集，从data里面分割数据
    if normalise_window:
        result = normalise_windows(result)
    result = np.array(result) # shape (4121,51) 4121代表行，51是seq_len+1
    row = round(0.9*result.shape[0]) # round() 方法返回浮点数x的四舍五入值
    train = result[:int(row), :] # 取前90%
    np.random.shuffle(train) # shuffle() 方法将序列的所有元素随机排序。
    x_train = train[:, :-1] # 取前50列，作为训练数据
    y_train = train[:, -1]  # 取最后一列作为标签
    x_test = result[int(row):, :-1] # 取后10% 的前50列作为测试集
    y_test = result[int(row):, -1] # 取后10% 的最后一列作为标签
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # 最后一个维度1代表一个数据的维度
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return [x_train, y_train, x_test, y_test]

x_train, y_train, x_test, y_test = load_data('./sp500.csv', 50, True)
model = Sequential()
model.add(LSTM(input_dim = 1, output_dim=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences= False))
model.add(Dropout(0.2))
model.add(Dense(output_dim = 1))
model.add(Activation('linear'))
start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print ('compilation time : ', time.time() - start)
model.fit(x_train, y_train, batch_size= 512, nb_epoch=1, validation_split=0.05)
import warnings
warnings.filterwarnings("ignore")
from numpy import newaxis
def predict_sequences_multiple(model, data, window_size, prediction_len):
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)): # 定滑动窗口的起始点
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len): # 与滑动窗口一样分析
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
predictions = predict_sequences_multiple(model, x_test, 50, 50)

import matplotlib.pylab as plt
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
plot_results_multiple(predictions, y_test, 50)