import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from read_data import read_20180829
time,single1,single2,single3 = read_20180829()
# 需要预测的长度是多少
long_predict = 40
def svm_timeseries_prediction(c_parameter,gamma_paramenter):
    X_data = time
    Y_data = single1
    print(len(X_data))
    # 整个数据的长度
    long = len(X_data)
    # 取前多少个X_data预测下一个数据
    X_long = 1
    error = []
    svr_rbf = SVR(kernel='rbf', C=c_parameter, gamma=gamma_paramenter)
    # svr_rbf = SVR(kernel='rbf', C=1e5, gamma=1e1)
    # svr_rbf = SVR(kernel='linear',C=1e5)
    # svr_rbf = SVR(kernel='poly',C=1e2, degree=1)
    X = []
    Y = []
    for k in range(len(X_data) - X_long - 1):
        t = k + X_long
        X.append(Y_data[k:t])
        Y.append(Y_data[t + 1])
    y_rbf = svr_rbf.fit(X[:-long_predict], Y[:-long_predict]).predict(X[:])
    for e in range(len(y_rbf)):
        error.append(Y_data[X_long + 1 + e] - y_rbf[e])
    return X_data,Y_data,X_data[X_long+1:],y_rbf,error


X_data,Y_data,X_prediction,y_prediction,error = svm_timeseries_prediction(10,1)
figure = plt.figure()
tick_plot = figure.add_subplot(2, 1, 1)
tick_plot.plot(X_data, Y_data, label='data', color='green', linestyle='-')
tick_plot.axvline(x=X_data[-long_predict], alpha=0.2, color='gray')
# tick_plot.plot(X_data[:-X_long-1], y_rbf, label='data', color='red', linestyle='--')
tick_plot.plot(X_prediction, y_prediction, label='data', color='red', linestyle='--')
tick_plot = figure.add_subplot(2, 1, 2)
tick_plot.plot(X_prediction,error)
plt.show()