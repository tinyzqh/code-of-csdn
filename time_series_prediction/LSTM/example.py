from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time #helper libraries

X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', 50, True)
#Step 2 Build Model
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
#Step 3 Train the model
model.fit(X_train, y_train, batch_size= 512, nb_epoch=1, validation_split=0.05)
# 点到点预测
# predictions = lstm.predict_point_by_point(model, X_test)
# lstm.plot_results(predictions, y_test)
# 滚动预测
# predictions = lstm.predict_sequence_full(model, X_test, 50)
# lstm.plot_results(predictions, y_test)
# 滑动窗口+滚动预测
predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
lstm.plot_results_multiple(predictions, y_test, 50)
