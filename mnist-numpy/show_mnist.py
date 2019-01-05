import numpy as np
import matplotlib.pyplot as plt
data_file = open('mnist_train_100.csv','r',encoding='utf-8-sig')
data_list = data_file.readlines()
data_file.close()
all_values = data_list[0].split(',')
print(all_values)
image_array = np.asfarray(all_values[1:]).reshape((28,28))
scaled_input = (np.asfarray(all_values[1:])/255*0.99)+0.01
onodes = 10
targets = np.zeros(onodes)+0.01
targets[int(all_values[0])] = 0.99
print(targets)
plt.imshow(image_array,cmap='Greys', interpolation='None')
plt.show()
print(len(data_list))
print(data_list[0])
