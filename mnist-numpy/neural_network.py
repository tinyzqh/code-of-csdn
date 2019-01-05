import numpy as np
import scipy.special as s
import matplotlib.pyplot as plt
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learninggrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learninggrate
        #产生一个高斯分布，均值0，方差为节点数目的-0.5次方。后面一个参数为大小
        self.w1 = np.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.w2 = np.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: s.expit(x)
        pass
    def train(self, inputs_list,targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.w1, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.w2, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.w2.T, output_errors)
        #反向传播
        self.w2 += self.lr*np.dot((output_errors*final_outputs*(1-final_outputs)), np.transpose(hidden_outputs))
        self.w1 += self.lr*np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)), np.transpose(inputs))
        pass
    #前向传播
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.w1, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.w2, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
training_data_file = open('mnist_train_100.csv','r',encoding='utf-8-sig')
training_data_list = training_data_file.readlines()
training_data_file.close()
for record in training_data_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) /255 * 0.99)+0.01
    targets = np.zeros(output_nodes)+0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass
test_data_file = open('mnist_test_10.csv','r',encoding='utf-8-sig')
test_data_list = test_data_file.readlines()
test_data_file.close()
# all_values = test_data_list[0].split(',')
# print('Value of real:',all_values[0])
# image_array = np.asfarray(all_values[1:]).reshape(28,28)
# predict = n.query((np.asfarray(all_values[1:]) / 255 * 0.9) + 0.01)
# print('Value of predict:',predict)
# plt.imshow(image_array,cmap='Greys', interpolation='None')
# plt.show()
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label =int(all_values[0])
    print(correct_label, 'correct label')
    inputs = (np.asfarray(all_values[1:]) / 255*0.99)+0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass
scorecard_array = np.asarray(scorecard)
print('performance = ', scorecard_array.sum() / scorecard_array.size)
