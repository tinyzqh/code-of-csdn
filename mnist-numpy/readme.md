### <center>神经网络前向传播、反向传播</center>

&emsp;&emsp;对神经网络有些了解的人可能都知道，神经网络其实就是一个输入$X$到输出`$Y$`的映射函数：`$f(X)=Y$`，函数的系数就是我们所要训练的网络参数`$W$`，只要函数系数确定下来，对于任何输入`$x_{i}$`我们就能得到一个与之对应的输出`$y_{i}$`，至于`$y_{i}$`是否符合我们预期，这就属于如何提高模型性能方面的问题了，这里不做讨论。

&emsp;&emsp;那么问题来了，现在我们手中只有训练集的输入`$X$`和输出`$Y$`，我们应该如何调整网络参数`$W$`使网络实际的输出`$f(X)= \widehat{Y}$`与训练集的`$Y$`尽可能接近？

&emsp;&emsp;**前向传播**：

<center><img src="https://img-blog.csdn.net/20180806104220539?/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzE2MTM3NTY5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="600" hegiht="600"></center>
<center>前向传播示意图</center>

&emsp;&emsp;记`$w_{jk}^{l}$`为第`$l-1$`层第`K`个神经元到第`$l$`层第`$j$`个神经元的权重，`$b_{j}^{l}$`为第`$l$`第`$j$`个神经元的偏置，`$a_{j}^{l}$`为第`$l$`第`$j$`个神经元的激活值（激活函数的输出）。不难看出`$a_{j}^{l}$`的值取决于上一层神经元的激活：
```math
a_{j}^{l}= \sigma(\sum_{K} w_{jk}^{l}a_{k}^{l-1}+b_{j}^{l})
```
&emsp;&emsp;将上式重写为矩阵形式：
```math
a_{j}^{l}= \sigma(\sum_{K} w_{}^{l}a_{}^{l-1}+b_{}^{l})
```
&emsp;&emsp;为了方便表示，记`$z^{l}=w_{}^{l}a_{}^{l-1}+b_{}^{l}$`为每一层的权重输入,则上式可表示为`$a^{l}=\sigma(z^{l})$`。利用式一层层计算网络的激活值，最终能够根据输入`$X$`得到相应的输出`$ \widehat{Y}$`。

&emsp;&emsp;**反向传播**：

&emsp;&emsp;反向传播过程中要计算`$\frac{\partial C}{\partial w}$`和`$\frac{\partial C}{\partial b}$`，我们先对代价函数做两个假设，以二次损失函数为例：
```math
C = \frac{1}{2n} \sum_{x}||y(x)-a^{L}(x)||^2
```
&emsp;&emsp;其中`$n$`为训练样本`$x$`的总数，`$y=y(x)$`为期望的输出，即ground truth，`$L$`为网络的层数，`$a^{L}(x)$`为网络的输出向量。

**假设1**：总的代价函数可以表示为单个样本的代价函数之和的平均：
```math
C=\frac{1}{n}\sum_{x}C_{x}

C_{x} = \frac{1}{2} ||y-a^{L}||^2
```

&emsp;&emsp;这个假设的意义在于，因为反向传播过程中我们只能计算单个训练样本的`$\frac{\partial C}{\partial w}$`和`$\frac{\partial C}{\partial b}$`，在这个假设下，我们可以通过计算所有样本的平均来得到总体的`$\frac{\partial C}{\partial w}$`和`$\frac{\partial C}{\partial b}$`。

**假设2**：代价函数可以表达为网络输出的函数`$costC=C(a^L)$`,比如单个样本`$x$`的二次代价函数可以写为：
```math
C_{x} = \frac{1}{2}||y-a^{L}||^2= \frac{1}{2} \sum_{j}(y_{j}-a_{j}^{L})^2
```

&emsp;&emsp;问题的关键在于权重`$w$`和偏置`$b$`的改变如何影响代价函数`C`。最终，这意味着我们需要计算出每个`$\frac{\partial C}{\partial w_{jk}^{l}}$`和`$\frac{\partial C}{\partial b_{j}^{l}}$`,在讨论基本方程之前，我们引入误差
`$\delta$`的概念，`$\delta_{j}^{l}$`表示第`$l$`层第`$j$`个单元的误差。

1. **输出层的误差方程**
```math
\delta_{j}^{l} = \frac{\partial C}{\partial z_{j}^{L}} = \frac{\partial C}{\partial a_{j}^{L}} \frac{\partial a_{j}^{L}}{\partial z_{j}^{L}} = \frac{\partial C}{\partial a_{j}^{L}} \sigma^{'}(z_{j}^{L})
```
&emsp;&emsp;当激活函数饱和时，即`$\sigma^{'}(z_{j}^{L}) \approx 0 $`,此时无论
`$\frac{\partial C}{\partial a_{j}^{L}}$`多大，最终的`$\delta_{j}^{l} \approx 0$`,输出神经元进入饱和区，停止学习。如果代价函数`$C_{x} = \frac{1}{2} \sum_{j}(y_{j}-a_{j}^{L})^2$`,则`$\frac{\partial C}{\partial a_{j}^{L}} = a_{j}^{L}-y_{j}$`,同理，对激活函数`$\sigma(z)$`求`$z_{j}^{L}$`的偏导即可求得`$\sigma^{'}(z_{j}^{L})$`。将上式写为矩阵的形式：
```math
\delta^{L} = \bigtriangledown_{a}C \bigodot \sigma^{'}(z^{L})
```
&emsp;&emsp;`$\bigodot$`为Hadamard积，即矩阵的点积。

2. **误差传递方程**
```math
\delta_{j}^{l} = \frac{\partial C}{\partial z_{j}^{l}} = \sum_{k} \frac{\partial C}{\partial z_{k}^{l+1}} \frac{\partial z_{k}^{l+1}}{\partial z_{j}^{l}} = \sum_{k} \delta_{k}^{l+1} \frac{\partial z_{k}^{l+1}}{\partial z_{j}^{l}}
```
&emsp;&emsp;因为`$z_{k}^{l+1} = \sum_{j}w_{kj}^{l+1}a_{j}^{l}+b_{k}^{l+1} = \sum_{j}w_{kj}^{l+1} \sigma(z_{j}^{l})+b_{k}^{l+1}$`。所以`$\frac{\partial z_{k}^{l+1}}{\partial z_{j}^{l}} = w_{kj}^{l+1} \sigma^{'}(z_{j}^{l})$`。由此可得：
```math
\delta_{j}^{l}=\sum_{k}w_{kj}^{l+1} \delta_{k}^{l+1} \sigma^{'}(z_{j}^{l})
```

3. **代价函数对偏置的改变率** 
```math
\frac{\partial C}{\partial w_{jk}^{l}} = \frac{\partial C}{\partial z_{j}^{l}} \frac{\partial z_{j}^{l}}{\partial w_{jk}^{l}} = \frac{\partial C}{\partial z_{j}^{l}} = \delta_{j}^{l}
```
&emsp;&emsp;这里因为`$z_{j}^{l} = \sum_{j}w_{kj}^{l}a_{j}^{l}+b_{k}^{l+1}$`所以`$\frac{z_{j}^{L}}{\partial b_{j}^{L}}=1$`。

4. **代价函数对权重的改变率**：

```math
\frac{\partial C}{\partial w_{jk}^{l}} = \frac{\partial C}{\partial z_{j}^{l}} \frac{\partial z_{j}^{L}}{\partial w_{jk}^{l}} = \frac{\partial C}{\partial z} = a_{k}^{l-1}\delta_{j}^{l}
```

&emsp;&emsp;可以简写为：
```math
\frac{\partial C}{\partial w} = a_{in} \delta_{out}
```
&emsp;&emsp;不难发现，当上一层激活输出接近0的时候，无论返回的误差有多大`$\frac{\partial C}{\partial w}$`的改变都很小，这也就解释了为什么神经元饱和不利于训练。

&emsp;&emsp;从上面的推导我们不难发现，当输入神经元没有被激活，或者输出神经元处于饱和状态，权重和偏置会学习的非常慢，这不是我们想要的效果。这也说明了为什么我们平时总是说激活函数的选择非常重要。

<center>

[上文原文](https://blog.csdn.net/qq_16137569/article/details/81449209)
</center>

<center>

### 吴恩达老师的梯度下降算法学习笔记

</center>

&emsp;&emsp;假设样本只有两个特征`$x_{1}$`和`$x_{2}$`。因此`$z$`的计算公式为：`$z=w_{1}x_{1}+w_{2}x_{2}+b$`。回想一下逻辑回归的公式定义如下：
```math
\widehat{y}=a=\sigma(z)
```
&emsp;&emsp;其中`$z=w^{T}x+b$`,`$\sigma(z)=\frac{1}{1+e^{-z}}$`。代价函数可以表示为：
```math
J(w,b)=\frac{1}{m} \sum_{i=1}^{m}L(\widehat{y}^{i}, y^{i}) = -\frac{1}{m}\sum_{i=1}^{m} y^{i}log \widehat{y}+(1-y)^{i}log(1-\widehat{y})
```
&emsp;&emsp;我们如何理解上式得这个公式呢？假设我们的标签是1，那么我们只会剩下上公式的前半部份，并且预测值越接近1值也越大；假设我们的标签是0，那么我们只会剩下后半部分，并且我们的预测值越接近0，值越大。那么再在前面加上一个负号就会使这个值越小，越接近真实标签。从而得到代价函数。

&emsp;&emsp;为了使得逻辑回归中最小化代价函数`$L(a,y)$`,我们需要做的仅仅是修改参数`$w$`和`$b$`。其中`$a=\widehat{y}$`,现在让我们来讨论通过反向计算出导数:因为我们想要计算出代价函数`$L(a,y)$`的导数，首先我们需要反向计算出代价函数`$L(a,y)$`关于a的导数。通过微积分得到：
```math
\frac{dL{(a,y)}}{da} = \frac{-y}{a} + \frac{(1-y)}{(1-a)}
```

&emsp;&emsp;因为，`$\frac{da}{dz} = a(1-a)$`所以`$\frac{dL(a,y)}{dz} = \frac{dL}{dz} = (\frac{dL}{da})(\frac{da}{dz}) = (-\frac{y}{a}+\frac{(1-y)}{(1-a)})a(1-a)$`。之后就可以进行反向求导：
```math
dw_1 = \frac{1}{m} \sum_{i}^{m}x_{1}^{i}(a^{i}-y^{i})

dw_2 = \frac{1}{m} \sum_{i}^{m}x_{2}^{i}(a^{i}-y^{i})

db = \frac{1}{m} \sum_{i}^{m}(a^{i}-y^{i})
```
&emsp;&emsp;之后更新`$w_{1} = w_{1} - \alpha dw_{1}$`,更新`$w_2=w_2-\alpha dw_2$`,更新`$b = b - \alpha db$`。

```python
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

```




