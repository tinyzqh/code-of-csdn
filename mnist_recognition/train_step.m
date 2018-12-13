function [loss, y, W1, b1, W2, b2] = train_step(train_data, train_label, ...
                    W1, b1, ...
                    W2, b2, ...
                    lr)
% 一层隐藏层人工神经网络单步训练的实现
%
% :param train_data: 输入训练样本块
% :param train_label: 输入训练样本标签
% :param W1: 输入层权重矩阵
% :param b1: 输入层的偏置向量
% :param W2: 隐藏层的权重矩阵
% :param b2: 隐藏层的偏置向量
% :param lr: 学习率
%
% :return loss: 损失函数
% :return y: 网络输出单元
% :return W1: 梯度下降后的输入层权重矩阵
% :return b1: 梯度下降后的输入层的偏置向量
% :return W2: 梯度下降后的隐藏层的权重矩阵
% :return b2: 梯度下降后的隐藏层的偏置向量

% 程序实现1：人工神经网络的的前向传播================================
% 在matlab2016b中运行的
beta = 1;
a_hidden = train_data*W1+b1;
% hidden_output = [1;sigmf(W1*train_data,[beta 0])];
hidden_output = sigmf(a_hidden,[beta 0]);
b_hidden = hidden_output*W2+b2;
output = sigmf(b_hidden,[beta 0]);
y = output;
% % 计算输出单元的误差
% delta_c = (output - train_label).*output.*(1-output);
% % 计算隐藏层误差
% delta_h = (W2*delta_c).*output.*(1-output);
% delta_h = delta_h(2:end);
% W1 = W1 -lr*(train_data*delta_h);
% W2 = W2 -lr*(hidden_output*delta_c);


% 程序结束1=================================================================

% 损失函数计算
softmax_loss = softmax_forward(output, train_label);  % softmax损失
loss = softmax_loss;  % 总损失

% 损失函数后向传播
delta_c = softmax_backward(output,train_label);

% 程序实现2：人工神经网络的的后向传播=========================================
% 提示：每个参数求偏导应该是对损失求导，损失是一个标量，因此最后求偏导的结果应该
% 和原参数维度相同。
% 程序结束2=================================================================
b = delta_c*W2';
delta_h = (b).*hidden_output.*(1-hidden_output);
% delta_h = delta_h(2:end);
 
% 程序实现3：梯度下降========================================================
W1 = W1 -lr.*(train_data'*delta_h);
% dw = delta_h*ones(100,1);
% b1 = b1 -lr.*((delta_h*ones(100,1))');
W2 = W2 -lr.*(a_hidden'*delta_c);
% b2 = b2 -lr.*(delta_c);


% 程序结束3=================================================================
