function ds = softmax_backward(score, label)
% softmax损失函数后向传播
%
% :param score: 神经网络输出单元
% :param label: 训练样本真实标签
%
% :return ds: 输出单元对损失的梯度

% 程序实现：softmax损失函数后向传播==========================================

% 计算输出单元的误差
[mm, ~] = size(label);
ds = (score - label).*score.*(1-score)/ mm;





% 程序结束==================================================================
end