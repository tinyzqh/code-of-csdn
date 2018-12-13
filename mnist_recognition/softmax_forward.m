function loss = softmax_forward(score, label)

% softmax损失函数前向传播
%
% :param score: 神经网络输出单元
% :param label: 训练样本真实标签
%
% :return loss: softmax损失

% 代码实现：softmax损失函数前向传播==========================================
% 参数1表示返回行数，参数若是为2则表明返回列数。
% m = size(score,1);
% a = exp(score);
% % sum是一个行向量，每一个元素是a矩阵的每一列的和，然后运用bsxfun（@rdivide， ，）
% % 是a矩阵的第i列的每个元素以sum（a）向量的第i个元素
% p = bsxfun(@rdivide,a,sum(a));
% c = log(p);
% loss = -(1/m)*sum(c);


% 
e = label - score;
mm = size(score,1);
% loss = 1/2 * sum(sum(e.^2)) / mm;
loss = 1/2 * sum(sum(e.^2)) / mm;
% % 
% e = batch_y - a{n};
% L(ll) = 1/2 * sum(sum(e.^2)) / mm;



% 代码结束==================================================================
end