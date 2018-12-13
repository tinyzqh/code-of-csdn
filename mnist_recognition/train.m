% function train(hidden_size, lr)
function train(hidden_size, lr)

% 完整训练过程实现
%
% :param hidden_size: 隐藏层单元个数
% :param lr: 学习率

% 程序实现1：参数设置（可以尝试自己修改）=====================================
hidden_size = 100;
lr = 0.5;
input_size = 784;
output_size = 10;
train_num = 60000;
epoch = 5;
batch_size = 100;
log_freq = 100;
steps_per_epoch = train_num / batch_size;
% 程序结束1=================================================================

% 读取mnist数据集
[test_x, test_y, train_x, train_y] = read_mnist();
% disp(size(train_x))
m = size(train_x, 1);
% 初始化权重值
% W1 = normrnd(0, 1e-2, [input_size, hidden_size]);
W1 = (rand(784,100)-0.5)*8*sqrt(6/(100+784));
% disp(size(W1))
b1 = rand(1, hidden_size);
% W2 = normrnd(0, 1e-2, [hidden_size, output_size]);
W2 = (rand(100,10)-0.5)*8*sqrt(6/(10+100));
b2 = rand(1, output_size);

% log
loss_list = zeros(epoch * steps_per_epoch, 1);
err_list = zeros(epoch * steps_per_epoch, 1);
times = 1;

% 迭代训练
for i=1:epoch
    % 程序实现2：随机小批量训练集读取========================================
    % 提示：可以了解一下randperm函数
    %randperm是随机打乱一个数据序列
    kk = randperm(m);
    for j=1:steps_per_epoch
        train_data = train_x(kk((j - 1) * batch_size + 1 : j * batch_size), :);
%         train_data = train_x(j,:)
%         disp(size(train_data))
        train_label = train_y(kk((j - 1) * batch_size + 1 : j * batch_size), :);
%         train_label = train_y(j,:)
%           disp(size(train_label))
   % 程序结束2==============================================================
        
        % 单步训练，权重更新
        [loss, score, W1, b1, W2, b2] = train_step(train_data, train_label, W1, b1, W2, b2, lr);

        % log
        % 程序实现3：计算错误率==============================================
        test_size = size(train_data,1);
        num_correct = 0;
%         [max_unit, max_idx] = max(score(1,:));
        for i = [1:test_size]
            [max_unit_score, max_idx_score] = max(score(i,:));
            [max_unit_train_label, max_idx_train_label] = max(train_label(i,:));
            if (max_idx_score == max_idx_train_label)
                num_correct = num_correct +1;
            end
        end
        
        err = 1-num_correct/test_size(1);
        % 程序结束3=========================================================
        loss_list(times) = loss;
        err_list(times) = err;
        if mod(times, log_freq) == 0
            fprintf('epoch: %2d | steps: %6d | loss: %6.4f | error: %.4f\n', i, times, loss, err);
        end
        times = times + 1;
    end
end

% 输出训练结果
save weights.mat W1 b1 W2 b2
log_steps = 1:epoch * steps_per_epoch;
figure();
plot(log_steps, loss_list)
title('loss curve')
xlabel('steps')
ylabel('loss')
figure();
plot(log_steps, err_list);
title('error curve')
xlabel('steps')
ylabel('error')

% 请记录下你的隐藏层个数和学习率=============================================
% hidden size = 100
% learning rate =  0.5
% 训练误差0.07
% 测试误差0.36
%==========================================================================