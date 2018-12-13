function test()
% 网络模型测试函数

batch_size =10000;

[test_x, test_y, ~, ~] = read_mnist();
load weights.mat

% 程序实现：网络构建，测试错误率计算==========================================
epoch = 1;
m = size(test_x, 1);
steps_per_epoch = 10000 / batch_size;
for i=1:epoch
    % 程序实现2：随机小批量训练集读取========================================
    % 提示：可以了解一下randperm函数
    %randperm是随机打乱一个数据序列
    kk = randperm(m);
    for j=1:steps_per_epoch
        train_data = test_x(kk((j - 1) * batch_size + 1 : j * batch_size), :);
%         train_data = train_x(j,:)
%         disp(size(train_data))
        train_label = test_y(kk((j - 1) * batch_size + 1 : j * batch_size), :);
        a_hidden = train_data*W1+b1;
        beta = 1;
% hidden_output = [1;sigmf(W1*train_data,[beta 0])];
        hidden_output = sigmf(a_hidden,[beta 0]);
        b_hidden = hidden_output*W2+b2;
        output = sigmf(b_hidden,[beta 0]);
        y = b_hidden;
        
        
        test_size = size(train_data,1);
        num_correct = 0;
        for i = [1:test_size]
            [max_unit_score, max_idx_score] = max(y(i,:));
            [max_unit_train_label, max_idx_train_label] = max(train_label(i,:));
            if (max_idx_score == max_idx_train_label)
                num_correct = num_correct +1;
         
            end
        end
        err = 1-num_correct/test_size(1);
        fprintf('test error: %.4f\n', err);
    end
end

% test_size = size(train_data,1);
% num_correct = 0;
% a_hidden = test_x*W1+b1;
% beta = 1;
% % hidden_output = [1;sigmf(W1*train_data,[beta 0])];
% hidden_output = sigmf(a_hidden,[beta 0]);
% b_hidden = hidden_output*W2+b2;
% % output = sigmf(b_hidden,[beta 0]);
% y = b_hidden
% %         [max_unit, max_idx] = max(score(1,:));
% for i = [1:test_size]
%     [max_unit_score, max_idx_score] = max(y(i,:));
%     [max_unit_train_label, max_idx_train_label] = max(train_label(i,:));
%     if (max_idx_score == max_idx_train_label)
%         num_correct = num_correct +1;
%     end
% end
%         
% err = 1-num_correct/test_size(1);




% 程序结束==================================================================
fprintf('test error: %.4f\n', err);
