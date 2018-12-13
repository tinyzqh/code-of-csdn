function [test_x, test_y, train_x, train_y] = read_mnist()
% mnist数据集读取参考自：http://www.cnblogs.com/tiandsp/p/9042908.html

load mnist_uint8.mat;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

mu=mean(train_x);    
sigma=max(std(train_x),eps);
train_x=bsxfun(@minus,train_x,mu);  %每个样本分别减去平均值
train_x=bsxfun(@rdivide,train_x,sigma);  %分别除以标准差
end
