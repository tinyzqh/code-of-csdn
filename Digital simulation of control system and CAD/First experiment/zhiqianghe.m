clear;
clc;
% -----参数提取----------------
% L = 2;
% C = 0.000008;
% R = 280;
% num = [1];
% den = [L*C, R*C, 1];
% [z,p,k] = tf2zp(num,den)
% [A,B,C,D] = tf2ss(num,den)
% ---------------欧拉法-------------
% h = 0.0001;
% y = [0;0];
% for i = 0:1:300
%     t = i*h;
%     x(i+1) = t;
%     k1 = fun(y);
%     k2 = fun(y+h*k1);
%     y = y + (k1 +k2)*h/2;
%     output1(i+1) = y(1,1);
% end
% plot(x,output1)
%------------四阶-龙格-库塔-----------
h = 0.0001;
y = [0;0];
for i = 0:1:1000
    t = i*h;
    x(i+1) = t;
    k1 = fun(y);
    k2 = fun(y+h*k1/2);
    k3 = fun(y+h*k2/2);
    k4 = fun(y+h*k3);
    y = y + (k1 + 2*k2 +2*k3 + k4)*h/6;
    output1(i+1) = y(1,1);
end
plot(x,output1)


