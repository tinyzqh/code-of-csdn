clear all;
clear;
h = 0.001;
y = [0;0;0;0;0];
x = [];
outputy1 = [];
outputy2 = [];
outputy3 = [];
outputy4 = [];
outputy5 = [];
for i = 0:1:1500
    t = i*h;
    x(i+1) = t;
    k1 = fun(y);
    k2 = fun(y+h*k1/2);
    k3 = fun(y+h*k2/2);
    k4 = fun(y+h*k3);
    y = y + (k1 + 2*k2 +2*k3 + k4)*h/6;
    outputy1(i+1) = y(1,1);
    outputy2(i+1) = y(2,1);
    outputy3(i+1) = y(3,1);
    outputy4(i+1) = y(4,1);
    outputy5(i+1) = y(5,1);
end
plot(x,outputy1,x,outputy2,x,outputy3,x,outputy4,x,outputy5)
legend('y1','y2','y3','y4','y5')



