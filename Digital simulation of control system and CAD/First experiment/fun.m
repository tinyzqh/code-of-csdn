function [z]=fun(X)
% R =4000;
% C =0.000001;
% L = 1;
R =280;
C =0.000008;
L = 2;
A = [0, 1/C;-1/L,-R/L];
B = [0;1/L];
z = A*X+B;
end