function [z]=fun(X)
Kn = 26.7;
taon = 0.03;
Ki = 0.269;
taoi = 0.067;
Ks = 76;
Ts = 0.00167;
R = 6.58;
Tl = 0.018;
Tm = 0.25;
Ce = 0.131;
alpha = 0.00337;
beta = 0.4;
Idl = 0;
P = [0, taon, Kn, Kn*taon;
     0, taoi, Ki, Ki*taoi;
     1, Ts, Ks, 0;
     1, Tl, 1/R, 0;
     0, Tm*Ce, R, 0];
A = diag(P(:,1));
B = diag(P(:,2));
C = diag(P(:,3));
D = diag(P(:,4));
WIJ = [1, 0, 1;
       1, 5, -alpha;
       2, 1, 1;
       2, 4, -beta;
       3, 2, 1;
       4, 3, 1;
       4, 5, -Ce;
       5, 4, 1];
m = length(WIJ(:,3));
W0 = zeros(5,1);
W = zeros(5,5);
for k = 1:m;
    if (WIJ(k,2 )==0);
        W0(WIJ(k, 1)) = WIJ(k,3);
    else W(WIJ(k, 1),WIJ(k, 2))=WIJ(k,3);
    end
end      
Q = B-D*W;
Qn = inv(Q);
R = C * W-A;
V1 = C * W0;
Ab = Qn * R;
Bb = Qn * V1;
z = Ab*X+Bb;
end