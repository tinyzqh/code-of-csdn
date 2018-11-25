clear;
clear all;
Kn=26.7;
tn=0.03;
Ki=0.269;
ti=0.067;
Ks=76;
Ts=0.00167;
R=6.58;
Tl=0.018;
Tm=0.25;
Ce=0.031;
Alpha=0.00337;
Beta=0.4;
A=[0 0 1 1 0];
B=[tn ti Ts Tl Tm*Ce];
C=[Kn Ki Ks 1/R R];
D=[Kn*tn Ki*ti 0 0 0];
c=8;
r=1;
W=[0 0 0 0 -Alpha;
    1 0 0 -Beta 0;
    0 1 0 0 0;
    0 0 1 0 -Ce;
    0 0 0 1 0];
W0=[1 0 0 0 0 ]';
block_num=5;
h=0.001; 
t_end=0.5;
t=0:h:t_end;
for k=1:block_num
    if (A(k)==0) %求积分或积分比例环节各系数
        FI(k)=1;
        FIM(k)=h*C(k)/B(k);
        FIJ(k)=h*h*C(k)/B(k)/2; 
        FIC(k)=1;
        FID(k)=0;
        if(D(k)~=0) %若为积分比例，修正fai d
            FID(k)=D(k)/B(k);
        end
    else
        FI(k)=exp(-h*A(k)/B(k)); % 求惯性或惯性比例环节各系数 
        FIM(k)=(1-FI(k))*C(k)/A(k);
        FIJ(k)=h*C(k)/A(k)-FIM(k)*B(k)/A(k); 
        FIC(k)=1;
        FID(k)=0;
        if(D(k)~=0)
            FIC(k)=C(k)/D(k)-A(k)/B(k);
            FID(k)=D(k)/B(k);
        end
    end
end
Y0=[0 0 0 0 0]'; 
n=length(t);
Y=Y0;
X=zeros(block_num,1);
result=Y;
Uk=zeros(block_num,1);
Ub=Uk; 
for m=1:(n-1)
    Ub=Uk;
    Uk=W*Y+W0*r;
    Uf=2*Uk-Ub;
    Udot=(Uk-Ub)/h;
%     X=FI'.*X+FIM'.*Uk;
%     Y=FIC'.*X+FID'.*Uf;
    X=FI'.*X+FIM'.*Uk+FIJ'.*Udot;
    Y=FIC'.*X+FID'.*Uf;
    Y(1)=satur(Y(1),c);
    result=[result,Y];
end
plot(t,result(1,:),t,result(2,:),t,result(3,:),t,result(4,:),t,result(5,:))
legend('y1','y2','y3','y4','y5')

       