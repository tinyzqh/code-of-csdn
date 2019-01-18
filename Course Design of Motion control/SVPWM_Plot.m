figure(1)
Teta1=Theta(:,2);
Teta2=Teta1*180/pi;
Teta3=Teta2(1:8000);
Teta=Teta3(1:3:end);
Flux1=Amplitude(:,2);
Flux2=Flux1(1:8000);
Flux=Flux2(1:3:end);
x=Flux.*cos(Teta);
y=Flux.*sin(Teta);
plot(x,y);
xlabel('X');
ylabel('Y');
title('磁链方程');

figure(2)
plot(N.time(1:50000),N.signals.values(1:50000),'k');
grid on
xlabel('Times(s)');
ylabel('SVPMW扇区');
title('SVPMW扇区N计算结果');

figure(3)
plot(Tcm.time(1:50000),Tcm.signals.values(1:50000));
grid on
xlabel('Times(s)');
ylabel('时间Tcm');
title('SVPWM切换点时间Tcm');


figure(4)
plot(Ua.time(1:50000),Ua.signals.values(1:50000),'k');
grid on
xlabel('Times(s)');
ylabel('SVPWM输出相电压Ua');
title('SVPWM输出相电压Ua');

% figure(5)
% plot(Uab.time(1:50000),Uab.signals.values(1:50000),'k');
% grid on
% xlabel('Times(s)');
% ylabel('SVPWM输出线电压UAB');
% title('SVPWM输出线电压UAB');


figure(6)
plot(Iabc.time,Iabc.signals.values,'k');
grid on
xlabel('Times(s)');
ylabel('iabc(A)');
title('输出三相定子电流'); 

figure(7)
plot(Speed.time,Speed.signals.values,'k');
grid on
xlabel('Times(s)');
ylabel('n(rpm)');
title('输出电机转速');

figure(8)
plot(Te.time,Te.signals.values,'k');
grid on
xlabel('Times(s)');
ylabel('Te TL(N·m)');
title('输出电机转矩');

% figure(4)
% plot(Uabc.time,Uabc.signals.values,'k');
% grid on
% xlabel('Times(s)');
% ylabel('三相电压');
% title('SVPWM输出三相电压');
% 
% figure(8)
% plot(UABC.time,UABC.signals.values,'k');
% grid on
% xlabel('Times(s)');
% ylabel('SVPWM输出Uabc');
% title('SVPWM输出三相线电压');
