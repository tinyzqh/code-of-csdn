function q=opt_asr_plot(Kn)
result = fun4(Kn);
h=0.001; 
t_end=0.5;
t=0:h:t_end;
figure
subplot(4,1,1),plot(t,result(4,:)),grid,title('Id')
subplot(4,1,2),plot(t,result(3,:)),grid,title('Ud')
subplot(4,1,3),plot(t,result(2,:)),grid,title('ACR')
subplot(4,1,4),plot(t,result(1,:)),grid,title('ASR')
figure;
plot(t,result(5,:),'r');grid,title('n') 
end
