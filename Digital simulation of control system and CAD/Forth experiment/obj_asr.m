function q=obj_asr(Kn)
result = fun4(Kn);
y_inf = 296.7350;
q=100*(max(result(5,:))-y_inf)/y_inf; 
end