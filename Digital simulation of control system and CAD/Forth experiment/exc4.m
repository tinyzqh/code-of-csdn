clear;
clear all;
clear close;
sa=2;sb=20;l=sb-sa;
e=1e-5;
x1=sa+(1-0.618)*l;
x2=sa+0.618*l;
q1=obj_asr(x1);
q2=obj_asr(x2);
count=0;
x_dot=[x1,x2];
while((sb-sa)/l>e)
    if(q1>q2)
        sa=x1;
        x1=x2;
        q1=q2;
        x2=sa+0.618*(sb-sa);
        x_dot=[x_dot,x2];
        q2=obj_asr(x2);
    else
        sb=x2;
        x2=x1;
        q2=q1;
        x1=sa+(1-0.618)*(sb- sa);
        x_dot=[x_dot,x1];
        q1=obj_asr(x1);
    end
end
Kn_opt=(x1+x2)/2;
q_min=obj_asr(Kn_opt);
opt_asr_plot(Kn_opt);