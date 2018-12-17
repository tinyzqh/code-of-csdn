function [uo]=satur(ui,c)
if (abs(ui)<=c)
    uo=ui;
elseif (ui>c)
    uo=c;
else
    uo=-c;
end
end

