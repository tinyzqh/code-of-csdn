clc;
clear all;
close all;

NUM_BOXES = 163;
ALPHA = 0.5;             
GAMMA = 0.999;           
Q = zeros(NUM_BOXES,2);  
action = [10 -10];
MAX_FAILURES = 1000;
MAX_STEPS = 150000;
epsilon = 0;
steps = 0;
failures = 0;
thetaPlot = 0;
xPlot = 0;
theta = 0;
thetaDot = 0;
x = 0;
xDot = 0;
box = getBox4(theta,thetaDot,x,xDot);

while(steps<=MAX_STEPS && failures<+MAX_FAILURES)
    steps = steps + 1;
   
    if(rand>epsilon)       
        [~,actionMax] = max(Q(box,:));
        currentAction = action(actionMax);
    else                   
        currentAction = datasample(action,1);
    end
    
    actionIndex = find(action == currentAction); 
    
    [thetaNext,thetaDotNext,thetaacc,xNext,xDotNext] = cart_pole2(currentAction,theta,thetaDot,x,xDot);
    
    thetaPlot(end + 1) = thetaNext*180/pi;
    xPlot(end + 1) = xNext;
    newBox = getBox4(thetaNext,thetaDotNext,xNext,xDotNext);
    theta = thetaNext;
    thetaDot = thetaDotNext;
    x = xNext;
    xDot = xDotNext;
    if(newBox==163)
        r = -1;
        Q(newBox,:) = 0;
        figure(2);
        plot((1:length(thetaPlot)),thetaPlot,'-b');
        figure(3);
        plot((1:length(xPlot)),xPlot,'-b');

        thetaPlot = 0;
        xPlot = 0;
        theta = 0;
        thetaDot = 0;
        x = 0;
        xDot = 0;
        newBox = getBox4(theta,thetaDot,x,xDot);
        failures = failures + 1;
        fprintf('Trial %d was %d steps. \n',failures,steps);
        figure(1);
        plot(failures,steps,'-b');
        hold on;
        steps = 0;
    else
        r = 0;
    end
    Q(box,actionIndex) = Q(box,actionIndex) + ALPHA*(r + GAMMA*max(Q(newBox,:)) - Q(box,actionIndex));
    box = newBox;
end
if(failures == MAX_FAILURES)
    fprintf('Pole not balanced. Stopping after %d failures.',failures);
else
    fprintf('Pole balanced successfully for at least %d steps\n', steps);
    figure(1);
    plot(failures+1,steps,'-b');
    title('failures+1','FontSize',16); 
    hold on;
    figure(2);
    chouxi = 1:length(thetaPlot);
    plot((1:length(thetaPlot)),thetaPlot,'-b');
    title('thetaPlot','FontSize',16); 
    figure(3);
    plot((1:length(xPlot)),xPlot,'-b');
    title('xPlot','FontSize',16); 
    figure(4);
    plot((1:301),thetaPlot(1:301),'-b');
    title('thetaPlot-301','FontSize',16);
    hold on;
    figure(5);
    plot((1:301),xPlot(1:301),'-b');
    title('xPlot-301','FontSize',16);
    hold on;
end

    
    
        
    
    

    
    
    




