%  cart_pole:  Takes an Force and the current values of the
%              four state variables and updates their values by estimating
%              the state,
%              TAU seconds later.
function [thetaNext,thetaDotNext,thetaacc,xNext,xDotNext] = cart_pole2(force,theta,thetaDot,x,xDot)
%Parameters of inverted pendulum
GRAVITY = 9.8;
MASSCART = 1;
MASSPOLE = 0.1;
TOTAL_MASS = (MASSPOLE + MASSCART);
LENGTH = 0.5;
POLEMASS_LENGTH = (MASSPOLE*LENGTH);
TAU = 0.02;
FOURTHIRDS = 1.3333333333333;


temp = (force + POLEMASS_LENGTH*thetaDot*thetaDot*sin(theta))/TOTAL_MASS;

thetaacc = (GRAVITY*sin(theta) - cos(theta)*temp)/(LENGTH*(FOURTHIRDS - MASSPOLE*cos(theta)*cos(theta)/TOTAL_MASS));

xacc = temp - POLEMASS_LENGTH*thetaacc*cos(theta)/TOTAL_MASS;

%Update the four state variables, using Euler's method
xNext = x + TAU*xDot;
xDotNext = xDot + TAU*xacc;
thetaNext = theta + TAU*thetaDot;
thetaDotNext = thetaDot + TAU*thetaacc;

return;
