from cProfile import label
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from ipywidgets import interact

g = 9.8

# The following function gives the ordinary differential
# equation that our plant follows. Do not meddle with this.
def f(x, t, theta):
    return (x[1], (-5 * g / 7) * np.radians(theta))

def deriv(X, theta):
    dX = np.matmul(np.array([[0, 1], [0, 0]]), X) + np.array([0, (-5 * g / 7) * np.radians(theta)])*theta
    return dX

def sim(X,kp,ki,kd):   
    # setpoint
    Tsp = np.array([0, 0])

    # set initial conditions and cooling flow
    theta0 = 0
    theta = theta0

    # do simulation at fixed time steps dt
    dt = 0.05
    ti = 0.0
    tf = 8.0    

    # control parameters
    g = 9.8
    kp = 40
    ki = 80
    kd = 0
    beta = 0
    gamma = 0

    # create python list to log results
    log1, log2, log3 = [], [], []
    X0 = np.array([0, 0])
    X = X0

    # start simulation
    eP_ = beta*Tsp - X0
    eD_ = gamma*Tsp - X0
    eD__ = eD_ - X0

    for t in np.linspace(ti,tf,int((tf-ti)/dt)+1):
        # PID control calculations
        eP = beta*Tsp - X
        eI = Tsp - X
        eD = gamma*Tsp - X
        theta -= kp*(eP - eP_) + ki*dt*eI + kd*(eD - 2*eD_ + eD__)/dt
        
        # log data and update state
        log1.append(t)
        log2.append(theta)
        log3.append(X)
        X = odeint(deriv,X0,[t,t+dt])[-1]

        # save data for PID calculations
        eD__,eD_,eP_ = eD_,eD,eP

interact(sim,X = (0,0),kp = (0,80), ki=(0,160), kd=(0,10));