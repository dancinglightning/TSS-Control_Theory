# Importing Libraries
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from ipywidgets import interact

# Defining Cache storage and g
g = 9.8
log1, log2 = [], []

def deriv(X, t, theta):
    dX = np.matmul(np.array([[0, 1], [0, 0]]), X) + np.array([0, (-5 * g / 7)])*np.radians(theta)
    return dX

def cutoff(val, limit=float('inf')):
    return min(limit, max(-limit, val))

# starting simulation

def solve(x=0):

    # Achieving setpoints
    Xsp = np.array([x, 0])  # [position, accelaration]

    # set initial conditions
    theta = 0
    t = 0

    # do simulation at fixed time steps dt
    dt = 0.1
    ti = 0.0
    tf = 10.0

    # control parameters
    g = 9.8
    kp = 8
    ki = 4
    kd = 3

    I_initial = 0  # Integral sum

    X0 = np.array([10, 5])  # Initial state of [postion, acceleration]
    X_initial = X0
    e_prev = Xsp - X_initial
    # PID control calculations
    e = Xsp - X_initial
    I_initial += dt*(e)
    theta -= cutoff(sum(kp*(e) + ki*I_initial + kd*(e - e_prev)/dt), 1)
    theta = cutoff(theta, 15)
    
    # log data and update state
    log1.append(theta)
    log2.append(X_initial)
    X_prev = X_initial
    X_initial = odeint(deriv,X0,[t,t+dt], args=(theta,))[-1]
    X_initial[0] = cutoff(X_initial[0], 300)

    # save data for PID calculations
    e_prev = e
    t += dt

    return X_initial[0] - X_prev[0]

