import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Case:
    def __init__(self, C, M, alpha, beta, epsilon, mu, gamma, sigma, omega, delta, p, time_steps=100.0, dt=1.0, params=None):
        self.t = np.linspace(0, time_steps, int(time_steps / dt))
        self.time_steps = time_steps
        self.dt = dt

        self.C = C
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.mu = mu
        self.gamma = gamma
        self.sigma = sigma
        self.omega = omega
        self.delta = delta
        self.p = p

        if params is not None:
            self.beta = params['beta']
            self.epsilon = params['epsilon']
            self.mu = params['mu']
            self.gamma = params['gamma']
            self.sigma = params['sigma']
            self.omega = params['omega']
            self.delta = params['delta']
            self.p = params['p']

        
    def initialize_matrix(self):
        pass

    def f(self, y, t):
        S, E, A, I, Q, D, R = y
        N = S + E + A + I + Q + D + R
        dS = self.mu * R - self.epsilon * self.M - self.beta/N * self.C * S * (I+A)
        dE = self.beta/N * self.C * S * (I+A) - self.sigma * E
        dI = self.p * self.sigma * E - self.alpha * I
        dA = (1 - self.p) * self.sigma * E - self.gamma * A
        dQ = self.alpha * I - self.omega * Q
        dD = self.delta * self.omega * Q
        dR = self.gamma * A + (1 - self.delta) * self.omega * Q + self.epsilon * self.M * S - self.mu * R
        return [dS, dE, dI, dA, dQ, dD, dR]


    def simulate_epidemic(self):
        S0 = 10000
        E0 = 0
        A0 = 0
        I0 = 100
        Q0 = 0
        D0 = 0
        R0 = 0

        y0 = [S0, E0, A0, I0, Q0, D0, R0]
        y = odeint(self.f, y0, self.t)
        return y

    def plot_simulation(self, y):
        S = y[:,0]
        E = y[:,1]
        A = y[:,2]
        I = y[:,3]
        Q = y[:,4]
        D = y[:,5]
        R = y[:,6]

        plt.figure()
        plt.plot(self.t, S, label="S(t)")
        plt.plot(self.t, E, label="E(t)")
        plt.plot(self.t, A, label="A(t)")
        plt.plot(self.t, I, label="I(t)")
        plt.plot(self.t, Q, label="Q(t)")
        plt.plot(self.t, D, label="D(t)")
        plt.plot(self.t, R, label="R(t)")
        plt.legend()
        plt.show()