import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class SIR:
    def __init__(self, **kwargs):
        """ Initializes and construct a SIR instance
        Args: 
            **kwargs: keyword arguments {
                beta: effective contact rate
                gamma: recovery rate
                p: probability of disease transmission in a contact between a suceptible and infectious subject 
                sigma: average number of contacts per person per time
                y0: initial number of subjects in each compartment
                time_steps: length of simulation period
                dt: length of simulation time steps
                }
        Returns: 
            A SIR instance
        """
        self.beta = kwargs.pop("beta", 1)
        self.p = kwargs.pop("p", 0.05)
        self.sigma = kwargs.pop("sigma", 0.1)
        self.gamma = self.p * self.sigma                # effective contact rate
        self.y0 = kwargs.pop("y0", [10000, 100, 0])

        self.time_steps = kwargs.pop("time_steps", 100)
        self.dt = kwargs.pop("dt", 1)
        self.t = np.linspace(0, self.time_steps, int(self.time_steps / self.dt)) 
        
    def initialize_matrix(self):
        pass

    def f(self, y, t):
        """ Calculates the partial derivatives
        Args:
            y: number of subjects across all compartments 
            t: numpy.linspace of the simulation horizon
        Returns: 
            Partial derivatives values
        """
        S, I, R = y
        N = S + I + R
        dSdt = -(self.beta * I * S)/ N 
        dIdt =  (self.beta * I * S)/ N  - self.gamma * I 
        dRdt =  self.gamma * I 

        return dSdt, dIdt, dRdt


    def simulate_epidemic(self):
        """ Simulates an epidemic in time
        returns:
            Array containing the value of each y for each desired time in t, with the inition value y0 in the first row
        """
        y = odeint(self.f, self.y0, self.t)
        return y

    def plot_simulation(self, y):
        """ Plots number of subjects in each compartments over time
        """
        S = y[:,0]
        I = y[:,1]
        R = y[:,2]

        plt.figure()
        plt.plot(self.t, S, label="S(t)")
        plt.plot(self.t, I, label="I(t)")
        plt.plot(self.t, R, label="R(t)")
        plt.legend()
        plt.show()