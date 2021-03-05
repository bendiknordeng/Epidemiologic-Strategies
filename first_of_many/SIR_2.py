import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------------------------------
# SIR class represents a compartemental model that models the spread of an infectious disease.
# ------------------------------------------------------------------------------------------------------

class SIR:
    def __init__(self, N, beta, gamma, p, sigma, time_steps=100.0, dt=1.0, **kwargs):
        """ Initializes and construct a SIR instance
        Args: 
            beta: effective contact rate
            gamma: recovery rate
            p: probability of disease transmission in a contact between a suceptible and infectious subject 
            sigma: average number of contacts per person per time
            time_steps: length of simulation period
            dt: length of simulation time steps
            N: total population size
            kwargs: keyword arguments
        Returns: 
            A SIR instance
        """
        self.time_steps = time_steps
        self.dt = dt
        self.t = np.linspace(0, self.time_steps, int(self.time_steps / self.dt)) 
        self.beta = beta    
        self.gamma = gamma  
        self.p = p           
        self.sigma = sigma
        self.N  = N

        if kwargs is not None:
            self.beta = kwargs.pop('beta', None)
            self.gamma = kwargs.pop('beta', None)
            self.p = kwargs.pop('p', None)
            self.sigma = kwargs.pop('sigma', None)
            self.time_steps = kwargs.pop('time_steps', None)
            self.dt = kwargs.pop('dt', None)
        
        # Initialization of effective contact rate if sigma and p present
        if self.p and self.sigma is not None: 
            self.gamma = self.p * self.sigma 

    
    def f(self, y, t):
        """ Calculates the partial derivatives
        Args:
            y: number of subjects across all compartments 
            t: numpy.linspace of the simulation horizon
        Returns: 
            Partial derivatives values
        """
        S, I, R = y
        dSdt = -(self.beta * I * S)/self.N 
        dIdt =  (self.beta * I * S)/self.N  - self.gamma * I 
        dRdt =  self.gamma * I 
        return dSdt, dIdt, dRdt


    def simulate_epidemic(self):
        """ Simulates an epidemic in time
        returns:
            Array containing the value of each y for each desired time in t, with the inition value y0 in the first row
        """
        S0 = 10000
        I0 = 100
        R0 = 0

        y0 = [S0, I0, R0]
        y = odeint(self.f, y0, self.t)
        return y

    def plot_simulation(self, y):
        """ Plots the epidemic development in time
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


if __name__ == '__main__':
    kwargs = {"beta":0.5, "gamma":0.8, "p":0.4, "sigma":2, "time_steps":100, "dt":1, "N":100}
    model = SIR(**kwargs)

    y = model.simulate_epidemic()
    model.plot_simulation(y)

    