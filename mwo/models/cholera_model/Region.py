__author__ = "Martin Willoch Olstad"
__email__ = "martinwilloch@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Region:

    def __init__(self, A0=0, I0=0, R0=0, B0=0, N0=0, M0=0, mu=0, beta=0, kappa=0, gamma=0, alpha=0, p=0, W=0, muB=0, l=0,
                 longitude=0, latitude=0, altitude=0, time_steps=100, dt=1.0, name='NA', id='NA', nu0=0.0, beta0=1.0,
                 phi0=1.0, theta0=0.0, horizon=4):
        # Population
        self.A0 = A0  # Initial asymptomatically infected
        self.I0 = I0  # Initial symptomatically infected
        self.R0 = R0  # Initial recovered
        self.B0 = B0  # Initial cholera concentration in water source
        self.N0 = N0  # Initial population
        self.S0 = N0-A0-I0-R0  # Initial susceptibles
        self.M0 = M0  # Initial fatalities
        self.t = np.linspace(0, time_steps, int(time_steps/dt))
        X0 = np.array([self.S0, self.A0, self.I0, self.R0, self.B0]).T
        self.X = np.zeros(shape=(self.t.shape[0],len(X0)))
        self.X[0] = X0
        self.N = np.sum(self.X[:,:3], axis=1)
        self.dt = dt
        #print('N: ',self.N)

        self.horizon = horizon

        # Rates
        self.mu = mu  # Rate of birth and natural death
        self.beta = beta  # Rate of exposure to contaminated water
        self.kappa = kappa  # Half-saturation constant
        self.gamma = gamma  # Rate of recovery
        self.alpha = alpha  # Rate of cholera-induced death
        self.p = p  # Rate of excreted cholera bacteria per infected individual
        self.W = W  # Volume of contaminated water source
        self.muB = muB  # Rate of cholera bacteria death
        self.l = l  # Rate of cholera bacteria dispersal
        self.rates = (self.mu, self.beta, self.kappa, self.gamma, self.alpha, self.p, self.W, self.muB, self.l)

        self.nu = nu0
        self.beta = beta0
        self.phi = phi0
        self.theta = theta0
        self.W = 15*self.N0*365*1000

        self.nus = np.zeros(int(self.horizon+1))
        self.betas = np.zeros(int(self.horizon+1))
        self.phis = np.zeros(int(self.horizon+1))
        self.thetas = np.zeros(int(self.horizon+1))

        self.nus[0] = nu0
        self.betas[0] = beta0
        self.phis[0] = phi0
        self.thetas[0] = theta0

        # Geographic information
        self.name = name
        self.id = id
        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude

        horizon_ix = int(1+(self.horizon/self.dt))

        self.S = np.zeros(horizon_ix)
        self.A = np.zeros(horizon_ix)
        self.I = np.zeros(horizon_ix)
        self.R = np.zeros(horizon_ix)
        self.B = np.zeros(horizon_ix)
        self.M = np.zeros(horizon_ix)

        self.S[0] = self.S0
        self.A[0] = self.A0
        self.I[0] = self.I0
        self.R[0] = self.R0
        self.B[0] = self.B0
        self.M[0] = self.M0

        self.C = np.zeros(horizon_ix)

    def solve_ode(self):
        Y0 = self.S0, self.I0, self.R0, self.B0
        Y = odeint(self.construct_ode, Y0, self.t, args=(self.N0, self.mu, self.beta, self.kappa, self.gamma,
                                                         self.alpha, self.p, self.W, self.muB)).T
        return Y

    def update_param_list(self, time):
        self.nus[int(time)] = self.nu
        self.betas[int(time)] = self.beta
        self.phis[int(time)] = self.phi
        self.thetas[int(time)] = self.theta

    def plot_decision_params(self, show=True, decisions=('nu','beta','phi','theta')):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        time_span = len(self.nus)
        self.t = np.linspace(0, time_span, num=time_span)

        if self.name is not 'NA':
            ax1.set_title(self.name.replace('_',' '))

        ax1.set_xlabel('Time [days]')
        ax1.set_ylabel('Rate [individuals per day]')
        ax2.set_ylabel('Individuals')

        ax1.set_ylim(bottom=0.0)
        ax2.set_ylim(bottom=0.0, top=max(self.nus))

        plots = ax1.plot()

        if 'nu' in decisions: plots += ax2.plot(self.t, self.nus, label='Vaccinated', c='green')
        if 'beta' in decisions: plots += ax1.plot(self.t, self.betas, label='Disinfectant', c='red')
        if 'phi' in decisions: plots += ax1.plot(self.t, self.phis, label='Rehydration', c='blue')
        if 'theta' in decisions: plots += ax1.plot(self.t, self.thetas, label='Antibiotics', c='orange')

        #plots = vaccination_plot + disinfectant_plot + rehydration_plot + antiobiotics_plot
        labels = [plot.get_label() for plot in plots]
        ax1.legend(plots, labels, loc=0)
        if show:
            plt.show()
            print('VACCINATED: ',self.nus)
            print('DISINFECTANT: ',self.betas)
            print('REHYDRATION: ',self.phis)
            print('ANTIBIOTICS: ',self.thetas)

    def plot(self, show=True, susceptible=False):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        time_span = len(self.S)*self.dt
        self.t = np.linspace(0, time_span, num=len(self.S))

        if susceptible:
            susceptible_plot = ax1.plot(self.t, self.S, label='Susceptible', c='blue')
        asymptomatic_plot = ax1.plot(self.t, self.A, label='Asymptomatic', c='purple')
        infected_plot = ax1.plot(self.t, self.I, label='Symptomatic', c='red')
        #recovered_plot = ax1.plot(self.t, self.R, label='Recovered', c='green')
        concentration_plot = ax2.plot(self.t, self.B, label='Concentration', c='orange')
        fatality_plot = ax1.plot(self.t, self.M, label='Fatalities', c='black')

        if self.name is not 'NA':
            ax1.set_title(self.name.replace('_',' '))

        ax1.set_xlabel('Time [days]')
        ax1.set_ylabel('Individuals')
        ax2.set_ylabel('Vibrio cholera concentration [cells/ml]')

        ax1.set_ylim(bottom=0.0)
        ax2.set_ylim(bottom=0.0)

        if susceptible:
            plots = susceptible_plot + asymptomatic_plot + infected_plot + recovered_plot + concentration_plot + fatality_plot
        else:
            #plots = asymptomatic_plot + infected_plot + recovered_plot + concentration_plot + fatality_plot
            plots = asymptomatic_plot + infected_plot + concentration_plot + fatality_plot
        labels = [plot.get_label() for plot in plots]
        ax1.legend(plots, labels, loc=0)
        if show:
            plt.show()
