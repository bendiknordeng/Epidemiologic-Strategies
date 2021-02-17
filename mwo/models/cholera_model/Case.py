__author__ = "Martin Willoch Olstad"
__email__ = "martinwilloch@gmail.com"

import json
import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.integrate import odeint
import geopy.distance


class Case:

    def __init__(self, regions=[], time_steps=10.0, dt=1.0, mu=0*1/(61.0*365), beta=1.0, kappa=1.0e6, gamma=1/5.0,
                 alpha=1/10.0, p=10.0, l=1.0, muB=0.03, params=None, horizon=4, aggregation_constant=1, regional=False):
        self.regions = regions
        self.t = np.linspace(0, time_steps, int(time_steps / dt))
        self.time_steps = time_steps
        self.dt = dt
        self.number_of_regions = len(self.regions)
        self.distance_matrix = self.calculate_distance_matrix(self.regions)
        self.mean_dispersal_distance = 9.0
        self.probability_matrix = self.initialize_probability_matrix()
        self.update_probability_matrix(dt=0)

        self.horizon = horizon
        self.aggregation_constant = aggregation_constant

        self.mu = mu
        self.beta = beta
        self.kappa = kappa
        self.gamma = gamma
        self.alpha = alpha
        self.p = p
        self.l = l
        self.muB = muB

        self.regional = regional

        if params is not None:
            self.tau = params['tau']
            self.kappa = params['kappa']
            self.p = params['p']
            self.gamma = params['gamma']
            self.rhoA = params['rhoA']
            self.rhoI = params['rhoI']
            self.chi = params['chi']
            self.lam = params['lam']
            self.psi = params['psi']
            self.mu = params['mu']
            self.muB = params['muB']
            self.muCbar = params['muCbar']
            self.alpha_muC = params['alpha_muC']
            self.sigma_muC = params['sigma_muC']
            self.lbar = params['lbar']
            self.alpha_l = params['alpha_l']
            self.sigma_l = params['sigma_l']
            self.nu = [float(self.regions[i].nu) for i in range(self.number_of_regions)]
            self.beta = [float(self.regions[i].beta) for i in range(self.number_of_regions)]
            self.phi = [float(self.regions[i].phi) for i in range(self.number_of_regions)]
            self.theta = [float(self.regions[i].theta) for i in range(self.number_of_regions)]
            self.W = [float(self.regions[i].W) for i in range(self.number_of_regions)]

            self.mean_dispersal_distance = params['D']

            self.initial_prob_matrix = self.new_calculate_probability_matrix()
            self.dispersal = None

            self.rates = self.tau, self.kappa, self.p, self.gamma, self.rhoA, self.rhoI, self.chi, self.lam,\
                         self.psi, self.mu, self.muB, self.muCbar, self.alpha_muC, self.sigma_muC, self.lbar,\
                         self.alpha_l, self.sigma_l, self.nu, self.beta, self.phi, self.theta, self.W, self.distance_matrix, self.mean_dispersal_distance

            self.new_rates = self.tau, self.kappa, self.p, self.gamma, self.rhoA, self.rhoI, self.chi, self.lam, \
                         self.psi, self.mu, self.muB, self.muCbar, self.alpha_muC, self.sigma_muC, self.lbar, \
                         self.alpha_l, self.sigma_l, self.nu, self.beta, self.phi, self.theta, self.W, self.distance_matrix, self.mean_dispersal_distance, self.initial_prob_matrix

            self.rates_extended = self.tau, self.kappa, self.p, self.gamma, self.rhoA, self.rhoI, self.chi, self.lam, \
                         self.psi, self.mu, self.muB, self.muCbar, self.alpha_muC, self.sigma_muC, self.lbar, \
                         self.alpha_l, self.sigma_l, self.nu, self.beta, self.phi, self.theta, self.W, self.distance_matrix, self.mean_dispersal_distance, self.dispersal

    def update_params(self):
        self.nu = [float(self.regions[i].nu) for i in range(self.number_of_regions)]
        self.beta = [float(self.regions[i].beta) for i in range(self.number_of_regions)]
        self.phi = [float(self.regions[i].phi) for i in range(self.number_of_regions)]
        self.theta = [float(self.regions[i].theta) for i in range(self.number_of_regions)]
        self.W = [float(self.regions[i].W) for i in range(self.number_of_regions)]

        self.rates = self.tau, self.kappa, self.p, self.gamma, self.rhoA, self.rhoI, self.chi, self.lam, \
                     self.psi, self.mu, self.muB, self.muCbar, self.alpha_muC, self.sigma_muC, self.lbar, \
                     self.alpha_l, self.sigma_l, self.nu, self.beta, self.phi, self.theta, self.W, self.distance_matrix, self.mean_dispersal_distance

        self.rates_extended = self.tau, self.kappa, self.p, self.gamma, self.rhoA, self.rhoI, self.chi, self.lam, \
                              self.psi, self.mu, self.muB, self.muCbar, self.alpha_muC, self.sigma_muC, self.lbar, \
                              self.alpha_l, self.sigma_l, self.nu, self.beta, self.phi, self.theta, self.W, self.distance_matrix, self.mean_dispersal_distance, self.dispersal

        self.new_rates = self.tau, self.kappa, self.p, self.gamma, self.rhoA, self.rhoI, self.chi, self.lam, \
                         self.psi, self.mu, self.muB, self.muCbar, self.alpha_muC, self.sigma_muC, self.lbar, \
                         self.alpha_l, self.sigma_l, self.nu, self.beta, self.phi, self.theta, self.W, self.distance_matrix, self.mean_dispersal_distance, self.initial_prob_matrix

    def calculate_distance_matrix(self):
        distance_matrix = np.zeros((self.number_of_regions, self.number_of_regions))
        for i in range(len(self.regions)-1):
            for j in range(i+1, len(self.regions)):
                distance_matrix[i,j] = self.calculate_distance(self.regions[i],self.regions[j])
                distance_matrix[j,i] = distance_matrix[i,j]
        return distance_matrix

    @staticmethod
    def calculate_distance_matrix(regions):
        distance_matrix = np.zeros((len(regions), len(regions)))
        for i in range(len(regions)-1):
            for j in range(i+1, len(regions)):
                distance_matrix[i,j] = Case.calculate_distance(regions[i],regions[j])
                distance_matrix[j,i] = distance_matrix[i,j]
        return distance_matrix

    def new_calculate_probability_matrix(self):
        prob_matrix = np.zeros((self.number_of_regions,self.number_of_regions))
        counter = 0
        for i in range(self.number_of_regions):
            softmax_denominator = 0
            for k in range(self.number_of_regions):
                if i != k:
                    term1 = self.regions[k].S[0] + self.regions[k].A[0] + self.regions[k].I[0] + self.regions[k].R[0]
                    term2 = np.exp(-self.distance_matrix[i, k] / self.mean_dispersal_distance)
                    softmax_denominator += term1 * term2
                    counter += 1
            for j in range(self.number_of_regions):
                if i != j:
                    prob_matrix[i, j] = (self.regions[j].S[0] + self.regions[j].A[0] + self.regions[j].I[0] + self.regions[j].R[0]) * np.exp(-self.distance_matrix[i, j] / self.mean_dispersal_distance) / softmax_denominator
        return prob_matrix

    def initialize_probability_matrix(self):
        return np.zeros((self.t.shape[0], self.number_of_regions, self.number_of_regions))

    def calculate_probability_matrix(self):
        probability_matrix = np.zeros((self.t.shape[0], self.number_of_regions, self.number_of_regions))
        for dt in range(self.t.shape[0]):
            for i in range(len(self.regions)):
                softmax_denominator = 0
                for k in range(len(self.regions)):
                    if self.regions[i] != self.regions[k]:
                        softmax_denominator += self.regions[k].N[dt]*np.exp(-self.distance_matrix[i,k]/self.mean_dispersal_distance)
                for j in range(len(self.regions)):
                    if self.regions[i] != self.regions[j]:
                        probability_matrix[dt,i,j] = self.regions[j].N[dt]*np.exp(-self.distance_matrix[i,j]/
                                                                                self.mean_dispersal_distance)/softmax_denominator
        return probability_matrix

    def update_probability_matrix(self, dt):
        counter = 0
        for i in range(self.number_of_regions):
            softmax_denominator = 0
            if dt == 0: dt += 1
            for k in range(self.number_of_regions):
                assert self.regions[k].N[dt - 1] > 0.0
                if i != k:
                    term1 = self.regions[k].N[dt-1]
                    term2 = np.exp(-self.distance_matrix[i,k]/self.mean_dispersal_distance)
                    if term1 <= 0:
                        print('dt: ',dt)
                        print('N (full): ', self.regions[k].N)
                        print()
                    softmax_denominator += term1 * term2
                    counter += 1
            for j in range(self.number_of_regions):
                if i != j:
                    assert self.mean_dispersal_distance > 0.0
                    assert softmax_denominator > 0.0
                    self.probability_matrix[dt, i, j] = self.regions[j].N[dt-1] \
                                                        * np.exp(-self.distance_matrix[i, j] /
                                                                 self.mean_dispersal_distance) \
                                                        / softmax_denominator

    def simulate_epidemic(self, use_numba=False, begin=0, end=1, old=False):
        comps = 6
        Y0 = np.zeros(comps*self.number_of_regions)
        begin_ix = int(begin/self.dt)

        end_ix = int(1+(end/self.dt))
        tspan = np.linspace(begin, end, num=int(1+(end-begin)/self.dt))

        for i in range(self.number_of_regions):
            Y0[0 + i * comps] = float(self.regions[i].S[begin_ix])
            Y0[1 + i * comps] = float(self.regions[i].A[begin_ix])
            Y0[2 + i * comps] = float(self.regions[i].I[begin_ix])
            Y0[3 + i * comps] = float(self.regions[i].R[begin_ix])
            Y0[4 + i * comps] = float(self.regions[i].B[begin_ix])
            Y0[5 + i * comps] = float(self.regions[i].M[begin_ix])

        self.update_params()

        if self.regional:
            Y = odeint(self.g, Y0, tspan, args=(self.rates_extended,))
        elif old:
            Y = odeint(self.old_h, Y0, tspan, args=(self.rates,))
        else:
            func = numba.jit(self.h) if use_numba else self.h
            #Y = odeint(func, Y0, tspan, args=(self.rates,))
            Y = odeint(func, Y0, tspan, args=(self.new_rates,))

        for i in range(self.number_of_regions):
            self.regions[i].X = Y[:, i*comps:(i+1)*comps]
            self.regions[i].S[begin_ix:end_ix] = self.regions[i].X[:,0]
            self.regions[i].A[begin_ix:end_ix] = self.regions[i].X[:,1]
            self.regions[i].I[begin_ix:end_ix] = self.regions[i].X[:,2]
            self.regions[i].R[begin_ix:end_ix] = self.regions[i].X[:,3]
            self.regions[i].B[begin_ix:end_ix] = self.regions[i].X[:,4]
            self.regions[i].M[begin_ix:end_ix] = self.regions[i].X[:,5]

    def transition(self, decisions, information, actual_transition=True, dispersal=None, time=1.0, decision_period=1.0, old=False):
        time_ix = int(time/self.dt)
        end_ix = int((time+decision_period)/self.dt)

        initial_parameters = self.l, self.lbar, self.dispersal
        initial_decision_parameters = [(self.regions[i].nu, self.regions[i].beta, self.regions[i].phi, self.regions[i].theta) for i in range(self.number_of_regions)]

        self.l = information
        self.lbar = information
        if dispersal is not None:
            self.dispersal = dispersal[int(time)]
        if decisions is None:
            for i in range(self.number_of_regions):
                self.regions[i].nu = 0
                self.regions[i].update_param_list(time=time+decision_period)
        else:
            for i in range(self.number_of_regions):
                # Vaccine update
                less_than_vaccine_demand = self.aggregation_constant * decisions[i,0] <= self.regions[i].S[time_ix]
                if less_than_vaccine_demand:
                    self.regions[i].nu = self.aggregation_constant*decisions[i,0]
                else:
                    self.regions[i].nu = self.regions[i].S[time_ix]

                # Disinfectant update
                less_than_disinfectant_demand = self.aggregation_constant*decisions[i,1] <= self.regions[i].S[time_ix]
                non_zero_disinfectant_demand = self.regions[i].S[time_ix] > 0
                if not non_zero_disinfectant_demand:
                    proportion_drinking_disinfected_water = 0.0
                elif less_than_disinfectant_demand and non_zero_disinfectant_demand:
                    proportion_drinking_disinfected_water = self.aggregation_constant*decisions[i,1]/self.regions[i].S[time_ix]
                else:
                    proportion_drinking_disinfected_water = 1.0
                self.regions[i].beta = 1 - proportion_drinking_disinfected_water

                # Rehydration update
                less_than_rehydration_demand = self.aggregation_constant*decisions[i,2] <= self.regions[i].I[time_ix]
                non_zero_rehydration_demand = self.regions[i].I[time_ix] > 0
                if not non_zero_rehydration_demand:
                    proportion_receiving_rehydration = 0.0
                elif less_than_rehydration_demand and non_zero_rehydration_demand:
                    proportion_receiving_rehydration = self.aggregation_constant*decisions[i,2]/self.regions[i].I[time_ix]
                else:
                    proportion_receiving_rehydration = 1.0
                self.regions[i].phi = proportion_receiving_rehydration


                # Antibiotics update
                less_than_antibiotics_demand = self.aggregation_constant*decisions[i,3] <= self.regions[i].I[time_ix]
                non_zero_antibiotics_demand = self.regions[i].I[time_ix] > 0
                if not non_zero_antibiotics_demand:
                    proportion_receiving_antibiotics = 0.0
                elif less_than_antibiotics_demand and non_zero_antibiotics_demand:
                    proportion_receiving_antibiotics = self.aggregation_constant*decisions[i,3]/self.regions[i].I[time_ix]
                else:
                    proportion_receiving_antibiotics = 1.0
                self.regions[i].theta = proportion_receiving_antibiotics

                # Actual update
                self.regions[i].update_param_list(time=time+decision_period)

        self.simulate_epidemic(use_numba=False, begin=int(time), end=int(time+decision_period), old=old)

        if not actual_transition:
            self.l, self.lbar, self.dispersal = initial_parameters
            for i in range(self.number_of_regions):
                self.regions[i].nu, self.regions[i].beta, self.regions[i].phi, self.regions[i].theta = initial_decision_parameters[i]
                self.regions[i].update_param_list(time=time+decision_period)

        transition_results = np.array([[self.regions[i].S[end_ix], self.regions[i].A[end_ix], self.regions[i].I[end_ix],
                                        self.regions[i].R[end_ix], self.regions[i].B[end_ix],
                                        self.regions[i].M[end_ix]] for i in range(self.number_of_regions)])

        return transition_results

    def get_cumulative_infected(self, begin=0, end=1):
        comps = 7
        Y0 = np.zeros(comps * self.number_of_regions)
        begin_ix = int(begin / self.dt)
        end_ix = int(1 + (end / self.dt))
        tspan = np.linspace(0, self.horizon, num=int(1 + (end - begin) / self.dt))
        for i in range(self.number_of_regions):
            Y0[0 + i * comps] = float(self.regions[i].S[begin_ix])
            Y0[1 + i * comps] = float(self.regions[i].A[begin_ix])
            Y0[2 + i * comps] = float(self.regions[i].I[begin_ix])
            Y0[3 + i * comps] = float(self.regions[i].R[begin_ix])
            Y0[4 + i * comps] = float(self.regions[i].B[begin_ix])
            Y0[5 + i * comps] = float(self.regions[i].M[begin_ix])

        self.update_params()

        #Y = odeint(self.cumulative_h, Y0, tspan, args=(self.rates,))
        Y = odeint(self.cumulative_h, Y0, tspan, args=(self.new_rates,))

        cumulative_cases = np.zeros((Y.shape[0], self.number_of_regions))
        for i in range(self.number_of_regions):
            self.regions[i].X = Y[:, i * comps:(i + 1) * comps]
            self.regions[i].S[begin_ix:end_ix] = self.regions[i].X[:, 0]
            self.regions[i].A[begin_ix:end_ix] = self.regions[i].X[:, 1]
            self.regions[i].I[begin_ix:end_ix] = self.regions[i].X[:, 2]
            self.regions[i].R[begin_ix:end_ix] = self.regions[i].X[:, 3]
            self.regions[i].B[begin_ix:end_ix] = self.regions[i].X[:, 4]
            self.regions[i].M[begin_ix:end_ix] = self.regions[i].X[:, 5]
            cumulative_cases[:, i] = self.regions[i].X[:, 6] + self.regions[i].I[begin_ix]
        return cumulative_cases

    def plot(self):
        t,i,j = self.probability_matrix.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(i,j,t, zdir='t', c='red')
        plt.show()

    def plot_all(self):
        for i in range(self.number_of_regions):
            self.regions[i].plot()

    def convert_to_dataframe(self):
        df = pd.DataFrame({
            'Governorate':[],
            'Time':[],
            'lon':[],
            'lat':[],
            'id':[],
            'Susceptible':[],
            'Asymptomatic': [],
            'Infected':[],
            'Recovered':[],
            'Concentration':[],
            'Fatalities':[]
        })
        for area in self.regions:
            for t in range(0, int(1+self.horizon/self.dt), int(1/self.dt)):
                if True:
                    df_row = pd.DataFrame({
                        'Governorate': [area.name],
                        'Time': [t * self.dt],
                        'lon': [area.longitude],
                        'lat': [area.latitude],
                        'id': [area.id],
                        'Susceptible': [area.S[t]],
                        'Asymptomatic': [area.A[t]],
                        'Infected': [area.I[t]],
                        'Recovered': [area.R[t]],
                        'Concentration': [area.B[t]],
                        'Fatalities': [area.M[t]],
                        'Mortality': [area.M[t]/(area.S[0]+area.A[0]+area.I[0]+area.R[0])],
                        'Cumulative symptomatic infections': [area.C[t]],
                        'Symptomatic case (floor)': [(np.floor(area.C[t]) >= 1)],
                        'Symptomatic case (round)': [(np.round(area.C[t]) >= 1)],
                        'At least one symptomatic cholera case': [(area.C[t] >= 1)],
                        'Relative cumulative symptomatic infections': [area.C[t]/(area.S[t]+area.A[t]+area.I[t]+area.R[t])]

                    })
                    df = df.append(df_row, ignore_index=True)
        return df

    def map_plot(self, z='Fatalities', case='haiti'):
        df = self.convert_to_dataframe()
        c = px.colors.sequential.Reds
        if case == 'yemen':
            file_path = '../data/yemen_governorates.json'
            lat = 15.7
            lon = 46.0
            zoom = 5.4
        elif case == 'haiti':
            file_path = '../data/haiti_departments.json'
            lat = 19.0
            lon = -72.7
            zoom = 7.3
        with open(file_path, 'r') as read_file:
            regions = json.load(read_file)
        ms = 'white-bg'
        fig = px.choropleth_mapbox(data_frame=df, geojson=regions, locations='id',
                                   color=z, mapbox_style=ms, opacity=0.5,
                                   zoom=zoom, center={'lat': lat, 'lon': lon},
                                   animation_frame='Time', hover_name='Governorate',
                                   range_color=[df[z].min(), df[z].max()],
                                   color_continuous_scale=c)
        fig.show()


    @staticmethod
    def calculate_distance(area1, area2):
        coords1 = (area1.latitude, area1.longitude)
        coords2 = (area2.latitude, area2.longitude)
        dist = geopy.distance.vincenty(coords1, coords2).km
        return dist

    @staticmethod
    def h(Y, t, rates):
        dY = np.zeros(shape=Y.shape)
        comps = 6
        regions = int(len(dY) / comps)

        tau, kappa, p, gamma, rhoA, rhoI, chi, lam, psi, mu, muB, muC, alpha_muC, sigma_muC, l, alpha_l, sigma_l, nu, beta, phi, theta, W, dist_matrix, mean_dist, prob_matrix = rates

        water_concentration = np.array([Y[4 + comps * j] * W[j] for j in range(regions)])
        dispersal = np.dot(water_concentration, prob_matrix)

        for i in range(regions):
            S = Y[0 + comps * i]
            A = Y[1 + comps * i]
            I = Y[2 + comps * i]
            R = Y[3 + comps * i]
            B = Y[4 + comps * i]

            N = S + A + I + R
            nu_i = nu[i]
            beta_i = beta[i]
            phi_i = phi[i]
            theta_i = theta[i]
            W_i = W[i]

            dY[0 + comps * i] = mu * (N - S) - tau * nu_i - beta_i * (B / (kappa + B)) * S
            dY[1 + comps * i] = p * beta_i * (B / (kappa + B)) * S - gamma * A - mu * A
            dY[2 + comps * i] = (1 - p) * beta_i * (B / (kappa + B)) * S - muC * (
                        phi_i + (1 - phi_i) * chi) * I - gamma * ((1 - theta_i) + theta_i * lam) * I - mu * I
            dY[3 + comps * i] = tau * nu_i + gamma * (A + ((1 - theta_i) + theta_i * lam) * I) - mu * R
            dY[4 + comps * i] = (rhoA / W_i) * A + (rhoI / W_i) * (psi * theta_i + (1 - theta_i)) * I - muB * B - l * (
                        B - (1 / W_i) * dispersal[i])
            dY[5 + comps * i] = muC * (phi_i + (1 - phi_i) * chi) * I
        return dY

    @staticmethod
    def cumulative_h(Y, t, rates):
        dY = np.zeros(shape=Y.shape)
        comps = 7
        regions = int(len(dY) / comps)

        #tau, kappa, p, gamma, rhoA, rhoI, chi, lam, psi, mu, muB, muC, alpha_muC, sigma_muC, l, alpha_l, sigma_l, nu, beta, phi, theta, W, dist_matrix, mean_dist = rates
        tau, kappa, p, gamma, rhoA, rhoI, chi, lam, psi, mu, muB, muC, alpha_muC, sigma_muC, l, alpha_l, sigma_l, nu, beta, phi, theta, W, dist_matrix, mean_dist, prob_matrix = rates

        water_concentration = np.array([Y[4 + comps * j] * W[j] for j in range(regions)])
        dispersal = np.dot(water_concentration, prob_matrix)

        for i in range(regions):
            S = Y[0 + comps * i]
            A = Y[1 + comps * i]
            I = Y[2 + comps * i]
            R = Y[3 + comps * i]
            B = Y[4 + comps * i]

            N = S + A + I + R
            nu_i = nu[i]
            beta_i = beta[i]
            phi_i = phi[i]
            theta_i = theta[i]
            W_i = W[i]

            dY[0 + comps * i] = mu * (N - S) - tau * nu_i - beta_i * (B / (kappa + B)) * S
            dY[1 + comps * i] = p * beta_i * (B / (kappa + B)) * S - gamma * A - mu * A
            dY[2 + comps * i] = (1 - p) * beta_i * (B / (kappa + B)) * S - muC * (
                        phi_i + (1 - phi_i) * chi) * I - gamma * ((1 - theta_i) + theta_i * lam) * I - mu * I
            dY[3 + comps * i] = tau * nu_i + gamma * (A + ((1 - theta_i) + theta_i * lam) * I) - mu * R
            dY[4 + comps * i] = (rhoA / W_i) * A + (rhoI / W_i) * (psi * theta_i + (1 - theta_i)) * I - muB * B - l * (
                        B - (1 / W_i) * dispersal[i])
            dY[5 + comps * i] = muC * (phi_i + (1 - phi_i) * chi) * I
            dY[6 + comps * i] = (1 - p) * beta_i * (B / (kappa + B)) * S

        return dY
