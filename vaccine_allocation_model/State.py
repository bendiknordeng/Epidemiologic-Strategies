import numpy as np
from datetime import timedelta
from collections import defaultdict
from functools import partial
from utils import tcolors
from copy import copy

class State:
    def __init__(self, S, E1, E2, A, I, R, D, V, contact_weights, flow_scale, vaccines_available, new_infected, 
                total_infected, new_deaths, trend, trend_count, R_t, date, time_step=0):
        """State object for the Markov Decision Process. Keeps track of relevant information for running simulations and making decisions.

        Args:
            S (numpy.ndarray): number of suceptible in each region for each age group (#regions, #age_groups)
            E1 (numpy.ndarray): number of latent exposed in each region for each age group (#regions, #age_groups)
            E2 (numpy.ndarray): number of infectious exposed in each region for each age group (#regions, #age_groups)
            A (numpy.ndarray): number of asymptomatic infected in each region for each age group (#regions, #age_groups)
            I (numpy.ndarray): number of symptomatic infected in each region for each age group (#regions, #age_groups)
            R (numpy.ndarray): number of recovered in each region for each age group (#regions, #age_groups)
            D (numpy.ndarray): number of accumulated deaths in each region for each age group (#regions, #age_groups)
            V (numpy.ndarray): number of vaccinated in each region for each age group (#regions, #age_groups)
            contact_weights (list): contact weight for home, school, work and public
            flow_scale (float): scale for total commuting during a decision period
            vaccines_available (int): number of initial vaccines available
            new_infected (numpy.ndarray): number of new infected in each region for each age group (#regions, #age_groups)
            total_infected (numpy.ndarray): number of cumulative effected in each region for each age group (#regions, #age_groups)
            new_deaths (numpy.ndarray): number of new deaths in each region for each age group (#regions, #age_groups)
            wave_count (dict): count of wave states at this time_step
            strategy_count (dict): count of wave states when vaccines have been available (vaccines_available > 0)
            date (datetime.date): current date in the simulation
            time_step (int, optional): current time step in the simulation. Defaults to 0.
        """
        self.S = S
        self.E1 = E1
        self.E2 = E2
        self.A = A
        self.I = I
        self.R = R
        self.D = D
        self.V = V
        
        self.contact_weights = np.array(contact_weights)
        self.flow_scale = flow_scale
        self.vaccines_available = vaccines_available
        self.new_infected = new_infected
        self.total_infected = total_infected
        self.new_deaths = new_deaths
        self.trend = trend
        self.trend_count = trend_count
        self.R_t = R_t
        self.date = date
        self.time_step = time_step

    def get_transition(self, decision, information, epidemic_function, decision_period):
        """Transition fucntion for the current state in the process

        Args:
            decision (list): indicating the number of vaccines to be allocated to each region and each age group (#regions, #age_groups)
            information (dict): exogeneous information with keys ['vaccine_supply', 'wave_factor', 'contact_weights', 'flow_scale']
            epidemic_function (function): executable simulating the current step of the epidemic
            decision_period (int): number of timesteps before next decision

        Returns:
            State: the next state with new information and compartments
        """
        # Update information
        contact_weights = information['contact_weights']
        flow_scale = information['flow_scale']
        
        # Update compartment values
        S, E1, E2, A, I, R, D, V, new_infected, new_deaths, trend, R_t = epidemic_function(self, decision, decision_period, information)
        
        if trend != self.trend and self.vaccines_available > 0:
            self.trend = trend
            self.trend_count[trend] += 1

        # Update vaccine available
        vaccines_left = self.vaccines_available - np.sum(decision)
        new_vaccines = np.sum(information['vaccine_supply'])
        vaccines_available = vaccines_left + new_vaccines

        # Update time step
        time_step = self.time_step + decision_period
        date = self.date + timedelta(decision_period//4)

        return State(S, E1, E2, A, I, R, D, V, contact_weights, flow_scale, vaccines_available, new_infected, 
                    self.total_infected+new_infected, new_deaths, trend, self.trend_count, R_t, date, time_step)
    
    def get_compartments_values(self):
        """Retrieves compartments

        Returns:
            list: each compartment of the state
        """
        return [self.S, self.E1, self.E2, self.A, self.I, self.R, self.D, self.V]

    def __str__(self):
        total_pop = np.sum(self.get_compartments_values()[:-1])
        info = ["Susceptibles", "Exposed (latent)",
                "Exposed (presymptomatic)", "Asymptomatic infected",
                "Infected", "Recovered", "Dead", "Vaccinated", 
                "New infected", "Total infected"]
        values = [np.sum(compartment) for compartment in self.get_compartments_values()]
        values.append(np.sum(self.new_infected))
        values.append(np.sum(self.total_infected))
        percent = 100 * np.array(values)/total_pop
        status = f"Date: {self.date} (week {self.date.isocalendar()[1]})\n"
        status += f"Timestep: {self.time_step} (day {self.time_step//4})\n"
        status += f"Trend: {self.trend}\n"
        for i in range(len(info)):
            status += f"{info[i]:<25} {values[i]:>7.0f} ({percent[i]:>5.2f}%)\n"
        return status
    
    def __repr__(self):
        total_pop = np.sum(self.get_compartments_values()[:-1])
        info = ["Susceptibles", "Exposed (latent)",
                "Exposed (presymptomatic)", "Asymptomatic infected",
                "Infected", "Recovered", "Dead", "Vaccinated", 
                "New infected", "Total infected"]
        values = [np.sum(compartment) for compartment in self.get_compartments_values()]
        values.append(np.sum(self.new_infected))
        values.append(np.sum(self.total_infected))
        percent = 100 * np.array(values)/total_pop
        status = f"{tcolors.BOLD}MDP State object{tcolors.ENDC}\n"
        status += f"Date: {self.date} (week {self.date.isocalendar()[1]})\n"
        status += f"Timestep: {self.time_step} (day {self.time_step//4})\n"
        status += f"Trend: {self.trend}\n"
        for i in range(len(info)):
            status += f"{info[i]:<25} {values[i]:>7.0f} ({percent[i]:>5.2f}%)\n"
        return status

    @staticmethod
    def generate_initial_state(num_initial_infected, contact_weights, flow_scale, population, start_date, time_step=0):
        """Generate initial state for the Markov Decision Process

        Args:
            num_initial_infected (int): number of infected to be distributed randomly across regions
            contact_weights (list): contact weight for home, school, work and public
            flow_scale (float): scale for total commuting during a decision period
            population (pandas.DataFrame): information about population in reions and age groups
            start_date (datetime.date): starting date for simulation
            time_step (int, optional): starting time step for simulation. Defaults to 0.

        Returns:
            State: initial state object for the simulation
        """
        pop = population[population.columns[2:-1]].values
        S  = pop.copy()
        E1 = np.zeros(pop.shape)
        E2 = np.zeros(pop.shape)
        A  = np.zeros(pop.shape)
        I  = np.zeros(pop.shape)
        R  = np.zeros(pop.shape)
        D  = np.zeros(pop.shape)
        V  = np.zeros(pop.shape)

        while num_initial_infected > 0:
            region = np.random.randint(0, pop.shape[0])
            age_group = np.random.randint(0, pop.shape[1])
            if S[region][age_group] > 0:
                num_initial_infected -= 1
                S[region][age_group] -= 1
                I[region][age_group] += 1 

        return State(S, E1, E2, A, I, R, D, V, contact_weights, flow_scale, 0,
                    I.copy(), I.copy(), np.zeros(pop.shape), None, {"U": 0, "D": 0, "N": 0}, None, start_date, time_step)