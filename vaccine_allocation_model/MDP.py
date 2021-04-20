from covid.utils import generate_weekly_data
from vaccine_allocation_model.State import State
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import timedelta

class MarkovDecisionProcess:
    def __init__(self, population, epidemic_function, initial_state, horizon, decision_period, policy, historic_data=None):
        """ Initializes an instance of the class MarkovDecisionProcess, that administrates

        Parameters
            OD_matrices: Origin-Destination matrices giving movement patterns between regions
            population: A DataFrame with region_id, region_name and population
            epidemic_function: An epidemic model that enables simulation of the decision process
            vaccine_supply: Information about supply of vaccines, shape e.g. (#decision_period, #regions)
            horizon: The amount of decision_periods the decision process is run 
            decision_period: The number of time steps that every decision directly affects
            policy: How the available vaccines should be distributed.
            historic_data: dataframe, or None indicating whether or not to use fhi_data in simulation
        """
        self.horizon = horizon
        self.population = population
        self.epidemic_function = epidemic_function
        self.state = initial_state
        self.path = [self.state]
        self.decision_period = decision_period
        self.historic_data = historic_data
        self.policy_name = policy
        self.policy = {
            "no_vaccines": self._no_vaccines,
            "random": self._random_policy,
            "population_based": self._population_based_policy,
            "infection_based": self._infection_based_policy,
            "age_based": self._age_based_policy,
        }[policy]

    def run(self, verbose=False):
        """ Updates states from current time_step to a specified horizon

        Returns
            A path that shows resulting traversal of states
        """
        print(f"\033[1mRunning MDP with policy: {self.policy_name}\033[0m")
        run_range = range(self.state.time_step, self.horizon) if verbose else tqdm(range(self.state.time_step, self.horizon))
        for _ in run_range:
            if verbose: print(self.state, end="\n"*2)
            if np.sum(self.state.R) / np.sum(self.population.population) > 0.7: # stop if recovered population is 70 % of total population
                print("\033[1mReached stop-criteria. Recovered population > 70%.\033[0m\n")
                break
            if np.sum(self.state.E1) < 1: # stop if infections are zero
                print("\033[1mReached stop-criteria. Infected population is zero.\033[0m\n")
                break
            self.update_state()
        return self.path

    def get_exogenous_information(self, state):
        """ Recieves the exogenous information at time_step t

        Parameters
            t: time_step
            state: state that 
        Returns:
            returns a dictionary of information contain 'alphas', 'vaccine_supply', 'contact_matrices_weights'
        """
        today = pd.Timestamp(state.date)
        end_of_decision_period = pd.Timestamp(state.date+timedelta(self.decision_period//4))
        mask = (self.historic_data['date'] > today) & (self.historic_data['date'] <= end_of_decision_period)
        week_data = self.historic_data[mask]
        if week_data.empty:
            alphas = [1, 1, 1, 1, 0.1]
            vaccine_supply = np.ones((356,5))*10
            contact_matrices_weights =  np.array([0.1,0.3,0.3,0.1,0.2])
        else:
            data = week_data.iloc[-1]
            alphas = [data['alpha_s'], data['alpha_e1'], data['alpha_e2'], data['alpha_a'], data['alpha_i']]
            vaccine_supply = week_data['vaccine_supply_new'].sum()
            contact_matrices_weights = [data['w_c1'], data['w_c2'], data['w_c3'], data['w_c4'], data['w_c5']]
        
        information = {'alphas': alphas, 'vaccine_supply': vaccine_supply, 'contact_matrices_weights': contact_matrices_weights}
        return information

    def update_state(self, decision_period=28):
        """ Updates the state of the decision process.

        Parameters
            decision_period: number of periods forward whein time that the decision directly affects
        """
        decision = self.policy()
        information = self.get_exogenous_information(self.state)
        self.state = self.state.get_transition(decision, information, self.epidemic_function.simulate, decision_period)
        self.path.append(self.state)

    def _no_vaccines(self):
        """ Define allocation of vaccines to zero

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        n_regions, n_age_groups = pop.shape
        return np.zeros(shape=(self.decision_period, n_regions, n_age_groups))

    def _random_policy(self):
        """ Define allocation of vaccines based on random distribution

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        n_regions, n_age_groups = pop.shape
        vaccine_allocation = np.zeros((self.decision_period, n_regions, n_age_groups))
        demand = self.state.S.copy()
        M = self.state.vaccines_available
        while M > 0:
            period, region, age_group = np.random.randint(self.decision_period), np.random.randint(n_regions), np.random.randint(n_age_groups)
            if demand[region][age_group] > 0:
                M -= 1
                vaccine_allocation[period][region][age_group] += 1
                demand[region][age_group] -= 1
        return vaccine_allocation

    def _population_based_policy(self):
        """ Define allocation of vaccines based on number of inhabitants in each region

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        vaccine_allocation = np.zeros((self.decision_period, pop.shape[0], pop.shape[1]))
        for i in range(self.decision_period):
            total_allocation = self.state.vaccines_available * self.state.S/np.sum(self.state.S)
            vaccine_allocation[i] = total_allocation/self.decision_period
        return vaccine_allocation

    def _infection_based_policy(self):
        """ Define allocation of vaccines based on number of infected in each region

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        vaccine_allocation = np.zeros((self.decision_period, pop.shape[0], pop.shape[1]))
        for i in range(self.decision_period):
            total_allocation = self.state.vaccines_available * self.state.E1/np.sum(self.state.E1)
            vaccine_allocation[i] = total_allocation/self.decision_period
        return vaccine_allocation

    def _age_based_policy(self):
        """ Define allocation of vaccines based on age prioritization (oldest first)

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        vaccine_allocation = np.zeros((self.decision_period, pop.shape[0], pop.shape[1]))
        M = self.state.vaccines_available
        demand = self.state.S.copy()

        def find_prioritized_age_group(demand):
            for a in range(pop.shape[1]-1,0,-1):
                if np.sum(demand[:,a]) > 0:
                    return a
                    
        age_group = find_prioritized_age_group(demand)
        for i in range(self.decision_period):
            age_group_demand = demand[:,age_group]
            total_age_group_demand = np.sum(age_group_demand)
            age_allocation = M * age_group_demand/total_age_group_demand
            allocation = np.zeros((pop.shape[0], pop.shape[1]))
            allocation[:,age_group] = age_allocation
            vaccine_allocation[i] = allocation/self.decision_period
            demand -= allocation
            if total_age_group_demand < 1:
                age_group = find_prioritized_age_group(demand)
        return vaccine_allocation
