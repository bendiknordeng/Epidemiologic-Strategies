from covid.utils import generate_weekly_data
from vaccine_allocation_model.State import State
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import timedelta

class MarkovDecisionProcess:
    def __init__(self, config, decision_period, population, epidemic_function, initial_state, horizon,
                policy, wave_weeks, government_strictness, verbose, historic_data=None):
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
        self.config = config
        self.decision_period = decision_period
        self.horizon = horizon
        self.population = population
        self.epidemic_function = epidemic_function
        self.state = initial_state
        self.path = [self.state]
        self.historic_data = historic_data
        self.wave_weeks = wave_weeks
        self.government_strictness = government_strictness
        self.policy_name = policy
        self.verbose = verbose
        self.policy = {
            "no_vaccines": self._no_vaccines,
            "random": self._random_policy,
            "population_based": self._population_based_policy,
            "susceptible_based": self._susceptible_based_policy,
            "infection_based": self._infection_based_policy,
            "adults_first": self._adults_first_policy,
            "oldest_first": self._oldest_first_policy,
        }[policy]

    def run(self):
        """ Updates states from current time_step to a specified horizon

        Returns
            A path that shows resulting traversal of states
        """
        print(f"\033[1mRunning MDP with policy: {self.policy_name}\033[0m")
        run_range = range(self.state.time_step, self.horizon) if self.verbose else tqdm(range(self.state.time_step, self.horizon))
        for week in run_range:
            if self.verbose: print(self.state, end="\n"*2)
            if np.sum(self.state.R) / np.sum(self.population.population) > 0.9: # stop if recovered population is 70 % of total population
                print("\033[1mReached stop-criteria. Recovered population > 90%.\033[0m\n")
                break
            if np.sum([self.state.E1, self.state.E2, self.state.A, self.state.I]) < 0.1: # stop if infections are zero
                print("\033[1mReached stop-criteria. Infected population is zero.\033[0m\n")
                break
            self.update_state()

    def get_exogenous_information(self, state):
        """ Recieves the exogenous information at time_step t

        Parameters
            t: time_step
            state: state that 
        Returns:
            returns a dictionary of information contain 'alphas', 'vaccine_supply', 'contact_matrices_weights', 'wave_incline', 'wave_decline'
        """
        today = pd.Timestamp(state.date)
        end_of_decision_period = pd.Timestamp(state.date+timedelta(self.decision_period//4))
        mask = (self.historic_data['date'] > today) & (self.historic_data['date'] <= end_of_decision_period)
        week_data = self.historic_data[mask]
        if week_data.empty:
            vaccine_supply = np.ones((356,5))*10
        else:
            vaccine_supply = int(week_data['vaccine_supply_new'].sum()/2) # supplied vaccines need two doses, model uses only one dose

        contact_weights, alphas = self._map_infection_to_control_measures(self.state.contact_weights, self.state.alphas)
        
        information = {'alphas': alphas, 
                       'vaccine_supply': vaccine_supply,
                       'contact_weights': contact_weights}
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

    def _map_infection_to_control_measures(self, previous_cw, previous_alphas):
        simulation_week = self.state.time_step//self.decision_period
        if simulation_week in self.wave_weeks:
            wave_strength = np.random.normal(1.5, 0.1)
            if self.verbose:
                print("\033[1mInfection wave\033[0m")
                print(f"Wavestrength: {wave_strength}\n\n")
                previous_cw *= wave_strength
                previous_alphas *= wave_strength

        new_infected_current = np.sum(self.state.new_infected)
        n_days = self.decision_period/self.config.periods_per_day
        if len(self.path) > 2 and new_infected_current >= n_days:
            new_infected_historic = np.sum(self.path[-3].new_infected)
            infection_rate = new_infected_current/new_infected_historic
            maximum_new_infected = max([np.sum(state.new_infected) for state in self.path])
            infected_per_100k = np.sum(self.state.I)/(self.population.population.sum()/1e5)
            increasing_trend = infection_rate > 1.15 and new_infected_current > 0.1 * maximum_new_infected
            decreasing_trend = infection_rate < 0.85
            slope = (new_infected_current-new_infected_historic)/n_days
            loc = 4/((1+np.exp(0.01*slope))*(1+np.exp(0.01*infected_per_100k)))
            factor = max(np.random.normal(loc, 0.05), 1-self.government_strictness)

            if self.verbose:
                if increasing_trend:
                    print("\033[1mIncreasing trend\033[0m")
                elif decreasing_trend:
                    print("\033[1mDecreasing trend\033[0m")
                else:
                    print("\033[1mNeutral trend\033[0m")
                print(f"R_eff: {self.state.r_eff:.2f}")
                print(f"Infected per 100k: {infected_per_100k:.1f}")
                print(f"New infected last week: {new_infected_historic}")
                print(f"New infected current week: {new_infected_current}")
                print(f"Maximum new infected: {maximum_new_infected}")
                print(f"Current infections/last week infections: {infection_rate:.3f}")
                print(f"Change in new infected per day: {slope:.3f}")
                print(f"Control measure factor: {factor:.3f}")
                print(f"Previous weights: {previous_cw}\n\n")

            if increasing_trend:
                return previous_cw * factor, previous_alphas * factor
            elif decreasing_trend:
                return previous_cw * factor,  previous_alphas * factor
            else:
                return previous_cw, previous_alphas
        else:
            return previous_cw, previous_alphas

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
        demand = self.state.S.copy()-(1-self.config.efficacy)*self.state.V.copy()
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
            if (i-1)%4 == 0 or (i-2)%4 == 0: # only vaccinate each morning and midday
                total_allocation = self.state.vaccines_available * pop/np.sum(pop)
                vaccine_allocation[i] = total_allocation/(self.decision_period/2) 
        return vaccine_allocation

    def _susceptible_based_policy(self):
        """ Define allocation of vaccines based on number of susceptible inhabitants in each region

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        vaccine_allocation = np.zeros((self.decision_period, pop.shape[0], pop.shape[1]))
        for i in range(self.decision_period):
            if (i-1)%4 == 0 or (i-2)%4 == 0: # only vaccinate each morning and midday
                total_allocation = self.state.vaccines_available * self.state.S/np.sum(self.state.S)
                vaccine_allocation[i] = total_allocation/(self.decision_period/2)
        return vaccine_allocation

    def _infection_based_policy(self):
        """ Define allocation of vaccines based on number of infected in each region

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        vaccine_allocation = np.zeros((self.decision_period, pop.shape[0], pop.shape[1]))
        for i in range(self.decision_period):
            if (i-1)%4 == 0 or (i-2)%4 == 0: # only vaccinate each morning and midday
                total_allocation = self.state.vaccines_available * self.state.E1/np.sum(self.state.E1)
                vaccine_allocation[i] = total_allocation/(self.decision_period/2)
        return vaccine_allocation


    def _population_density_policy(self):
        """ Define allocation of vaccines based on denisty of population per m^2

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pass

    def _adults_first_policy(self):
        """ Define allocation of vaccines based on age, prioritize the middle groups (epidemic drivers)

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        vaccine_allocation = np.zeros((self.decision_period, pop.shape[0], pop.shape[1]))
        M = self.state.vaccines_available
        demand = self.state.S.copy()-(1-self.config.efficacy)*self.state.V.copy()

        def find_prioritized_age_group(demand):
            for a in [3,4,5,6,7,2,1,0]:
                if np.sum(demand[:,a]) > 0:
                    return a
                    
        age_group = find_prioritized_age_group(demand)
        vaccines_per_period = M/(self.decision_period/2)
        for i in range(self.decision_period):
            if (i-1)%4 == 0 or (i-2)%4 == 0: # only vaccinate each morning and midday
                vaccines_left = vaccines_per_period
                allocation = np.zeros((pop.shape[0], pop.shape[1]))
                age_group_demand = demand[:,age_group]
                total_age_group_demand = np.sum(age_group_demand)
                if vaccines_per_period > total_age_group_demand:
                    age_allocation = age_group_demand
                    allocation[:,age_group] = age_allocation
                    demand[:,age_group] -= allocation[:,age_group]
                    vaccines_left -= total_age_group_demand
                    age_group = find_prioritized_age_group(demand)
                    age_group_demand = demand[:,age_group]
                    total_age_group_demand = np.sum(age_group_demand)
                age_allocation = (age_group_demand/total_age_group_demand) * vaccines_left
                M -= vaccines_per_period
                allocation[:,age_group] = age_allocation
                vaccine_allocation[i] = allocation
                demand[:,age_group] -= allocation[:,age_group]
        return vaccine_allocation

    def _oldest_first_policy(self):
        """ Define allocation of vaccines based on age, prioritize the oldest group

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        vaccine_allocation = np.zeros((self.decision_period, pop.shape[0], pop.shape[1]))
        M = self.state.vaccines_available
        demand = self.state.S.copy()-(1-self.config.efficacy)*self.state.V.copy()

        def find_prioritized_age_group(demand):
            for age_group in range(pop.shape[1]-1,0,-1):
                if np.round(np.sum(demand[:,age_group])) > 0:
                    return age_group
                    
        age_group = find_prioritized_age_group(demand)
        vaccines_per_period = M/(self.decision_period/2)
        for i in range(self.decision_period):
            if (i-1)%4 == 0 or (i-2)%4 == 0: # only vaccinate each morning and midday
                vaccines_left = vaccines_per_period
                allocation = np.zeros((pop.shape[0], pop.shape[1]))
                age_group_demand = demand[:,age_group]
                total_age_group_demand = np.sum(age_group_demand)
                if vaccines_per_period > total_age_group_demand:
                    age_allocation = age_group_demand
                    allocation[:,age_group] = age_allocation
                    demand[:,age_group] -= allocation[:,age_group]
                    vaccines_left -= total_age_group_demand
                    age_group = find_prioritized_age_group(demand)
                    age_group_demand = demand[:,age_group]
                    total_age_group_demand = np.sum(age_group_demand)
                age_allocation = (age_group_demand/total_age_group_demand) * vaccines_left
                M -= vaccines_per_period
                allocation[:,age_group] = age_allocation
                vaccine_allocation[i] = allocation
                demand[:,age_group] -= allocation[:,age_group]
        return vaccine_allocation
