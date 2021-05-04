import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import timedelta
import time

class MarkovDecisionProcess:
    def __init__(self, config, decision_period, population, epidemic_function, initial_state, 
                response_measure_model, use_response_measures, wave_timeline, wave_state_timeline, 
                horizon, policy, verbose, historic_data=None):
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
        self.initial_state = initial_state
        self.response_measure_model = response_measure_model
        self.use_response_measures = use_response_measures
        self.wave_timeline = wave_timeline
        self.wave_state_timeline = wave_state_timeline
        self.historic_data = historic_data
        self.policy_name = policy
        self.verbose = verbose
        self.policies = {
            "random": self._random_policy,
            "no_vaccines": self._no_vaccines,
            "susceptible_based": self._susceptible_based_policy,
            "infection_based": self._infection_based_policy,
            "oldest_first": self._oldest_first_policy,
            "weighted": self._weighted_policy
        }
        self.policy = self.policies[policy]
        self.weighted_policy_weights = np.array([1,0,0,0]) # default no vaccines-policy
        self.reset()
    
    def reset(self):
        self.initial_state.wave_count = {"U":0, "D":0, "N":0}
        self.state = self.initial_state
        self.path = [self.state]

    def run(self, weighted_policy_weights):
        """ Updates states from current time_step to a specified horizon

        Returns
            A path that shows resulting traversal of states
        """
        if self.policy != self._weighted_policy:
            print(f"\033[1mRunning MDP with policy: {self.policy_name}\033[0m")
        run_range = range(self.state.time_step, self.horizon) if self.verbose else tqdm(range(self.state.time_step, self.horizon))
        for week in run_range:
            self.week = week
            if self.verbose: print(self.state, end="\n"*2)
            if np.sum(self.state.R) / np.sum(self.population.population) > 0.9: # stop if recovered population is 70 % of total population
                print("\033[1mReached stop-criteria. Recovered population > 90%.\033[0m\n")
                break
            if np.sum([self.state.E1, self.state.E2, self.state.A, self.state.I]) < 0.1: # stop if infections are zero
                print("\033[1mReached stop-criteria. Infected population is zero.\033[0m\n")
                break
            self.update_state(weighted_policy_weights)

    def get_exogenous_information(self, state):
        """ Recieves the exogenous information at time_step t

        Parameters
            t: time_step
            state: state that 
        Returns:
            returns a dictionary of information contain 'flow_scale', 'vaccine_supply', 'contact_matrices_weights', 'wave_incline', 'wave_decline'
        """
        today = pd.Timestamp(state.date)
        end_of_decision_period = pd.Timestamp(state.date+timedelta(self.decision_period//self.config.periods_per_day))
        mask = (self.historic_data['date'] > today) & (self.historic_data['date'] <= end_of_decision_period)
        week_data = self.historic_data[mask]
        if week_data.empty:
            vaccine_supply = np.zeros(self.state.S.shape)
        else:
            vaccine_supply = int(week_data['vaccine_supply_new'].sum()/2) # supplied vaccines need two doses, model uses only one dose
        if self.use_response_measures:
            contact_weights, alphas, flow_scale = self._map_infection_to_response_measures(self.state.contact_weights, self.state.alphas, self.state.flow_scale)
        else:
            contact_weights, alphas, flow_scale = self.config.initial_contact_weights, self.config.initial_alphas, self.config.initial_flow_scale
        information = {
            'vaccine_supply': vaccine_supply,
            'R': self.wave_timeline[self.week],
            'wave_state': self.wave_state_timeline[self.week],
            'contact_weights': contact_weights,
            'alphas': alphas,
            'flow_scale': flow_scale
            }

        return information

    def update_state(self, weighted_policy_weights, decision_period=28):
        """ Updates the state of the decision process.

        Parameters
            decision_period: number of periods forward whein time that the decision directly affects
        """
        i = {"U":0, "D":1, "N":2}[self.state.wave_state]
        j = self.state.wave_count[self.state.wave_state]
        self.weighted_policy_weights = weighted_policy_weights[i][j-1]
        decision = self.policy()
        information = self.get_exogenous_information(self.state)
        self.state = self.state.get_transition(decision, information, self.epidemic_function.simulate, decision_period)
        self.path.append(self.state)

    def _map_infection_to_response_measures(self, previous_cw, previous_alphas, previous_flow_scale):
        if len(self.path) > 3:
            # Features for cases of infection
            active_cases = np.sum(self.state.I) * 1e5/self.population.population.sum()
            cumulative_total_cases = np.sum(self.state.total_infected) * 1e5/self.population.population.sum()
            cases_past_week = np.sum(self.state.new_infected) * 1e5/self.population.population.sum()
            cases_2w_ago = np.sum(self.path[-2].new_infected) * 1e5/self.population.population.sum()

            # Features for deaths
            cumulative_total_deaths = np.sum(self.state.D) * 1e5/self.population.population.sum()
            deaths_past_week = np.sum(self.state.new_deaths) * 1e5/self.population.population.sum()
            deaths_2w_ago = np.sum(self.path[-2].new_deaths) * 1e5/self.population.population.sum()

            features = np.array([active_cases, cumulative_total_cases, cases_past_week, cases_2w_ago, 
                                cumulative_total_deaths, deaths_past_week, deaths_2w_ago])

            models, scalers = self.response_measure_model

            # Contact weights
            initial_cw = np.array(self.config.initial_contact_weights)
            cw_mapper = {
                'home': lambda x: initial_cw[0] + x * 0.1,
                'school': lambda x: initial_cw[1] - x * 0.3,
                'work': lambda x: initial_cw[2] - x * 0.3,
                'public': lambda x: initial_cw[3] - x * 0.2
            }
            new_cw = []
            for category in ['home', 'school', 'work', 'public']:
                input = scalers[category].transform(features.reshape(1,-1))
                measure = models[category].predict(input)[0]
                new_cw.append(cw_mapper[category](measure))

            # Alphas
            initial_alphas = np.array(self.config.initial_alphas)
            alpha_mapper = {
                'E2': lambda x: initial_alphas[0]*(1 - x * 1e-4),
                'A': lambda x: initial_alphas[1]*(1 - x * 1e-4),
                'I': lambda x: initial_alphas[2]*(1 - x * 4e-3)
            }
            input = scalers['alpha'].transform(features.reshape(1,-1))
            measure = min(max(models['alpha'].predict(input)[0], 1), 100)
            new_alphas = []
            for comp in ['E2', 'A', 'I']:
                new_alphas.append(alpha_mapper[comp](measure))

            input = scalers['movement'].transform(features.reshape(1,-1))
            measure = models['movement'].predict(input)[0]
            new_flow_scale = self.config.initial_flow_scale - 0.1 * measure

            if self.verbose:
                print("Per 100k:")
                print(f"Active cases: {active_cases:.3f}")
                print(f"Cumulative cases: {cumulative_total_cases:.3f}")
                print(f"New infected last week: {cases_past_week:.3f}")
                print(f"New infected two weeks ago: {cases_2w_ago:.3f}")
                print(f"Cumulative deaths: {cumulative_total_deaths:.3f}")
                print(f"New deaths last week: {deaths_past_week:.3f}")
                print(f"New deaths two weeks ago: {deaths_2w_ago:.3f}")
                print(f"Previous weights: {previous_cw}")
                print(f"New weights: {new_cw}")
                print(f"Previous alphas: {previous_alphas}")
                print(f"New alphas: {new_alphas}")
                print(f"Previous flow scale: {previous_flow_scale}")
                print(f"New flow scale: {new_flow_scale}\n\n")
                time.sleep(0.2)

            return new_cw, new_alphas, new_flow_scale
        
        return previous_cw, previous_alphas, previous_flow_scale

    def _random_policy(self):
        """ Define allocation of vaccines based on random distribution

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        n_regions, n_age_groups = pop.shape
        vaccine_allocation = np.zeros((n_regions, n_age_groups))
        demand = self.state.S.copy()-(1-self.config.efficacy)*self.state.V.copy()
        M = self.state.vaccines_available
        while M > 0:
            possible_regions = np.nonzero(demand > 0)[0]
            region = np.random.choice(possible_regions)
            possible_age_groups = np.nonzero(demand[region] > 0)[0]
            age_group = np.random.choice(possible_age_groups)
            allocation = np.min([M, demand[region][age_group], 1]) # consider fractional populations
            M -= allocation
            vaccine_allocation[region][age_group] += allocation
            demand[region][age_group] -= allocation
        decision = np.minimum(demand, vaccine_allocation).clip(min=0)
        return decision

    def _no_vaccines(self, M=None):
        """ Define allocation of vaccines to zero

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        return np.zeros(pop.shape)

    def _susceptible_based_policy(self, M=None):
        """ Define allocation of vaccines based on number of susceptible inhabitants in each region

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        vaccine_allocation = np.zeros(pop.shape)
        demand = self.state.S.copy()-(1-self.config.efficacy)*self.state.V.copy()
        if M is None: M = self.state.vaccines_available
        if M > 0:
            vaccine_allocation = M * demand/np.sum(demand)
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            return decision
        return vaccine_allocation

    def _infection_based_policy(self, M=None):
        """ Define allocation of vaccines based on number of infected in each region

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        vaccine_allocation = np.zeros(pop.shape)
        total_infection = np.sum(self.state.I)
        if M is None: M = self.state.vaccines_available
        if M > 0:
            if total_infection > 0:
                demand = self.state.S.copy()-(1-self.config.efficacy)*self.state.V.copy()
                infection_density = self.state.I.sum(axis=1)/total_infection
                regional_allocation = M * infection_density
                vaccine_allocation = demand * regional_allocation.reshape(-1,1)/demand.sum(axis=1).reshape(-1,1)
                decision = np.minimum(demand, vaccine_allocation).clip(min=0)
                return decision
        return vaccine_allocation

    def _oldest_first_policy(self, M=None):
        """ Define allocation of vaccines based on age, prioritize the oldest group

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        vaccine_allocation = np.zeros(pop.shape)
        if M is None: M = self.state.vaccines_available
        if M > 0:
            demand = self.state.S.copy()-(1-self.config.efficacy)*self.state.V.copy()
            for age_group in range(pop.shape[1]-1,0,-1):
                age_group_demand = demand[:,age_group]
                total_age_group_demand = np.sum(age_group_demand)
                if M < total_age_group_demand:
                    vaccine_allocation[:,age_group] = M * age_group_demand/total_age_group_demand
                    decision = np.minimum(demand, vaccine_allocation).clip(min=0)
                    return decision
                else:
                    vaccine_allocation[:,age_group] = M * age_group_demand/total_age_group_demand
                    M -= total_age_group_demand
                    demand[:,age_group] -= age_group_demand
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            return decision
        return vaccine_allocation

    def _weighted_policy(self):
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        vaccine_allocation = np.zeros(pop.shape)
        weighted_policies = ["no_vaccines", "susceptible_based", "infection_based", "oldest_first"]
        M = self.state.vaccines_available
        if M > 0:
            demand = self.state.S.copy()-(1-self.config.efficacy)*self.state.V.copy()
            vaccines_per_policy = M * self.weighted_policy_weights
            for i, policy in enumerate(weighted_policies):
                vaccine_allocation += self.policies[policy](M=vaccines_per_policy[i])
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            return decision
        return vaccine_allocation