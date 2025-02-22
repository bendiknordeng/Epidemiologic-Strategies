import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import timedelta
from utils import get_wave_timeline, tcolors
from copy import copy, deepcopy

class MarkovDecisionProcess:
    def __init__(self, config, decision_period, population, epidemic_function, 
                initial_state, response_measure_model, use_response_measures, 
                use_wave_factor, horizon, end_date, policy, verbose, historic_data=None):
        """ A Markov decision process adminestering states, decisions and exogeneous information for an epidemic

        Args:
            config (namedtuple): case specific data
            decision_period (int): number of time steps between each decision
            population (pandas.DataFrame): information about population in reions and age groups
            epidemic_function (function): executable simulating the current step of the epidemic
            initial_state (State): initial state object for the simulation
            response_measure_model (dict, dict): dictionaries with an MLPClassifier and a StandardScaler for each response measure
            use_response_measures (boolean): True if the simulation should involve response measures
            horizon (int): giving the number of decision periods to simulate
            policy (Policy): vaccine allocation policy
            verbose (boolean): True if one wants information in the form of terminal output
            historic_data (pandas.DataFrame, optional): historic data from Folkehelseinstituttet (FHI) regarding vaccine supply. Defaults to None.
        """
        self.config = config
        self.decision_period = decision_period
        self.population = population
        self.epidemic_function = epidemic_function
        self.response_measure_model = response_measure_model
        self.use_response_measures = use_response_measures
        self.use_wave_factor = use_wave_factor
        self.horizon = horizon
        self.end_date = end_date
        self.policy = policy
        self.verbose = verbose
        self.historic_data = historic_data
        self.initial_state = initial_state
    
    def init(self):
        self.state = deepcopy(self.initial_state)
        self.path = [self.state]
        self.simulation_period = 0
        self.epidemic_function.daily_cases = []
        if self.use_response_measures:
            self.measures_timeline_generated = False
        if self.use_wave_factor:
            self.wave_timeline, self.wave_state_timeline = get_wave_timeline(self.horizon, self.decision_period, self.config.periods_per_day)
        initial_run = True
        while initial_run:
            initial_run = self.update_state()
            self.path.append(self.state)
            self.simulation_period += 1
        counter = 0
        if self.reached_stop_criteria() and counter < 5:
            self.init()
            counter += 1
        self.start_state = self.state
        self.start_path = self.path
        self.start_simulation_period = self.simulation_period
        if self.use_wave_factor:
            self.start_wave_timeline = self.wave_timeline
            self.start_wave_state_timeline = self.wave_state_timeline
        if self.verbose: print(f"\n{tcolors.BOLD}Starting state:\n{self.start_state}{tcolors.ENDC}")

    def reset(self, reset_measures=True):
        """ Resets the MarkovDecisionProcess to make multible runs possible"""
        self.measures_timeline_generated = False
        self.state = deepcopy(self.start_state)
        self.state.trend_count = dict.fromkeys(self.state.trend_count, 0)
        self.state.trend_count[self.state.trend] += 1
        self.epidemic_function.reset(self.state.time_step//self.config.periods_per_day)
        self.path = copy(self.start_path)
        self.simulation_period = self.start_simulation_period
        self.policy.fhi_vaccine_plan = pd.read_csv("data/fhi_vaccine_plan.csv")
        if self.use_response_measures and reset_measures:
            self._reset_measures_timeline()
            self.measures_timeline_generated = True
        if self.use_wave_factor:
            self.wave_timeline, self.wave_state_timeline = get_wave_timeline(self.horizon, self.decision_period, 
                                                                        self.config.periods_per_day, self.start_wave_timeline, 
                                                                        self.start_wave_state_timeline, self.simulation_period)                                                                    

    def run(self, weighted_policy_weights=None):
        """ Updates states from current time_step until horizon is reached"""
        while not self.reached_stop_criteria():
            if self.verbose: print(self.state, end="\n"*2)
            self.update_state(weighted_policy_weights)
            self.path.append(self.state)
            self.simulation_period += 1
    
    def reached_stop_criteria(self):
        """ Checks if a stop criteria is reached

        Returns:
            boolean: True if stop criteria is reached
        """
        if self.simulation_period == self.horizon:
            return True
        if np.sum(self.state.R) / np.sum(self.population.population) > 0.7: # stop if recovered population is 70 % of total population
            if self.verbose: print(f"{tcolors.BOLD}Reached stop-criteria in decision period {self.simulation_period}. Recovered population > 70%.{tcolors.ENDC}\n")
            return True
        
        if np.sum([self.state.E1, self.state.E2, self.state.A, self.state.I]) < 1: # stop if infections are zero
            if self.verbose: print(f"{tcolors.BOLD}Reached stop-criteria in decision period {self.simulation_period}. Infected population is zero.{tcolors.ENDC}\n")
            return True
        return False

    def get_exogenous_information(self):
        """ Retrieves the exogenous information for the current decision period

        Returns:
            dict: exogeneous information regarding 'vaccine_supply', 'wave_factor', 'wave_state', 'contact_weights', 'alphas' and 'flow_scale'
        """
        today = pd.Timestamp(self.state.date)
        end_of_decision_period = pd.Timestamp(self.state.date+timedelta(self.decision_period//self.config.periods_per_day))
        mask = (self.historic_data['date'] > today) & (self.historic_data['date'] <= end_of_decision_period)
        decision_period_data = self.historic_data[mask]
        if decision_period_data.empty:
            vaccine_supply = np.zeros(self.state.S.shape)
        else:
            vaccine_supply = int(decision_period_data['vaccine_supply_new'].sum()/2) # supplied vaccines need two doses, model uses only one dose
        information = {'vaccine_supply': vaccine_supply}
                            
        if self.use_response_measures:
            if self.measures_timeline_generated:
                timeline_period = self.simulation_period-self.start_simulation_period
                contact_weights = self.measures_timeline['contact_weights'][timeline_period]
                flow_scale = self.measures_timeline['flow_scale'][timeline_period]
            else:
                contact_weights, flow_scale = self._map_infection_to_response_measures(self.state.contact_weights, self.state.flow_scale)
        else:
            contact_weights, flow_scale = self.config.initial_contact_weights, self.config.initial_flow_scale
        if self.use_wave_factor:
            information['wave_factor'] = self.wave_timeline[self.simulation_period]

        information["contact_weights"] = contact_weights
        information["flow_scale"] = flow_scale
        return information

    def update_state(self, weighted_policy_weights=None):
        """ Updates the state

        Args:
            weighted_policy_weights (numpy.ndarray, optional): weights for the different policies if current policy is weighted. Defaults to None
        Returns:
            boolean: True if initial runs are complete (runs before decisions are relevant)
        """
        decision = self.policy.get_decision(self.state, self.state.vaccines_available, weighted_policy_weights)
        information = self.get_exogenous_information()
        self.state = self.state.get_transition(decision, information, self.epidemic_function.simulate, self.decision_period)
        return self.state.vaccines_available == 0 # stopping criteria for initial run

    def _map_infection_to_response_measures(self, previous_cw, previous_flow_scale):
        """ Maps infection numbers to response measure using neural network models

        Args:
            previous_cw (numpy.ndarray): previous contact weights
            previous_flow_scale (numpy.ndarray): previous mobility scale

        Returns:
            (numpy.ndarray): new contact weights 
            (numpy.ndarray): new mobility scales
        """
        if len(self.path) > 2:
            # Features for cases of infection
            active_cases = np.sum(self.state.I) * 1e5/self.population.population.sum()
            cumulative_total_cases = np.sum(self.state.total_infected) * 1e5/self.population.population.sum()
            cases_past_week = np.sum(self.state.new_infected) * 1e5/self.population.population.sum()
            cases_2w_ago = np.sum(self.path[-1].new_infected) * 1e5/self.population.population.sum()

            # Features for deaths
            cumulative_total_deaths = np.sum(self.state.D) * 1e5/self.population.population.sum()
            deaths_past_week = np.sum(self.state.new_deaths) * 1e5/self.population.population.sum()
            deaths_2w_ago = np.sum(self.path[-1].new_deaths) * 1e5/self.population.population.sum()

            # Effective reproduction number feature
            R_t = self.state.R_t

            features = np.array([active_cases, cumulative_total_cases, cases_past_week, cases_2w_ago, 
                                cumulative_total_deaths, deaths_past_week, deaths_2w_ago, R_t])

            models, scalers = self.response_measure_model

            # Contact weights
            initial_cw = np.array(self.config.initial_contact_weights)
            cw_mapper = {
                'home': lambda x: initial_cw[0] * (1 + x * 0.07), # x in [0, 1, 2, 3]
                'school': lambda x: initial_cw[1] * (1 - x * 0.27), # x in [0, 1, 2, 3]
                'work': lambda x: initial_cw[2] * (1 - x * 0.25), # x in [0, 1, 2, 3]
                'public': lambda x: initial_cw[3] * (1 - x * 0.20) # x in [0, 1, 2, 3, 4]
            }
            
            new_cw = []
            for category in ['home', 'school', 'work', 'public']:
                input = scalers[category].transform(features.reshape(1,-1))
                measure = models[category].predict(input)[0]
                new_cw.append(cw_mapper[category](measure))

            input = scalers['movement'].transform(features.reshape(1,-1))
            measure = models['movement'].predict(input)[0]
            new_flow_scale = self.config.initial_flow_scale * (1 - measure * 0.3) # measure in [0, 1, 2]

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
                print(f"Previous flow scale: {previous_flow_scale}")
                print(f"New flow scale: {new_flow_scale}\n\n")

            return new_cw, new_flow_scale
        
        return previous_cw, previous_flow_scale

    def _reset_measures_timeline(self):
        current_policy = self.policy.vaccine_allocation
        self.policy.vaccine_allocation = self.policy.policies['fhi_policy']
        self.measures_timeline = {"contact_weights": [], "flow_scale": []}
        while self.simulation_period < self.horizon:
            if self.verbose: print(self.state, end="\n"*2)
            self.update_state(weighted_policy_weights=None)
            self.measures_timeline["contact_weights"].append(self.state.contact_weights)
            self.measures_timeline["flow_scale"].append(self.state.flow_scale)
            self.path.append(self.state)
            self.simulation_period += 1
        self.reset(reset_measures=False)
        self.policy.vaccine_allocation = current_policy
        
    def __str__(self):
        status = f"{tcolors.BOLD}MarkovDecisionProcess object.{tcolors.ENDC}\n"
        status = f"Horizon: {self.horizon}\n"
        status += f"Decision period: {self.decision_period}\n"
        status += f"Policy: {self.policy}\n"
        return status

    def __repr__(self):
        status = f"{tcolors.BOLD}MarkovDecisionProcess object.{tcolors.ENDC}\n"
        status += f"Horizon: {self.horizon}\n"
        status += f"Decision period: {self.decision_period}\n"
        status += f"Policy: {self.policy}\n"
        return status