import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import timedelta
import time
from covid.utils import get_wave_timeline, tcolors

class MarkovDecisionProcess:
    def __init__(self, config, decision_period, population, epidemic_function, initial_state,
                response_measure_model, use_response_measures, horizon, policy, verbose, historic_data=None):
        """A Markov decision process adminestering states, decisions and exogeneous information for an epidemic.

        Args:
            config (namedtuple): case specific data
            decision_period (int): number of time steps between each decision
            population (pandas.DataFrame): information about population in reions and age groups
            epidemic_function (function): executable simulating the current step of the epidemic
            initial_state (State): initial state object for the simulation
            response_measure_model (dict, dict): dictionaries with an MLPClassifier and a StandardScaler for each response measure
            use_response_measures (boolean): True if the simulation should involve response measures
            horizon (int): giving the number of decision periods to simulate
            policy (string): indicating which vaccine allocation policy to follow
            verbose (boolean): True if one wants information in the form of terminal output
            historic_data (pandas.DataFrame, optional): historic data from Folkehelseinstituttet (FHI) regarding vaccine supply. Defaults to None.
        """
        self.config = config
        self.decision_period = decision_period
        self.population = population
        self.epidemic_function = epidemic_function
        self.initial_state = initial_state
        self.response_measure_model = response_measure_model
        self.use_response_measures = use_response_measures
        self.horizon = horizon
        self.policy = policy
        self.verbose = verbose
        self.historic_data = historic_data
        self.weighted_policy_weights = None
        self.wave_timeline = None
        self.wave_state_timeline = None
    
    def init(self):
        """Resets the MarkovDecisionProcess to make multible runs possible"""
        self.initial_state.wave_state = 'U'
        self.initial_state.wave_count = {"U": 1, "D": 0, "N": 0}
        self.initial_state.strategy_count.clear()
        self.state = self.initial_state
        self.path = [self.state]
        self.wave_timeline, self.wave_state_timeline = get_wave_timeline(self.horizon, self.decision_period, self.config.periods_per_day)

    def run(self, weighted_policy_weights=None):
        """ Updates states from current time_step to a specified horizon

        Returns
            A path that shows resulting traversal of states
        """
        run_range = range(self.state.time_step, self.horizon) if self.verbose or weighted_policy_weights is not None else tqdm(range(self.state.time_step, self.horizon))
        for week in run_range:
            if self.verbose: print(self.state, end="\n"*2)
            if self.check_stop_criteria(week):
                break
            self.update_state(weighted_policy_weights, week)
    
    def check_stop_criteria(self, week):
        if np.sum(self.state.R) / np.sum(self.population.population) > 0.9: # stop if recovered population is 70 % of total population
            print(f"{tcolors.BOLD}Reached stop-criteria week {week}. Recovered population > 90%.{tcolors.ENDC}\n")
            return True
        if np.sum([self.state.E1, self.state.E2, self.state.A, self.state.I]) < 0.1: # stop if infections are zero
            print(f"{tcolors.BOLD}Reached stop-criteria on week {week}. Infected population is zero.{tcolors.ENDC}\n")
            return True
        if np.sum(self.state.V) / np.sum(self.population.population) > 0.3:
            print(f"{tcolors.BOLD}Reached stop-criteria week {week}. Vaccinated population > 30%.{tcolors.ENDC}\n")
            return True
        return False

    def get_exogenous_information(self, state, week):
        """ Receives the exogenous information at time_step t

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
            'R': self.wave_timeline[week],
            'wave_state': self.wave_state_timeline[week],
            'contact_weights': contact_weights,
            'alphas': alphas,
            'flow_scale': flow_scale
            }
        return information

    def update_state(self, weighted_policy_weights, week):
        """ Updates the state of the decision process.

        Parameters
            decision_period: number of periods forward whein time that the decision directly affects
        """
        if weighted_policy_weights is not None:
            i = {"U": 0, "D": 1, "N": 2}[self.state.wave_state]
            j = self.state.wave_count[self.state.wave_state]
            self.weighted_policy_weights = weighted_policy_weights[i][j-1]
        decision = self.policy.get_decision(self.state, self.state.vaccines_available)
        information = self.get_exogenous_information(self.state, week)
        self.state = self.state.get_transition(decision, information, self.epidemic_function.simulate, self.decision_period)
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