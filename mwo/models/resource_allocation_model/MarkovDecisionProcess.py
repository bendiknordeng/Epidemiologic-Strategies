__author__ = "Martin Willoch Olstad"
__email__ = "martinwilloch@gmail.com"

from ortools.sat.python import cp_model
import time
import numpy as np
import random
import sys
from resource_allocation_model.ValueFunctionApproximation import ValueFunctionApproximation
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from scipy.special import softmax
from sklearn.preprocessing import normalize
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

class MarkovDecisionProcess:

    def __init__(self, state, horizon, epidemic_function, plot_function, cumulative_function, map_plot_function, params=None,
                 decision_period=7.0, dt=0.1, training_size=360, hidden_dims=None, case=None, load=False, demand_relaxation=False, num_regions=None):
        self.state = state
        self.path = [self.state]
        self.horizon = horizon
        self.time = 0
        self.epidemic_function = epidemic_function
        self.plot_function = plot_function
        self.map_plot_function = map_plot_function
        if params is not None:
            self.num_regions = params['num_regions']
            self.num_facility_types = params['num_facility_types']
            self.num_intervention_types = params['num_intervention_types']
            self.total_personnel = params['total_personnel']
            self.resources_available = params['resources_available']
            self.facility_intervention_capacity = params['facility_intervention_capacity']
            self.intervention_personnel_capacity = params['intervention_personnel_capacity']
            self.facility_personnel_capacity = params['facility_personnel_capacity']
            self.locations_available = params['locations_available']
            self.epsilon = params['epsilon']
            self.aggregation_constant = params['aggregation_constant']
        state_dim = self.num_intervention_types+2*self.num_regions+1
        if hidden_dims is None:
            self.value_function = ValueFunctionApproximation(input_dim=state_dim, output_dim=1, hidden_dims=params['hidden_dims'])
        else:
            self.value_function = ValueFunctionApproximation(input_dim=state_dim, output_dim=1, hidden_dims=hidden_dims)
        self.lr = params['learning_rate']
        self.optimizer = optim.Adam(self.value_function.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.loss = []
        self.iteration_loss = []
        self.update_loss = []
        self.predictions = []

        self.cumulative_function = cumulative_function
        self.case = case

        self.load = load

        self.demand_relaxation = demand_relaxation

        if num_regions is not None:
            self.num_regions = num_regions

        self.scenarios = [0, 0.025, 0.25]
        self.scenario_probs = [0.25, 0.5, 0.25]

        self.num_scenarios = len(self.scenarios)

        self.training_size = training_size

        self.all_observations = torch.tensor(data=[])
        self.all_targets = torch.tensor(data=[])
        self.observations = torch.tensor(data=[])
        self.targets = torch.tensor(data=[])
        self.test_observations = torch.tensor(data=[])
        self.test_targets = torch.tensor(data=[])

        self.costs = np.zeros(int(self.horizon + 1))

        self.decision_period = decision_period
        self.dt = dt

        self.decision_functions = {'ADP':self.get_adp_decision,
                                   'Myopic':self.get_myopic_decision,
                                   'Greedy':self.get_greedy_decision,
                                   'Random':self.get_random_decision,
                                   'Nothing':self.get_nothing_decision,
                                   'Myopic_Ceil':self.get_myopic_decision,
                                   'Myopic_Round':self.get_myopic_decision,
                                   'Myopic_Floor':self.get_myopic_decision}



    def get_cost(self, new_state):
        return new_state.fatalities-self.state.fatalities

    def get_exogenous_information(self, elements=(0.0, 0.0125, 0.025, 0.0375, 0.05),
                                  probs=(0.20, 0.20, 0.20, 0.20, 0.20)):
        return np.random.choice(a=self.scenarios, p=self.scenario_probs)

    def update_value_observations(self, new_state, target, test=False):
        state_input = new_state.to_tensor()
        if test:
            self.test_observations = torch.cat((self.test_observations, state_input), 0)
            self.test_targets = torch.cat((self.test_targets, target), 0)
        else:
            self.all_observations = torch.cat((self.all_observations, state_input), 0)
            self.all_targets = torch.cat((self.all_targets, target), 0)
            if self.observations.shape[0] >= self.training_size:
                self.observations = torch.cat((self.observations[1:], state_input))
                self.targets = torch.cat((self.targets[1:], target))
            else:
                self.observations = torch.cat((self.observations, state_input), 0)
                self.targets = torch.cat((self.targets, target), 0)

    def update_value_function(self, epochs=10, mbs=32):
        self.value_function.train()

        mean = self.observations.mean(dim=0, keepdim=True)
        std = self.observations.std(dim=0, keepdim=True)
        standardized_observations = (self.observations - mean) / std
        standardized_observations = torch.where(torch.isnan(standardized_observations),
                                                torch.zeros_like(standardized_observations), standardized_observations)

        dataset = TensorDataset(standardized_observations, self.targets.unsqueeze(-1))

        for e in range(epochs):
            rand_sampler = RandomSampler(dataset, num_samples=mbs, replacement=True)
            mini_batches = DataLoader(dataset, batch_size=mbs, sampler=rand_sampler, drop_last=True)
            for (observation, target) in mini_batches:
                predicted = self.value_function(observation)
                loss = self.loss_func(predicted, target)
                self.loss.append(loss)
                self.update_loss.append(loss.item())

                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()

        self.value_function.eval()

    def initialize_policy(self):
        self.time = 0
        self.state = self.path[0]
        self.path = [self.state]
        self.costs = np.zeros(int(self.horizon + 1))

    def get_adp_decision(self):
        self.value_function.eval()
        decision = self.get_greedy_decision()
        decision = self.local_search(decision=decision)
        return decision

    def get_myopic_decision(self, aggregation='ceil'):
        decision = np.zeros(shape=(self.num_regions, self.num_intervention_types))
        available_resources = np.zeros(shape=self.num_intervention_types)
        if self.time+self.decision_period < self.horizon:
            cumulative_cases = self.cumulative_function(begin=self.time, end=self.time + self.decision_period)
            demands = np.zeros(shape=self.state.demands.shape)
            demands[:, 0] = np.copy(self.state.demands[:, 0])
            demands[:, 1] = np.copy(cumulative_cases[-1])
        else:
            demands = np.copy(self.state.demands)
        #ceiled_demand = np.ceil(demands / self.aggregation_constant)
        ceiled_demand = np.floor(demands / self.aggregation_constant)
        print('MYOPIC DEMANDS')
        print(demands)
        print(ceiled_demand)

        decision_ratio = normalize(ceiled_demand, norm='l1',axis=0)

        aggregated_demand = np.sum(ceiled_demand, axis=0)

        available_resources[0] = min(aggregated_demand[0], self.state.resources[0])
        available_resources[1] = min(aggregated_demand[0], self.state.resources[1])

        available_resources[2] = min(aggregated_demand[1], self.state.resources[2])
        available_resources[3] = min(aggregated_demand[1], self.state.resources[3])

        for m in range(self.num_intervention_types):
            decision[:,m] = self.basic_rounding(ratios=decision_ratio[:,1], items=available_resources[m])

        return decision

    def get_greedy_decision(self, step_size=10):
        decision = self.get_initial_decision()
        if self.is_feasible(resource_allocation=decision):
            best_decision = decision
        else:
            while not self.is_feasible(resource_allocation=decision):
                print('Infeasible greedy decision')
                print(decision)
                neg_marginal_cost, pos_marginal_cost = self.calculate_marginal_costs(
                    initial_decision=decision, greedy=True)
                neg_marginal_cost[decision <= 0] = np.inf

                neg_ix = np.unravel_index(np.argmin(neg_marginal_cost, axis=None), neg_marginal_cost.shape)

                print(neg_marginal_cost)
                print('Neg ix: ',neg_ix)

                decision[neg_ix] -= np.ceil(0.5*decision[neg_ix])
                while self.is_feasible(decision):
                    decision[neg_ix] += np.ceil(0.5*decision[neg_ix])
                decision[neg_ix] -= np.ceil(0.5*decision[neg_ix])


            best_decision = decision
        return best_decision

    def get_nothing_decision(self, ors_covered=False):
        decision = np.zeros(shape=(self.num_regions, self.num_intervention_types), dtype=int)
        return decision

    def policy(self, policy_type='ADP', information_path=None, verbose=False, epidemic_plots=False, val_func_plots=False, policy_plot=False, old=False):
        self.initialize_policy()

        vaccine_allocation = []
        disinfectant_allocation = []
        rehydration_allocation = []
        antibiotics_allocation = []
        for t in tqdm(range(int(self.horizon))):
            if t % self.decision_period == 0:
                decision = self.decision_functions[policy_type]()
                aggregated_decision = np.sum(decision, axis=0)
                print('RESOURCES: ')
                print(self.state.resources)
                print('AGGREGATED DECISION: ')
                print(aggregated_decision)
                vaccine_allocation.append(aggregated_decision[0])
                disinfectant_allocation.append(aggregated_decision[1])
                rehydration_allocation.append(aggregated_decision[2])
                antibiotics_allocation.append(aggregated_decision[3])

                if information_path is None:
                    information = self.get_exogenous_information()
                else:
                    information = information_path[int(t/self.decision_period)]
                new_state = self.state.get_transition(decisions=decision,
                                                    information=information,
                                                    epidemic_function=self.epidemic_function,
                                                    actual_transition=True,
                                                    previous_decision=False,
                                                    old=old)

                if verbose:
                    print(policy_type)
                    print(decision)
                    print(self.is_feasible(decision, verbose=True))

            else:
                try:
                    decision[:,0] = 0  # Turn vaccines to zero
                except UnboundLocalError:
                    decision = np.zeros(shape=(self.num_regions, self.num_intervention_types))
                try:
                    new_state = self.state.get_transition(decisions=decision,
                                                    information=information,
                                                    epidemic_function=self.epidemic_function,
                                                    actual_transition=True,
                                                    previous_decision=True,
                                                    old=old)
                except UnboundLocalError:
                    information = self.get_exogenous_information()
                    new_state = self.state.get_transition(decisions=decision,
                                                    information=information,
                                                    epidemic_function=self.epidemic_function,
                                                    actual_transition=True,
                                                    previous_decision=True,
                                                    old=old)
            cost = self.get_cost(new_state)
            if cost < 0:
                print('###########################################################')
                print('POLICY COST: ', cost)
            self.costs[self.time] = cost
            self.state = new_state
            self.path.append(self.state)
            self.time += 1

        if policy_type == 'ADP':
            for t in range(len(self.costs)-1, -1, -1):
                target = torch.tensor([np.sum(self.costs[t:-1])])
                self.update_value_observations(new_state=self.path[t], target=target, test=True)
            if self.observations.shape[0] > 0:
                self.update_value_function()
        self.state.to_string()
        if val_func_plots:
            self.value_func_prediction_plot(test=True, all=False)
            self.value_func_prediction_plot(test=True, all=True)
        if epidemic_plots:
            self.plot_function()
        if policy_plot:
            allocation = np.array([vaccine_allocation,
                                   disinfectant_allocation,
                                   rehydration_allocation,
                                   antibiotics_allocation]).T
            self.policy_plot(allocation=allocation)

    def naive_policy(self, information_path=None, allocation_plot=True, epidemic_plots=False, policy_plot=False):
        infected = np.zeros((int(self.horizon), self.num_regions))
        self.initialize_policy()

        for i in range(self.num_regions):
            self.case.regions[i].nu = 0.0
            self.case.regions[i].beta = 1.0
            self.case.regions[i].phi = 0
            self.case.regions[i].theta = 0

        self.case.lbar = 0.025
        self.case.l = 0.025
        self.case.update_params()

        cumulative_infected = self.case.get_cumulative_infected(begin=0, end=self.horizon)
        relevant_cumulative_infected = []
        for t in range(int(self.horizon)):
            if t % self.decision_period == 0:
                relevant_cumulative_infected.append(cumulative_infected[int(t/self.case.dt)])
        relevant_cumulative_infected = np.array(relevant_cumulative_infected)
        relevant_weekly_infected = np.diff(relevant_cumulative_infected, axis=0)
        infected = np.concatenate((np.array([relevant_cumulative_infected[0]]), relevant_weekly_infected))

        if allocation_plot:
            x = np.arange(infected.shape[1])
            y = np.arange(infected.shape[0])

            xx, yy = np.meshgrid(x, y)

            z = infected[yy, xx]

            ax = Axes3D(plt.figure())

            ax.set_xlabel('Regions')
            ax.set_ylabel('Decision stage')
            ax.set_zlabel('Symptomatic infected')

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            ax.plot_surface(xx, yy, z, cmap=plt.cm.viridis)
            plt.show()

        decisions = np.zeros((infected.shape[0], infected.shape[1], self.num_intervention_types))
        stacked_infected = np.array([infected.flatten()])

        decision_ratio = normalize(stacked_infected, norm='l1')

        for m in range(self.num_intervention_types):
            stacked_allocation = self.basic_rounding(ratios=decision_ratio.squeeze(), items=self.state.resources[m])
            allocation = np.reshape(stacked_allocation, newshape=(infected.shape[0], self.num_regions))
            decisions[:,:,m] = allocation
        for t in range(int(self.horizon)):
            if self.state.time_dependent_resources.get(str(t)) is not None:
                new_resources = np.array(self.state.time_dependent_resources.get(str(t)))
                tau = int(np.floor(t/self.decision_period))
                new_infected = infected[tau:]
                new_stacked_infected = np.array([new_infected.flatten()])
                new_decision_ratio = normalize(new_stacked_infected, norm='l1')
                for m in range(self.num_intervention_types):
                    stacked_allocation = self.basic_rounding(ratios=new_decision_ratio.squeeze(), items=new_resources[m])
                    allocation = np.reshape(stacked_allocation, newshape=(new_infected.shape[0], self.num_regions))
                    zeros = np.zeros((infected.shape[0]-new_infected.shape[0], self.num_regions))
                    final_allocation = np.concatenate((zeros, allocation))
                    decisions[:,:,m] += final_allocation

        print('Naive')
        print(decisions)

        self.initialize_policy()

        vaccine_allocation = []
        disinfectant_allocation = []
        rehydration_allocation = []
        antibiotics_allocation = []

        week = 0
        for t in range(int(self.horizon)):
            if t % self.decision_period == 0:
                decision = decisions[week]
                aggregated_decision = np.sum(decision, axis=0)
                vaccine_allocation.append(aggregated_decision[0])
                disinfectant_allocation.append(aggregated_decision[1])
                rehydration_allocation.append(aggregated_decision[2])
                antibiotics_allocation.append(aggregated_decision[3])
                if information_path is None:
                    information = self.get_exogenous_information()
                else:
                    information = information_path[int(t/self.decision_period)]
                new_state = self.state.get_transition(decisions=decision,
                                                      information=information,
                                                      epidemic_function=self.epidemic_function,
                                                      previous_decision=False)

                week += 1
            else:
                try:
                    decision[:,0] = 0  # Turn vaccines to zero
                except UnboundLocalError:
                    decision = np.zeros(shape=(self.num_regions, self.num_intervention_types))
                try:
                    new_state = self.state.get_transition(decisions=decision,
                                                    information=information,
                                                    epidemic_function=self.epidemic_function,
                                                    actual_transition=True,
                                                    previous_decision=True)
                except UnboundLocalError:
                    information = self.get_exogenous_information()
                    new_state = self.state.get_transition(decisions=decision,
                                                    information=information,
                                                    epidemic_function=self.epidemic_function,
                                                    actual_transition=True,
                                                    previous_decision=True)
            cost = self.get_cost(new_state)
            self.costs[self.time] = cost
            self.state = new_state
            self.path.append(self.state)
            self.time += 1
        self.state.to_string()

        if epidemic_plots:
            self.plot_function()
        if policy_plot:
            allocation = np.array([vaccine_allocation,
                                   disinfectant_allocation,
                                   rehydration_allocation,
                                   antibiotics_allocation]).T
            self.policy_plot(allocation=allocation)

    def custom_policy(self, epidemic_plots=False):
        self.time = 0
        self.state = self.path[0]
        self.path = [self.state]
        self.costs = np.zeros(int(self.horizon + 1))
        policy_df = pd.read_csv('../data/haiti_custom_policy.csv')
        for i in tqdm(range(int(self.horizon))):
            decision = np.zeros(shape=(self.num_regions,self.num_intervention_types), dtype=int)
            decision_df = policy_df[policy_df['Time'] == i]
            names = ['Ouest', 'Sud-Est', 'Nord', 'Nord-Est', 'Artibonite', 'Centre', 'Sud', 'Grande Anse', 'Nord-Ouest', 'Nippes']
            interventions = ['Vaccine', 'Disinfectant', 'Rehydration', 'Antibiotics']
            for j in range(self.num_regions):
                for m in range(self.num_intervention_types):
                    region_df = decision_df[decision_df['Region'] == names[j]]
                    intervention_df = region_df[region_df['Intervention'] == interventions[m]]
                    dec = intervention_df['Amount'].to_numpy()
                    if dec.shape[0] == 1:
                        decision[j, m] = dec[0]
            information = self.get_exogenous_information()
            new_state = self.state.get_transition(decisions=decision,
                                                  information=information,
                                                  epidemic_function=self.epidemic_function,
                                                  actual_transition=True)
            self.state = new_state
            self.time += 1
        self.state.to_string()
        if epidemic_plots:
            self.plot_function()

    def epsilon_greedy_policy(self, max_solutions=10, max_iters=2, epidemic_plots=False, val_func_plots=False, loss_plot=False):
        for i in tqdm(range(max_iters)):
            if i <= 0.1*max_iters:
                self.epsilon = 1.0
            else:
                self.epsilon = 1/(0.1*i)
            self.initialize_policy()

            standardized_observation = self.scale(self.state.to_tensor())
            self.predictions.append(self.value_function(standardized_observation))
            self.update_loss = []

            for t in range(int(self.horizon)):
                if t % self.decision_period == 0:
                    if i >= 0.95*max_iters:
                        st2 = time.time()
                        decision = self.get_adp_decision()
                        et2 = time.time()
                    else:
                        draw1 = random.random()
                        draw2 = random.random()
                        if draw1 < self.epsilon:
                            decision = self.get_random_decision()
                        elif draw2 < self.epsilon:
                            decision = self.get_greedy_decision()
                        else:
                            st2 = time.time()
                            decision = self.get_adp_decision()
                            et2 = time.time()
                    information = self.get_exogenous_information()
                    new_state = self.state.get_transition(decisions=decision,
                                                          information=information,
                                                          epidemic_function=self.epidemic_function,
                                                          actual_transition=True,
                                                          previous_decision=False)
                else:
                    decision[:, 0] = 0  # Turn vaccines to zero
                    new_state = self.state.get_transition(decisions=decision,
                                                          information=information,
                                                          epidemic_function=self.epidemic_function,
                                                          actual_transition=True,
                                                          previous_decision=True)
                cost = self.get_cost(new_state)
                self.costs[self.time] = cost
                self.state = new_state
                self.path.append(self.state)
                self.time += 1
            for t in range(len(self.costs)-1, -1, -1):
                target = torch.tensor([np.sum(self.costs[t:-1])])
                self.update_value_observations(new_state=self.path[t], target=target, test=False)
            if self.observations.shape[0] > 0:
                self.update_value_function()
                self.iteration_loss.append(np.mean(np.array(self.update_loss)))
        if val_func_plots:
            self.value_func_prediction_plot(test=False)
        if epidemic_plots:
            self.plot_function()
        if loss_plot:
            self.plot_loss()

    def policy_plot(self, allocation):
        tspan = range(0, int(allocation.shape[0]))

        fig, axs = plt.subplots(4, 1)
        axs[0].bar(tspan, allocation[:,0], label='Vaccines')
        axs[0].set_title('Vaccines')

        axs[1].bar(tspan, allocation[:,1], label='Disinfectant')
        axs[1].set_title('Disinfectant')

        axs[2].bar(tspan, allocation[:,2], label='Rehydration')
        axs[2].set_title('Rehydration')

        axs[3].bar(tspan, allocation[:,3], label='Antibiotics')
        axs[3].set_title('Antibiotics')

        fig.text(0.5, 0.005, 'Time [Weeks]', ha='center')
        fig.text(0.005, 0.5, 'Number of resource allocated', va='center', rotation='vertical')

        plt.show()

    def plot_loss(self):
        fig, ax = plt.subplots()
        predicted_plot = ax.plot(self.predictions, c='blue', label='Predicted fatalities')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Predicted fatalities')

        ax2 = ax.twinx()
        loss_plot = ax2.plot(self.iteration_loss, c='red', label='Loss')
        ax2.set_ylabel('Average update loss')

        plots = predicted_plot + loss_plot
        labels = [plot.get_label() for plot in plots]
        ax.legend(plots, labels)
        plt.show()

        plt.plot(self.loss)
        plt.xlabel('Updates')
        plt.ylabel('Loss')
        plt.show()

    def scale(self, data):
        if not self.load:
            self.mean = self.observations.mean(dim=0, keepdim=True)
            self.std = self.observations.std(dim=0, keepdim=True)
        standardized_data = (data - self.mean) / self.std
        standardized_data[standardized_data == float('inf')] = 0.0
        standardized_data = torch.where(torch.isnan(standardized_data),
                                                torch.zeros_like(standardized_data), standardized_data)
        return standardized_data

    def evaluate_decision(self, decision, greedy=False):
        state_value = 0
        for i in range(self.num_scenarios):
            new_state = self.state.get_transition(decisions=decision,
                                                  information=self.scenarios[i],
                                                  epidemic_function=self.epidemic_function,
                                                  actual_transition=False)
            prob = self.scenario_probs[i]
            cost = self.get_cost(new_state)

            standardized_observation = self.scale(new_state.to_tensor())
            val_approx = 0 if greedy else self.value_function(standardized_observation)
            state_value += prob * (cost + val_approx)
        return state_value

    def get_initial_decision(self):
        initial_decision = np.zeros(shape=(self.num_regions,self.num_intervention_types))
        available_resources = np.zeros(shape=self.num_intervention_types)
        ceiled_demand = np.ceil(self.state.demands/self.aggregation_constant)
        decision_ratio = normalize(ceiled_demand, norm='l1',axis=0)
        aggregated_demand = np.sum(ceiled_demand, axis=0)

        available_resources[0] = min(aggregated_demand[0], self.state.resources[0])
        available_resources[1] = min(aggregated_demand[0], self.state.resources[1])

        available_resources[2] = min(aggregated_demand[1], self.state.resources[2])
        available_resources[3] = min(aggregated_demand[1], self.state.resources[3])

        for m in range(self.num_intervention_types):
            initial_decision[:,m] = self.basic_rounding(ratios=decision_ratio[:,1], items=available_resources[m])
        return initial_decision

    @staticmethod
    def basic_rounding(ratios, items, epsilon=1e-6):
        allocation = np.zeros(len(ratios))
        items_allocated = 0
        for i in range(len(ratios)):
            prev_items_allocated = items_allocated
            items_allocated += ratios[i]*items
            allocation[i] = int(items_allocated+epsilon) - int(prev_items_allocated+epsilon)

        # Adjust for numerical instability
        allocation += epsilon
        allocation = allocation.astype(int)
        if items != np.sum(allocation):
            print('WRONG AMOUNT ALLOCATED!')
            print('Ratios: ')
            print(ratios)
            print('Allocation: ')
            print(allocation)
            print('Items: ', items)
            print('Items allocated: ',items_allocated)
            print()
        return allocation

    def get_random_decision(self, verbose=False):
        infeasible = True
        attempts = 0
        while infeasible:
            decision = np.zeros(shape=(self.num_regions,self.num_intervention_types), dtype=int)
            low_array = np.zeros(shape=(self.state.resources.shape))
            available_resources = np.zeros(shape=self.num_intervention_types)

            aggregated_demand = np.ceil(np.sum(self.state.demands, axis=0) / self.aggregation_constant)

            available_resources[0] = min(aggregated_demand[0], self.state.resources[0])
            available_resources[1] = min(aggregated_demand[0], self.state.resources[1])

            available_resources[2] = min(aggregated_demand[1], self.state.resources[2])
            available_resources[3] = min(aggregated_demand[1], self.state.resources[3])

            resources_available = np.random.randint(low=low_array,
                                                    high=available_resources + 1,
                                                    size=available_resources.shape)
            ix = np.arange(self.num_regions)
            np.random.shuffle(ix)
            for m in range(self.num_intervention_types):
                for k in range(ix.size-1):
                    if m == 0 or m == 1:
                        high = min(resources_available[m]+1, self.state.demands[ix[k],0]+1)
                    else:
                        high = min(resources_available[m]+1, self.state.demands[ix[k],1]+1)
                    decision[ix[k],m] = np.random.randint(low=0, high=high)
                    resources_available[m] -= decision[ix[k],m]
                decision[ix[ix.size-1],m] = resources_available[m]
            if self.is_feasible(resource_allocation=decision):
                if verbose:
                    return decision, attempts
                return decision
            attempts += 1

    def highest_descent(self, decision, delta, step_size=1, allow_ascent=False):
        start_time = time.time()
        neg_marginal_cost, pos_marginal_cost = self.calculate_marginal_costs(initial_decision=decision, greedy=False)
        neg_marginal_cost[decision <= 0] = np.inf
        neg_ix = np.unravel_index(np.argmin(neg_marginal_cost, axis=None), neg_marginal_cost.shape)
        pos_ix = np.unravel_index(np.argmin(pos_marginal_cost, axis=None), pos_marginal_cost.shape)

        current_value = self.evaluate_decision(decision=decision)
        print('INITIAL DESCENT DECISION VALUE: ', current_value)

        if allow_ascent:
            tabu_ix = []
            best_ix = pos_ix if pos_marginal_cost[pos_ix] < neg_marginal_cost[neg_ix] else neg_ix
            while neg_marginal_cost[best_ix] < delta or pos_marginal_cost[best_ix] < delta:
                feasible = True
                if pos_marginal_cost[best_ix] < neg_marginal_cost[best_ix]:
                    decision[best_ix] += step_size
                    feasible = self.is_feasible(resource_allocation=decision)
                    if not feasible:
                        decision[best_ix] -= step_size
                        tabu_ix.append(best_ix)
                else:
                    decision[best_ix] -= step_size
                decision[decision < 0.0] = 0.0

                neg_marginal_cost, pos_marginal_cost = self.calculate_marginal_costs(initial_decision=decision)
                for ix in tabu_ix:
                    pos_marginal_cost[ix] = np.inf
                #if not feasible:
                #    pos_marginal_cost[best_ix] = np.inf
                neg_marginal_cost[decision <= 0] = np.inf
                neg_ix = np.unravel_index(np.argmin(neg_marginal_cost, axis=None), neg_marginal_cost.shape)
                pos_ix = np.unravel_index(np.argmin(pos_marginal_cost, axis=None), pos_marginal_cost.shape)
                best_ix = pos_ix if pos_marginal_cost[pos_ix] < neg_marginal_cost[neg_ix] else neg_ix

                current_value = self.evaluate_decision(decision=decision)
                print('UPDATED DESCENT DECISION VALUE: ', current_value)
                if pos_marginal_cost[best_ix] < neg_marginal_cost[best_ix]:
                    print('ASCENT UPDATE: ',best_ix)
        else:
            while neg_marginal_cost[neg_ix] < delta:
                decision[neg_ix] -= step_size
                decision[decision < 0.0] = 0.0

                neg_marginal_cost, pos_marginal_cost = self.calculate_marginal_costs(initial_decision=decision)
                neg_marginal_cost[decision <= 0] = np.inf
                neg_ix = np.unravel_index(np.argmin(neg_marginal_cost, axis=None), neg_marginal_cost.shape)

                current_value = self.evaluate_decision(decision=decision)
                print('UPDATED DESCENT DECISION VALUE: ', current_value)
        print('Highest descent time: ', time.time() - start_time)
        return decision

    def feasibility_descent(self, decision, step_size=1):
        st = time.time()
        if self.is_feasible(resource_allocation=decision):
            best_decision = decision
        else:
            while not self.is_feasible(resource_allocation=decision):
                print('infeasible')
                print(decision)
                if not self.is_resource_feasible(decision=decision):
                    print('RESOURCE INFEASIBLE')
                if not self.is_facility_and_personnel_feasible(resource_allocation=decision):
                    print('FACILITY INFEASIBLE')
                neg_marginal_cost, pos_marginal_cost = self.calculate_marginal_costs(initial_decision=decision)
                neg_marginal_cost[decision <= 0] = np.inf
                neg_ix = np.unravel_index(np.argmin(neg_marginal_cost, axis=None), neg_marginal_cost.shape)
                decision[neg_ix] -= step_size
                decision[decision < 0.0] = 0.0
            best_decision = decision
        return best_decision

    def two_opt_swap(self, decision, delta, step_size=1):
        st = time.time()
        prev_i = None

        neg_marginal_cost, pos_marginal_cost = self.calculate_marginal_costs(initial_decision=decision)
        neg_marginal_cost[decision <= 0] = np.inf

        i, j, m = self.best_mc_swap_indices(pos_marginal_cost=pos_marginal_cost, neg_marginal_cost=neg_marginal_cost)

        count = 0

        current_value = self.evaluate_decision(decision=decision)
        print('INITIAL 2-OPT DECISION VALUE: ', current_value)

        while pos_marginal_cost[i, m] + neg_marginal_cost[j, m] < delta:
            if i == j:
                pos_marginal_cost[i, m] = np.inf
                i, j, m = self.best_mc_swap_indices(pos_marginal_cost=pos_marginal_cost,
                                                    neg_marginal_cost=neg_marginal_cost)
            else:
                swap_size = min(step_size, decision[j, m])
                decision[i, m] += swap_size
                decision[j, m] -= swap_size
                if self.is_feasible(decision):
                    neg_marginal_cost, pos_marginal_cost = self.calculate_marginal_costs(
                        initial_decision=decision)
                    neg_marginal_cost[decision <= 0] = np.inf
                    count += 1
                    print('Current i, j, m: ', i, j, m)
                    if prev_i is not None:
                        print('Previous i, j, m: ', prev_i, prev_j, prev_m)
                    current_value = self.evaluate_decision(decision=decision)
                    print('UPDATED 2-OPT DECISION VALUE: ', current_value)
                else:
                    decision[i, m] -= swap_size
                    decision[j, m] += swap_size
                    pos_marginal_cost[i, m] = np.inf

                prev_i, prev_j, prev_m = i, j, m
                i, j, m = self.best_mc_swap_indices(pos_marginal_cost=pos_marginal_cost,
                                                    neg_marginal_cost=neg_marginal_cost)
                # Avoid immediate swap back
                if (i, j, m) == (prev_j, prev_i, prev_m):
                    print('Swapping back')
                    break

        best_decision = decision
        print('2-opt time: ', time.time() - st)
        return best_decision

    def local_search(self, decision, delta=(-1000), step_size=1):
        decision = self.highest_descent(decision=decision, delta=delta, step_size=step_size, allow_ascent=self.demand_relaxation)
        decision = self.feasibility_descent(decision=decision, step_size=step_size)
        decision = self.two_opt_swap(decision=decision, delta=delta, step_size=step_size)
        return decision

    def save_value_function(self, path='../models/trained_vfa_models/model_v6.pth'):
        torch.save(self.value_function, path)
        print('SAVED SCALER')
        print(type(self.mean))
        print(type(self.std))
        print(self.mean)
        print(self.std)
        with open('../models/trained_vfa_models/scaler_v6.txt', 'w') as f:
            f.write(str(self.mean.numpy()))
            f.write(str(self.std.numpy()))

    def load_value_function(self, path):
        directory = '../models/trained_vfa_models/'
        vfa_path = directory + 'model_' + path +'.pth'
        scaler_path = directory + 'scaler_' + path + '.txt'
        self.value_function = torch.load(vfa_path)
        self.value_function.eval()
        self.load = True
        file = open(scaler_path)
        lines = file.readlines()
        array = ''
        for line in lines:
            if ']][[' in line:
                line1, line2 = line.split('][')
                line1 += ']'
                line2 = '[' + line2
                array += line1
                array = array[1:]
                array = array[1:]
                array = array[:len(array)]
                array = array[:len(array)]
                self.mean = torch.from_numpy(np.array([np.fromstring(array, sep=' ')]))
                array = line2
            else:
                array += line
        array = array[1:]
        array = array[1:]
        array = array[:len(array)]
        array = array[:len(array)]
        self.std = torch.from_numpy(np.array([np.fromstring(array, sep=' ')]))
        print('LOADED SCALER')
        print(type(self.mean))
        print(type(self.std))
        print(self.mean)
        print(self.std)

    @staticmethod
    def best_mc_swap_indices(pos_marginal_cost, neg_marginal_cost):
        ixs = np.argmin(pos_marginal_cost, axis=0)
        jxs = np.argmin(neg_marginal_cost, axis=0)

        best_pos_mc = np.take_along_axis(pos_marginal_cost, np.array([ixs]), axis=0)
        best_neg_mc = np.take_along_axis(neg_marginal_cost, np.array([jxs]), axis=0)

        best_mc_swap = best_neg_mc + best_pos_mc

        m = np.argmin(best_mc_swap)

        i, j = ixs[m], jxs[m]

        return i, j, m

    def is_feasible(self, resource_allocation, verbose=False):
        resource_feasible = self.is_resource_feasible(decision=resource_allocation)
        facility_and_personnel_feasible = self.is_facility_and_personnel_feasible(resource_allocation=resource_allocation, verbose=verbose)
        feasible = resource_feasible and facility_and_personnel_feasible
        return feasible

    def is_resource_feasible(self, decision):
        agg_decision = np.sum(decision, axis=0)
        available_resources_feasible = (agg_decision <= self.state.resources).all()
        non_negative_resources_feasible = (agg_decision >= np.zeros(shape=agg_decision.shape)).all()
        resource_feasible = available_resources_feasible and non_negative_resources_feasible
        return resource_feasible

    def is_demand_feasible(self, decision):
        demands = np.copy(self.state.demands)
        ceiled_demand = np.ceil(demands/self.aggregation_constant)

        vaccine_feasible = (decision[:,0] <= ceiled_demand[:,0]).all()
        disinfectant_feasible = (decision[:,1] <= ceiled_demand[:,0]).all()

        rehydration_feasible = (decision[:,2] <= ceiled_demand[:,1]).all()
        antibiotics_feasible = (decision[:,3] <= ceiled_demand[:,1]).all()

        demand_feasible = vaccine_feasible and disinfectant_feasible and rehydration_feasible and antibiotics_feasible
        return demand_feasible


    def is_facility_and_personnel_feasible(self, resource_allocation, verbose=False):
        model = cp_model.CpModel()

        resource_allocation = resource_allocation.astype(int)

        resource_allocation *= self.aggregation_constant

        facilities_open = {}
        personnel_allocation = {}

        # Define variables

        for i in range(self.num_regions):
            personnel_allocation[i] = model.NewIntVar(0, self.total_personnel, 'z_t%ii%i' % (self.time, i))
            for n in range(self.num_facility_types):
                facilities_open[(i, n)] = model.NewIntVar(0, int(self.locations_available[i, n]),
                                                          'y_t%ii%in%i' % (self.time, i, n))
        # Define constraints
        t0 = time.time()
        for m in range(self.num_intervention_types):

            for i in range(self.num_regions):

                intervention_type_capacity = sum(
                    int(self.facility_intervention_capacity[n, m]) * facilities_open[(i, n)] for n in
                    range(self.num_facility_types))
                model.Add(intervention_type_capacity >= resource_allocation[i, m])

        for i in range(self.num_regions):
            deployment_capacity = sum(
                int(self.intervention_personnel_capacity[m]) * resource_allocation[i, m] for m in
                range(self.num_intervention_types))
            model.Add( personnel_allocation[i] >= deployment_capacity)

            operational_capacity = sum(
                self.facility_personnel_capacity[n] * facilities_open[(i, n)] for n in range(self.num_facility_types))
            model.Add(operational_capacity <= personnel_allocation[i])

            for n in range(self.num_facility_types):
                model.Add(facilities_open[(i, n)] <= self.locations_available[i, n])

        total_personnel_allocated = sum(personnel_allocation[i] for i in range(self.num_regions))
        model.Add(total_personnel_allocated <= self.total_personnel)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if verbose:
            if status == cp_model.FEASIBLE:
                facilities_decision = np.zeros(shape=(self.num_regions, self.num_facility_types))
                personnel_decision = np.zeros(shape=(self.num_regions))
                for i in range(self.num_regions):
                    personnel_decision[i] = solver.Value(personnel_allocation[i])
                    for n in range(self.num_facility_types):
                        facilities_decision[i,n] = solver.Value(facilities_open[(i,n)])
                print('Facilities open: ')
                print(facilities_decision)
                print('Personnel allocation: ')
                print(personnel_decision)
            else:
                print('Infeasible decision')


        return status == cp_model.FEASIBLE

    def calculate_marginal_costs(self, initial_decision, greedy=False):
        current_decision = initial_decision

        start_time = time.time()
        current_value = self.evaluate_decision(decision=initial_decision, greedy=greedy)
        end_time = time.time()

        neg_marginal_cost = np.zeros(shape=(self.num_regions,self.num_intervention_types))
        pos_marginal_cost = np.zeros(shape=(self.num_regions,self.num_intervention_types))
        for i in range(self.num_regions):
            for m in range(self.num_intervention_types):
                if current_decision[i,m] > 0:
                    neg_neighbor_decision = np.array(current_decision)

                    neg_neighbor_decision[i,m] -= 1

                    neg_marginal_cost[i,m] = self.evaluate_decision(decision=neg_neighbor_decision, greedy=greedy) - current_value

                pos_neighbor_decision = np.array(current_decision)
                pos_neighbor_decision[i, m] += 1
                pos_marginal_cost[i, m] = self.evaluate_decision(decision=pos_neighbor_decision, greedy=greedy) - current_value

        return neg_marginal_cost, pos_marginal_cost

    def get_losses(self, all=False, test=True):
        self.value_function.eval()
        if len(self.observations) > 0:
            if all:
                standardized_observations = self.scale(data=self.all_observations)
                targets = self.all_targets.unsqueeze(-1)
            else:
                standardized_observations = self.scale(data=self.observations)
                targets = self.targets.unsqueeze(-1)

            dataset = TensorDataset(standardized_observations, targets)
            predictions = []
            actuals = []
            data_generator = DataLoader(dataset, batch_size=dataset.__len__(), drop_last=True)
            for (observation, target) in data_generator:
                predicted = self.value_function(observation)
                predictions.append(predicted.detach().numpy())
                actuals.append(target.numpy())

        if test:
            test_standardized_observations = self.scale(data=self.test_observations)
            test_dataset = TensorDataset(test_standardized_observations, self.test_targets.unsqueeze(-1))
            test_predictions = []
            test_actuals = []
            test_data_generator = DataLoader(test_dataset, batch_size=test_dataset.__len__(), drop_last=True)
            for (test_observation, test_target) in test_data_generator:
                test_predicted = self.value_function(test_observation)
                test_predictions.append(test_predicted.detach().numpy())
                test_actuals.append(test_target.numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        test_predictions = np.array(test_predictions)
        test_actuals = np.array(test_actuals)

        training_loss = np.square(predictions-actuals)
        test_loss = np.square(test_predictions-test_actuals)
        return training_loss, test_loss

    def value_func_prediction_plot(self, test=False, all=False):
        self.value_function.eval()
        if len(self.observations) > 0:
            if all:
                standardized_observations = self.scale(data=self.all_observations)
                targets = self.all_targets.unsqueeze(-1)
            else:
                standardized_observations = self.scale(data=self.observations)
                targets = self.targets.unsqueeze(-1)
            dataset = TensorDataset(standardized_observations, targets)
            predictions = []
            actuals = []
            data_generator = DataLoader(dataset, batch_size=dataset.__len__(), drop_last=True)
            for (observation, target) in data_generator:
                predicted = self.value_function(observation)
                predictions.append(predicted.detach().numpy())
                actuals.append(target.numpy())

        if test:
            test_standardized_observations = self.scale(data=self.test_observations)
            test_dataset = TensorDataset(test_standardized_observations, self.test_targets.unsqueeze(-1))

            test_predictions = []
            test_actuals = []
            test_data_generator = DataLoader(test_dataset, batch_size=test_dataset.__len__(), drop_last=True)
            for (test_observation, test_target) in test_data_generator:
                test_predicted = self.value_function(test_observation)
                test_predictions.append(test_predicted.detach().numpy())
                test_actuals.append(test_target.numpy())
        fig, ax = plt.subplots()
        if len(self.observations) > 0:
            ax.scatter(actuals, predictions, s=40, c='slateblue', marker='o', alpha=0.4, label='Training')
        if test: ax.scatter(test_actuals, test_predictions, s=40, c='red', marker='o', alpha=0.4, label='Test')

        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]

        ax.plot(lims, lims, 'k-', alpha=0.8, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel('Actual fatalities')
        ax.set_ylabel('Predicted fatalities')

        if test: ax.legend()

        ax.grid()

        plt.show()

# The following class is from the Google OR-Tools website and not developed by the author of this thesis
class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, limit, num_regions, num_intervention_types):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solution_limit = limit
        self.solutions = []
        self.num_regions = num_regions
        self.num_intervention_types = num_intervention_types

    def get_solutions(self):
        return self.solutions

    def on_solution_callback(self):
        solution = np.zeros(shape=(self.num_regions,self.num_intervention_types))
        self.__solution_count += 1

        counter = 0
        i = 0
        m = 0
        for v in self.__variables:
            if counter < self.num_regions*self.num_intervention_types:
                solution[i,m] = self.Value(v)
            counter += 1
            m += 1
            if m % self.num_intervention_types == 0:
                m = 0
                i += 1
        self.solutions.append(solution)

        if self.__solution_count >= self.__solution_limit:
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count