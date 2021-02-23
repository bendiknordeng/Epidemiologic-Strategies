__author__ = "Martin Willoch Olstad"
__email__ = "martinwilloch@gmail.com"

import numpy as np
import json
import time
from cholera_model.Region import Region
from cholera_model.Case import Case
from resource_allocation_model.MarkovDecisionProcess import MarkovDecisionProcess
from resource_allocation_model.State import State
from sklearn.preprocessing import normalize
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from scipy.optimize import fmin


class Instance:

    def __init__(self, max_iters=10, horizon=10, max_regions=3, dt=0.1, reallocation_iters=1, decomposition='stage',
                 parallel=False, decision_period=1.0, hypothetical=False,
                 resource_path=None, demand_relaxation=False):
        self.max_iters = max_iters
        self.horizon = horizon
        self.max_regions = max_regions
        self.dt = dt
        self.reallocation_iters = reallocation_iters

        self.epidemic_params, self.resource_allocation_params, self.case_params = self.load_parameters(hypothetical=hypothetical,
                                                                                                       resource_path=resource_path)
        self.regions = self.initialize_regions()
        self.cases = []
        self.num_regions = len(self.regions)

        self.decomposition = decomposition
        self.parallel = parallel
        self.decision_period = decision_period

        self.dist_matrix = Case.calculate_distance_matrix(regions=self.regions)
        self.mean_dist = self.dist_matrix.mean()

        self.stage_fatalities = []
        self.regional_fatalities = []
        self.regional_reallocation_fatalities = []

        self.demand_relaxation = demand_relaxation

    def run(self, load_path=None):
        if self.decomposition is 'stage':
            self.run_stage_decomposition(load_path=load_path)
        elif self.decomposition is 'regional':
            self.run_regional_decomposition(parallel=self.parallel)
        else:
            print('Please specify decomposition method')

    def run_naive_policy_tests(self, load_path):
        for i in range(10):
            case = self.initialize_case(regions=self.regions)
            mdp = self.initialize_stage_decomposition(case=case)
            mdp.load_value_function(path=load_path)
            information_path = []
            for t in range(int(self.horizon)):
                if t % self.decision_period == 0:
                    information_path.append(mdp.get_exogenous_information())
            st_naive = time.time()
            mdp.naive_policy(epidemic_plots=False, information_path=information_path, policy_plot=True)
            et_naive = time.time()
            naive_fatalities = mdp.state.fatalities

    def run_stage_and_regional_comparison(self, max_regions=10, load_stage_path=None):
        stage_speed = []
        regional_speed = []

        for i in range(1, max_regions+1):
            self.max_regions = i
            self.num_regions = i
            self.regions = self.initialize_regions()
            st_stage = time.time()
            self.run_stage_decomposition(regional_comparison=True, load_path=load_stage_path)
            et_stage = time.time()
            stage_speed.append(et_stage-st_stage)
            print('STAGE DECOMP WITH %d REGIONS' % i)

            st_regional = time.time()
            self.run_regional_decomposition()
            et_regional = time.time()
            regional_speed.append(et_regional-st_regional)
            print('REGIONAL DECOMP WITH %d REGIONS' % i)


        stage_speed = np.array(stage_speed)
        regional_speed = np.array(regional_speed)

        stage_fatalities = np.array(self.stage_fatalities)
        regional_fatalities = np.array(self.regional_fatalities)
        reallocation_fatalities = np.array(self.regional_reallocation_fatalities)
        reallocation_fatalities = np.squeeze(reallocation_fatalities)

        tspan = range(1, max_regions+1)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.set_xlabel('Number of regions')
        ax1.set_ylabel('Computation time [Seconds]')
        ax2.set_ylabel('Cholera-induced fatalities')

        plots = ax1.plot()

        plots += ax1.plot(tspan, stage_speed, label='Stage computation time', c='red')
        plots += ax1.plot(tspan, regional_speed, label='Regional computation time', c='blue')
        plots += ax2.plot(tspan, stage_fatalities, '--', label='Stage fatalities', c='red')
        plots += ax2.plot(tspan, regional_fatalities, '--', label='Regional fatalities', c='blue')

        labels = [plot.get_label() for plot in plots]
        ax1.legend(plots, labels)
        plt.savefig('../figures/speed_plot_v2')
        plt.show()

        print('REALLOCATION ITERS:')
        print(range(self.reallocation_iters))
        print(reallocation_fatalities)
        print(reallocation_fatalities.shape)

        mean_reallocation_fatalities = np.mean(reallocation_fatalities, axis=0)
        upper_perc = np.percentile(reallocation_fatalities, 95, axis=0)
        lower_perc = np.percentile(reallocation_fatalities, 5, axis=0)

        print(mean_reallocation_fatalities)
        print(upper_perc)
        print(lower_perc)

        tspan = range(1, self.reallocation_iters + 1)

        plt.plot(tspan, mean_reallocation_fatalities, c='blue')
        plt.fill_between(tspan, upper_perc, lower_perc, color='blue', alpha=0.2)
        plt.xlabel('Reallocation iterations')
        plt.ylabel('Cholera-induced fatalities')
        plt.savefig('../figures/speed_reallocation_plot_v2')
        plt.show()

        for i in range(self.max_regions):
            plt.plot(range(1, self.reallocation_iters+1), reallocation_fatalities[i])
        plt.xlabel('Reallocation iterations')
        plt.ylabel('Cholera-induced fatalities')
        plt.savefig('../figures/speed_reallocation_plot_v3')
        plt.show()

    def run_dispersal_prob_comparison(self, max_regions=10, load_path='hypothetical_adp_best'):
        old_speed = []
        new_speed = []
        old_pop_diff = []
        new_pop_diff = []
        for i in range(1, max_regions+1):
            print('ACTUALLY DOING THIS, ALREADY ON ITERATION: ', i)
            self.max_regions = i
            case = self.initialize_case(regions=self.regions)
            mdp = self.initialize_stage_decomposition(case=case)
            mdp.load_value_function(path=load_path)
            information_path = []
            for t in range(int(self.horizon)):
                if t % self.decision_period == 0:
                    information_path.append(mdp.get_exogenous_information())

            st_old = time.time()
            mdp.policy(old=True, policy_type='ADP', information_path=information_path, epidemic_plots=False)
            et_old = time.time()
            old_initial_population = 0
            old_end_population = 0
            for j in range(case.number_of_regions):
                old_initial_population += case.regions[j].S[0] + case.regions[j].A[0] + case.regions[j].I[0] + case.regions[j].R[0]
                old_end_population += case.regions[j].S[-1] + case.regions[j].A[-1] + case.regions[j].I[-1] + case.regions[j].R[-1]

            st_new = time.time()
            mdp.policy(old=False, policy_type='ADP', information_path=information_path, epidemic_plots=False)
            et_new = time.time()
            new_initial_population = 0
            new_end_population = 0
            for j in range(case.number_of_regions):
                new_initial_population += case.regions[j].S[0] + case.regions[j].A[0] + case.regions[j].I[0] + case.regions[j].R[0]
                new_end_population += case.regions[j].S[-1] + case.regions[j].A[-1] + case.regions[j].I[-1] + case.regions[j].R[-1]

            old_speed.append(et_old-st_old)
            new_speed.append(et_new-st_new)
            old_pop_diff.append((old_initial_population-old_end_population)/old_initial_population)
            new_pop_diff.append((new_initial_population-new_end_population)/new_initial_population)

        old_speed = np.array(old_speed)
        new_speed = np.array(new_speed)
        old_pop_diff = np.array(old_pop_diff)
        new_pop_diff = np.array(new_pop_diff)

        tspan = range(1, max_regions+1)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.set_xlabel('Number of regions')
        ax1.set_ylabel('Computation time [Seconds]')
        ax2.set_ylabel('Difference in population')

        plots = ax1.plot()

        plots += ax1.plot(tspan, old_speed, label='Old computation time', c='red')
        plots += ax1.plot(tspan, new_speed, label='New computation time', c='blue')

        labels = [plot.get_label() for plot in plots]
        ax1.legend(plots, labels, loc=0)
        plt.show()

    def run_hyperparameter_tuning(self, name, run_lrs=True, lrs=(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0), architectures=([])):
        case = self.initialize_case(regions=self.regions)
        bp_training_losses = []
        bp_test_losses = []
        if run_lrs:
            label_strings = [str(lr) for lr in lrs]
            for lr in lrs:
                mdp = self.initialize_stage_decomposition(case=case)
                mdp.lr = lr
                mdp.epsilon_greedy_policy(max_iters=self.max_iters, loss_plot=True, val_func_plots=False, epidemic_plots=False)
                mdp.policy(policy_type='ADP', verbose=True, epidemic_plots=False, val_func_plots=True, policy_plot=True)
                training_losses, test_losses = mdp.get_losses()
                bp_training_losses.append(training_losses.squeeze())
                bp_test_losses.append(test_losses.squeeze())
        else:
            label_strings = [str(hidden_dims) for hidden_dims in architectures]
            for hidden_dims in architectures:
                mdp = self.initialize_stage_decomposition(case=case, hidden_dims=hidden_dims)

                mdp.epsilon_greedy_policy(max_iters=self.max_iters, loss_plot=True, val_func_plots=False,
                                          epidemic_plots=False)
                mdp.policy(policy_type='ADP', verbose=True, epidemic_plots=False, val_func_plots=True, policy_plot=True)
                training_losses, test_losses = mdp.get_losses()

                bp_training_losses.append(training_losses.squeeze())
                bp_test_losses.append(test_losses.squeeze())

        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)

        box_train = plt.boxplot(bp_training_losses, widths=0.6, sym='')
        train_color = 'blue'
        set_box_color(box_train, train_color)

        if run_lrs:
            plt.xlabel('Learning rates')
        else:
            plt.xlabel('Architectures')
        plt.ylabel('Loss [MSE]')

        ticks = [dim for dim in label_strings]

        plt.xticks(range(1, len(ticks)+1), ticks)
        plt.tight_layout()
        plt.savefig('../figures/final_v4_hyp_box_plot_training_%s' % name)
        plt.show()

        box_test = plt.boxplot(bp_test_losses, widths=0.6, sym='')
        test_color = 'red'
        set_box_color(box_test, test_color)
        if run_lrs:
            plt.xlabel('Learning rates')
        else:
            plt.xlabel('Architectures')
        plt.ylabel('Loss [MSE]')

        ticks = [dim for dim in label_strings]

        plt.xticks(range(1, len(ticks)+1), ticks)
        plt.tight_layout()
        plt.savefig('../figures/final_v4_hyp_box_test_plot_%s' % name)
        plt.show()

    def run_policy_comparison(self, iterations=2, load_path=None, confidence=0.95, name='', elems=None, probs=None):
        case = self.initialize_case(regions=self.regions)
        mdp = self.initialize_stage_decomposition(case=case)

        if elems is not None and probs is not None:
            mdp.scenarios = elems
            mdp.scenario_probs = probs
            mdp.num_scenarios = len(mdp.scenarios)

        adp_cumulative_fatalities = []
        greedy_cumulative_fatalities = []
        naive_cumulative_fatalities = []
        myopic_cumulative_fatalities = []

        adp_infections = []
        greedy_infections = []
        naive_infections = []
        myopic_infections = []

        if load_path is None:
            mdp.epsilon_greedy_policy(max_iters=self.max_iters, loss_plot=True, val_func_plots=False, epidemic_plots=False)
        else:
            print('Loading model...')
            print('Default model:')
            print(list(mdp.value_function.parameters()))
            mdp.load_value_function(path=load_path)
            print('Loaded model:')
            print(list(mdp.value_function.parameters()))

        for i in range(iterations):
            information_path = []
            for t in range(int(self.horizon)):
                if t % self.decision_period == 0:
                    information_path.append(mdp.get_exogenous_information())

            mdp.policy(policy_type='ADP', information_path=information_path, verbose=True, epidemic_plots=False, val_func_plots=False, policy_plot=False)
            fatalities = np.zeros(int(self.horizon/self.dt)+1)
            infections = np.zeros(int(self.horizon/self.dt)+1)
            for k in range(self.num_regions):
                fatalities += case.regions[k].M
                infections += case.regions[k].I
            adp_cumulative_fatalities.append(fatalities)
            adp_infections.append(infections)

            mdp.policy(policy_type='Greedy', information_path=information_path,  verbose=True, epidemic_plots=False, policy_plot=False)
            fatalities = np.zeros(int(self.horizon/self.dt)+1)
            infections = np.zeros(int(self.horizon/self.dt)+1)
            for k in range(self.num_regions):
                fatalities += case.regions[k].M
                infections += case.regions[k].I
            greedy_cumulative_fatalities.append(fatalities)
            greedy_infections.append(infections)

            mdp.naive_policy(information_path=information_path, epidemic_plots=False, allocation_plot=False)
            fatalities = np.zeros(int(self.horizon/self.dt)+1)
            infections = np.zeros(int(self.horizon/self.dt)+1)
            for k in range(self.num_regions):
                fatalities += case.regions[k].M
                infections += case.regions[k].I
            naive_cumulative_fatalities.append(fatalities)
            naive_infections.append(infections)

            mdp.policy(policy_type='Myopic',information_path=information_path,  verbose=True, epidemic_plots=False, policy_plot=False)
            fatalities = np.zeros(int(self.horizon/self.dt)+1)
            infections = np.zeros(int(self.horizon/self.dt)+1)
            for k in range(self.num_regions):
                fatalities += case.regions[k].M
                infections += case.regions[k].I
            myopic_cumulative_fatalities.append(fatalities)
            myopic_infections.append(infections)

        adp_cumulative_fatalities = np.array(adp_cumulative_fatalities)
        greedy_cumulative_fatalities = np.array(greedy_cumulative_fatalities)
        naive_cumulative_fatalities = np.array(naive_cumulative_fatalities)
        myopic_cumulative_fatalities = np.array(myopic_cumulative_fatalities)

        mean_adp_cumulative_fatalities = np.mean(adp_cumulative_fatalities, axis=0)
        mean_greedy_cumulative_fatalities = np.mean(greedy_cumulative_fatalities, axis=0)
        mean_naive_cumulative_fatalities = np.mean(naive_cumulative_fatalities, axis=0)
        mean_myopic_cumulative_fatalities = np.mean(myopic_cumulative_fatalities, axis=0)

        std_adp_cumulative_fatalities = np.std(adp_cumulative_fatalities, axis=0)
        std_greedy_cumulative_fatalities = np.std(greedy_cumulative_fatalities, axis=0)
        std_naive_cumulative_fatalities = np.std(naive_cumulative_fatalities, axis=0)
        std_myopic_cumulative_fatalities = np.std(myopic_cumulative_fatalities, axis=0)

        adp_infections = np.array(adp_infections)
        greedy_infections = np.array(greedy_infections)
        naive_infections = np.array(naive_infections)
        myopic_infections = np.array(myopic_infections)

        mean_adp_infections = np.mean(adp_infections, axis=0)
        mean_greedy_infections = np.mean(greedy_infections, axis=0)
        mean_naive_infections = np.mean(naive_infections, axis=0)
        mean_myopic_infections = np.mean(myopic_infections, axis=0)

        print('ADP CUMULATIVE FATALITIES: ')
        print(adp_cumulative_fatalities.shape)
        print(adp_cumulative_fatalities)
        print(adp_cumulative_fatalities[-1])
        print(adp_cumulative_fatalities[:,-1])

        plt.hist(adp_cumulative_fatalities[:,-1], color='b')
        plt.xlabel('Total cholera-induced fatalities')
        plt.ylabel('Frequency')
        plt.show()
        plt.hist(greedy_cumulative_fatalities[:,-1], color='r')
        plt.xlabel('Total cholera-induced fatalities')
        plt.ylabel('Frequency')
        plt.show()
        plt.hist(naive_cumulative_fatalities[:,-1], color='g')
        plt.xlabel('Total cholera-induced fatalities')
        plt.ylabel('Frequency')
        plt.show()
        plt.hist(myopic_cumulative_fatalities[:,-1], color='y')
        plt.xlabel('Total cholera-induced fatalities')
        plt.ylabel('Frequency')
        plt.show()


        print('SIMPLY FINDING THE BEST!')
        print(adp_cumulative_fatalities.shape)
        print(adp_cumulative_fatalities)
        print(adp_cumulative_fatalities[:,-1].shape)
        print(adp_cumulative_fatalities[:,-1])

        adp_best_ix = np.argmin(adp_cumulative_fatalities[:,-1])
        adp_worst_ix = np.argmax(adp_cumulative_fatalities[:,-1])
        adp_best = adp_cumulative_fatalities[adp_best_ix]
        adp_worst = adp_cumulative_fatalities[adp_worst_ix]

        greedy_best_ix = np.argmin(greedy_cumulative_fatalities[:,-1])
        greedy_worst_ix = np.argmax(greedy_cumulative_fatalities[:,-1])
        greedy_best = greedy_cumulative_fatalities[greedy_best_ix]
        greedy_worst = greedy_cumulative_fatalities[greedy_worst_ix]

        naive_best_ix = np.argmin(naive_cumulative_fatalities[:,-1])
        naive_worst_ix = np.argmax(naive_cumulative_fatalities[:,-1])
        naive_best = naive_cumulative_fatalities[naive_best_ix]
        naive_worst = naive_cumulative_fatalities[naive_worst_ix]

        myopic_best_ix = np.argmin(myopic_cumulative_fatalities[:,-1])
        myopic_worst_ix = np.argmax(myopic_cumulative_fatalities[:,-1])
        myopic_best = myopic_cumulative_fatalities[myopic_best_ix]
        myopic_worst = myopic_cumulative_fatalities[myopic_worst_ix]

        alpha = 1-(1-confidence)/2
        critical_value = norm.ppf(alpha)

        adp_upper_pred_interval = mean_adp_cumulative_fatalities+critical_value*std_adp_cumulative_fatalities
        adp_lower_pred_interval = mean_adp_cumulative_fatalities-critical_value*std_adp_cumulative_fatalities

        greedy_upper_pred_interval = mean_greedy_cumulative_fatalities+critical_value*std_greedy_cumulative_fatalities
        greedy_lower_pred_interval = mean_greedy_cumulative_fatalities-critical_value*std_greedy_cumulative_fatalities

        naive_upper_pred_interval = mean_naive_cumulative_fatalities+critical_value*std_naive_cumulative_fatalities
        naive_lower_pred_interval = mean_naive_cumulative_fatalities-critical_value*std_naive_cumulative_fatalities

        myopic_upper_pred_interval = mean_myopic_cumulative_fatalities + critical_value * std_myopic_cumulative_fatalities
        myopic_lower_pred_interval = mean_myopic_cumulative_fatalities - critical_value * std_myopic_cumulative_fatalities

        tspan = np.arange(mean_adp_cumulative_fatalities.size)
        tspan = np.linspace(0, int(mean_adp_infections.size*self.dt), mean_adp_infections.size)

        plt.plot(tspan, mean_adp_cumulative_fatalities, c='blue', label='ADP')
        plt.fill_between(tspan, adp_worst, adp_best, color='blue', alpha=0.2)
        plt.plot(tspan, mean_greedy_cumulative_fatalities, c='red', label='Greedy')
        plt.fill_between(tspan, greedy_worst, greedy_best, color='red', alpha=0.2)
        plt.plot(tspan, mean_naive_cumulative_fatalities, c='green', label='Naive')
        plt.fill_between(tspan, naive_worst, naive_best, color='green', alpha=0.2)
        plt.plot(tspan, mean_myopic_cumulative_fatalities, c='orange', label='Myopic')
        plt.fill_between(tspan, myopic_worst, myopic_best, color='orange', alpha=0.2)

        plt.xlabel('Time [Days]')
        plt.ylabel('Cumulative fatalities')
        plt.legend()
        plt.savefig('../figures/%s_comparison_best_worst' % name)
        plt.show()

        adp_upper_perc = np.percentile(adp_cumulative_fatalities, 95, axis=0)
        adp_lower_perc = np.percentile(adp_cumulative_fatalities, 5, axis=0)

        greedy_upper_perc = np.percentile(greedy_cumulative_fatalities, 95, axis=0)
        greedy_lower_perc = np.percentile(greedy_cumulative_fatalities, 5, axis=0)

        naive_upper_perc = np.percentile(naive_cumulative_fatalities, 95, axis=0)
        naive_lower_perc = np.percentile(naive_cumulative_fatalities, 5, axis=0)

        myopic_upper_perc = np.percentile(myopic_cumulative_fatalities, 95, axis=0)
        myopic_lower_perc = np.percentile(myopic_cumulative_fatalities, 5, axis=0)

        plt.plot(tspan, mean_adp_cumulative_fatalities, c='blue', label='ADP')
        plt.fill_between(tspan, adp_upper_perc, adp_lower_perc, color='blue', alpha=0.2)
        plt.plot(tspan, mean_greedy_cumulative_fatalities, c='red', label='Greedy')
        plt.fill_between(tspan, greedy_upper_perc, greedy_lower_perc, color='red', alpha=0.2)
        plt.plot(tspan, mean_naive_cumulative_fatalities, c='green', label='Naive')
        plt.fill_between(tspan, naive_upper_perc, naive_lower_perc, color='green', alpha=0.2)
        plt.plot(tspan, mean_myopic_cumulative_fatalities, c='orange', label='Myopic')
        plt.fill_between(tspan, myopic_upper_perc, myopic_lower_perc, color='orange', alpha=0.2)

        plt.xlabel('Time [Days]')
        plt.ylabel('Cumulative fatalities')
        plt.legend()
        plt.savefig('../figures/%s_comparison_percentile' % name)
        plt.show()

        print()
        print('Mean ADP: ', mean_adp_cumulative_fatalities[-1])
        print('Best ADP: ', adp_best[-1])
        print('Worst ADP: ', adp_worst[-1])
        print()
        print('Mean greedy: ', mean_greedy_cumulative_fatalities[-1])
        print('Best greedy: ', greedy_best[-1])
        print('Worst greedy: ', greedy_worst[-1])
        print()
        print('Mean naive: ', mean_naive_cumulative_fatalities[-1])
        print('Best naive: ', naive_best[-1])
        print('Worst naive: ', naive_worst[-1])
        print()
        print('Mean myopic: ', mean_myopic_cumulative_fatalities[-1])
        print('Best myopic: ', myopic_best[-1])
        print('Worst myopic: ', myopic_worst[-1])
        print()

        plt.plot(tspan, mean_adp_cumulative_fatalities, c='blue', label='ADP')
        plt.fill_between(tspan, adp_worst, adp_best, color='blue', alpha=0.2)
        plt.plot(tspan, mean_greedy_cumulative_fatalities, c='red', label='Greedy')
        plt.fill_between(tspan, greedy_worst, greedy_best, color='red', alpha=0.2)
        plt.plot(tspan, mean_naive_cumulative_fatalities, c='green', label='Naive')
        plt.fill_between(tspan, naive_worst, naive_best, color='green', alpha=0.2)

        plt.xlabel('Time [Days]')
        plt.ylabel('Cumulative fatalities')
        plt.legend()
        plt.savefig('../figures/%s_comparison_best_worst_no_myopic' % name)
        plt.show()

        plt.plot(tspan, mean_adp_cumulative_fatalities, c='blue', label='ADP')
        plt.fill_between(tspan, adp_upper_perc, adp_lower_perc, color='blue', alpha=0.2)
        plt.plot(tspan, mean_greedy_cumulative_fatalities, c='red', label='Greedy')
        plt.fill_between(tspan, greedy_upper_perc, greedy_lower_perc, color='red', alpha=0.2)
        plt.plot(tspan, mean_naive_cumulative_fatalities, c='green', label='Naive')
        plt.fill_between(tspan, naive_upper_perc, naive_lower_perc, color='green', alpha=0.2)

        plt.xlabel('Time [Days]')
        plt.ylabel('Cumulative fatalities')
        plt.legend()
        plt.savefig('../figures/%s_comparison_percentile_no_myopic' % name)
        plt.show()

    def run_stage_decomposition(self, load_path=None, regional_comparison=False):
        case = self.initialize_case(regions=self.regions)
        mdp = self.initialize_stage_decomposition(case=case)
        #mdp.num_regions = self.num_regions

        st_eg = time.time()
        if load_path is None:
            mdp.epsilon_greedy_policy(max_iters=self.max_iters, loss_plot=True, val_func_plots=False, epidemic_plots=False)
        else:
            print('Loading model...')
            mdp.load_value_function(path=load_path)
        et_eg = time.time()

        information_path = []
        for t in range(int(self.horizon)):
            if t % self.decision_period == 0:
                information_path.append(mdp.get_exogenous_information())

        st_adp = time.time()
        mdp.policy(policy_type='ADP', information_path=information_path, verbose=True, epidemic_plots=False, val_func_plots=False, policy_plot=True)
        et_adp = time.time()
        adp_fatalities = mdp.state.fatalities

        # For stage and regional decomposition comparison
        self.stage_fatalities.append(adp_fatalities)
        if not regional_comparison:
            st_naive = time.time()
            mdp.naive_policy(epidemic_plots=False, information_path=information_path, policy_plot=False)
            et_naive = time.time()
            naive_fatalities = mdp.state.fatalities

            st_myopic = time.time()
            mdp.policy(policy_type='Myopic', information_path=information_path, verbose=True, epidemic_plots=False, val_func_plots=False, policy_plot=False)
            et_myopic = time.time()
            myopic_fatalities = mdp.state.fatalities

            st_greedy = time.time()
            mdp.policy(policy_type='Greedy', information_path=information_path, verbose=True, epidemic_plots=False, policy_plot=True)
            et_greedy = time.time()
            greedy_fatalities = mdp.state.fatalities

            st_random = time.time()
            mdp.policy(policy_type='Random', epidemic_plots=False)
            et_random = time.time()
            random_fatalities = mdp.state.fatalities

            st_nothing = time.time()
            mdp.policy(policy_type='Nothing', information_path=information_path, epidemic_plots=False)
            et_nothing = time.time()
            nothing_fatalities = mdp.state.fatalities

            print('Information path: ')
            print(information_path)
            print('Epsilon greedy time: ',et_eg-st_eg)

            print('ADP fatalities: ',adp_fatalities)
            print('ADP time: ',et_adp-st_adp)
            print()
            print('Naive fatalities: ',naive_fatalities)
            print('Naive time: ',et_naive-st_naive)
            print()
            print('Myopic fatalities: ',myopic_fatalities)
            print('Myopic time: ',et_myopic-st_myopic)
            print()
            print('Greedy fatalities: ',greedy_fatalities)
            print('Greedy time: ',et_greedy-st_greedy)
            print()
            print('Random fatalities: ',random_fatalities)
            print('Random time: ',et_random-st_random)
            print()
            print('Nothing fatalities: ',nothing_fatalities)
            print('Nothing time: ',et_nothing-st_nothing)
            print()
        if load_path is None:
            save_path = 'trained_vfa_models/model_v6.pth'
            mdp.save_value_function(path=save_path)

    def run_regional_decomposition(self, parallel=False, max_regions=10):
        st1 = time.time()
        reallocation_fatalities_list = []
        for k in range(self.reallocation_iters):
            if k == 0:
                resource_allocation, personnel_allocation = self.fatality_based_allocation()
            else:
                resource_allocation, personnel_allocation = self.marginal_benefit_based_allocation(mdps=self.mdps,
                                                                                                   prev_resource_allocation=resource_allocation,
                                                                                                   prev_personnel_allocation=personnel_allocation)
            self.mdps = self.initialize_regional_decomposition(resource_allocation, personnel_allocation)
            self.set_dispersal()

            if parallel:
                processes = []
                for i in range(self.num_regions):
                    process = multiprocessing.Process(target=self.multiprocessing_mdp_training, args=(i, self.mdps, self.max_iters))
                    processes.append(process)
                    process.start()
                for process in processes:
                    process.join()
            else:
                reallocation_fatalities = 0
                for i in range(self.num_regions):
                    self.mdps[i].epsilon_greedy_policy(max_iters=self.max_iters, loss_plot=False, val_func_plots=False)
                    reallocation_fatalities += self.mdps[i].state.fatalities
                reallocation_fatalities_list.append(reallocation_fatalities)
        self.regional_reallocation_fatalities.append([reallocation_fatalities_list])
        et1 = time.time()

        adp_fatalities = 0
        greedy_fatalities = 0

        self.set_dispersal()

        st2 = time.time()
        if parallel:
            processes = []
            for i in range(self.num_regions):
                process = multiprocessing.Process(target=self.multiprocessing_policy, args=(i, self.mdps, False))
                adp_fatalities += self.mdps[i].state.fatalities
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
        else:
            for i in range(self.num_regions):
                print('REGION: %d' % i)
                self.mdps[i].policy(policy_type='ADP', verbose=True, epidemic_plots=False,
                           val_func_plots=False, policy_plot=False)
                adp_fatalities += self.mdps[i].state.fatalities

        et2 = time.time()

        print('Training time: ',et1-st1)
        print('Solving time: ',et2-st2)

        self.regional_fatalities.append(adp_fatalities)

        print('Regional fatalities: ',adp_fatalities)

    def set_dispersal(self):
        prob_matrix = np.zeros((self.num_regions, self.num_regions))
        counter = 0

        dispersal = np.zeros((int(self.horizon), self.num_regions))

        for t in range(int(self.horizon)):

            water_concentration = np.array([self.regions[j].B[t] * self.regions[j].W for j in range(self.num_regions)])
            for i in range(self.num_regions):
                softmax_denominator = 0
                for k in range(self.num_regions):
                    if i != k:
                        term1 = self.regions[i].S[t] + self.regions[i].A[t] + self.regions[i].I[t] + self.regions[i].R[
                            t]
                        # term1 = Y[0 + comps * k] + Y[1 + comps * k] + Y[2 + comps * k] + Y[3 + comps * k]
                        term2 = np.exp(-self.dist_matrix[i, k] / self.mean_dist)
                        softmax_denominator += term1 * term2
                        counter += 1
                for j in range(self.num_regions):
                    if i != j:
                        prob_matrix[i, j] = (self.regions[i].S[t] + self.regions[i].A[t] + self.regions[i].I[t] +
                                             self.regions[i].R[t]) * np.exp(
                            -self.dist_matrix[i, j] / self.mean_dist) / softmax_denominator
            dispersal[t] = np.dot(water_concentration, prob_matrix)
        for i in range(len(self.cases)):
            self.mdps[i].state.dispersal = dispersal[:,i]

    @staticmethod
    def multiprocessing_mdp_training(index, mdps, max_iters):
        mdps[index].epsilon_greedy_policy(max_iters=max_iters, loss_plot=False, val_func_plots=False)

    @staticmethod
    def multiprocessing_policy(index, mdps, greedy=False):
        if greedy:
            mdps[index].greedy_policy()
        else:
            mdps[index].adp_policy()

    def fatality_based_allocation(self):
        fatalities = np.zeros(self.num_regions)
        for i in range(self.num_regions):
            case = Case(regions=[self.regions[i]], dt=self.dt, params=self.epidemic_params, horizon=self.horizon,
                        aggregation_constant=self.resource_allocation_params['aggregation_constant'])
            case.simulate_epidemic(begin=0, end=self.horizon)
            fatalities[i] = case.regions[0].M[-1]
        allocation_ratio = normalize(fatalities.reshape(-1,1), norm='l1',axis=0).T[0]
        resource_allocation = np.zeros((self.num_regions,self.resource_allocation_params['num_intervention_types']))
        for m in range(self.resource_allocation_params['num_intervention_types']):
            resource_allocation[:,m] = MarkovDecisionProcess.fair_rounding(ratios=allocation_ratio, items=self.resource_allocation_params['resources_available'][m])
        personnel_allocation = MarkovDecisionProcess.fair_rounding(ratios=allocation_ratio, items=self.resource_allocation_params['total_personnel'])
        return resource_allocation, personnel_allocation

    def marginal_benefit_based_allocation(self, mdps, prev_resource_allocation, prev_personnel_allocation, step_size=1):
        marginal_benefit = np.zeros((self.num_regions,self.resource_allocation_params['num_intervention_types']))
        for i in range(self.num_regions):
            for m in range(self.resource_allocation_params['num_intervention_types']):
                marginal_benefit[i,m] = prev_resource_allocation[i,m]/mdps[i].path[-1].fatalities

        worst_marginal_benefit = np.copy(marginal_benefit)
        worst_marginal_benefit[worst_marginal_benefit < step_size] = np.inf
        worst_ix = np.argmin(worst_marginal_benefit, axis=0)
        best_ix = np.argmax(marginal_benefit, axis=0)

        print('PREV RESOURCE ALLOC')
        print(prev_resource_allocation)
        print('PREV PERSONNEL ALLOC')
        print(prev_personnel_allocation)

        for m in range(self.resource_allocation_params['num_intervention_types']):
            prev_resource_allocation[worst_ix[m],m] -= step_size
            prev_resource_allocation[best_ix[m],m] += step_size
            personnel_step_size = step_size*(prev_resource_allocation[worst_ix[m],m]/(prev_resource_allocation[worst_ix[m],m]+step_size))
            prev_personnel_allocation[worst_ix[m]] -= personnel_step_size
            prev_personnel_allocation[best_ix[m]] += personnel_step_size

        print('NEW RESOURCE ALLOC')
        print(prev_resource_allocation)
        print('NEW PERSONNEL ALLOC')
        print(prev_personnel_allocation)
        print()

        return prev_resource_allocation, prev_personnel_allocation

    def initialize_regional_decomposition(self, resource_allocation, personnel_allocation):
        mdps = []
        self.cases = []
        for i in range(self.num_regions):
            ra_params = dict(self.resource_allocation_params)
            ra_params['num_regions'] = 1
            ra_params['locations_available'] = np.array([self.resource_allocation_params['locations_available'][i]])
            ra_params['initial_demands'] = self.resource_allocation_params['initial_demands'][i]
            ra_params['total_personnel'] = int(personnel_allocation[i])

            case = Case(regions=[self.regions[i]], dt=self.dt, params=self.epidemic_params, horizon=self.horizon,
                        aggregation_constant=ra_params['aggregation_constant'], regional=True)
            self.cases.append(case)
            state = State(resources=resource_allocation[i],
                          demands=np.array([ra_params['initial_demands']]),
                          fatalities=0)
            mdp = MarkovDecisionProcess(state=state, horizon=self.horizon,
                                        epidemic_function=case.transition,
                                        plot_function=case.plot_all, map_plot_function=case.map_plot,
                                        params=ra_params,
                                        cumulative_function=case.get_cumulative_infected,
                                        hidden_dims=None, case=case
                                        )
            mdps.append(mdp)
        return mdps

    def initialize_regions(self):
        regions = []
        for index, row in self.case_params.iterrows():
            if index < self.max_regions:
                regions.append(
                    Region(name=row.governorate, id=row.id, longitude=row.longitude, latitude=row.latitude, A0=row.A0,
                         I0=row.I0,
                         R0=row.R0,
                         B0=row.B0, N0=row.N0, M0=row.M0, dt=self.dt,
                         horizon=self.horizon, nu0=row.nu0, beta0=row.beta0, phi0=row.phi0, theta0=row.theta0))
        return regions

    def initialize_case(self, regions):
        case = Case(regions=regions, dt=self.dt, params=self.epidemic_params, horizon=self.horizon,
                        aggregation_constant=self.resource_allocation_params['aggregation_constant'])
        self.cases.append(case)
        return case

    def initialize_stage_decomposition(self, case, hidden_dims=None):
        initial_state = State(resources=self.resource_allocation_params['resources_available'],
                              demands=np.array(self.resource_allocation_params['initial_demands']), fatalities=0,
                              time_dependent_resources=self.resource_allocation_params['time_dependent_resources'])
        mdp = MarkovDecisionProcess(state=initial_state, horizon=self.horizon, epidemic_function=case.transition,
                                    plot_function=case.plot_all, map_plot_function=case.map_plot,
                                    params=self.resource_allocation_params,
                                    cumulative_function=case.get_cumulative_infected,
                                    hidden_dims=None, case=case,
                                    demand_relaxation=self.demand_relaxation)
        return mdp

    def load_parameters(self, hypothetical=False, resource_path=None):
        #case_path = '../data/yemen_parameters.csv'
        #case_params = pd.read_csv(case_path, delimiter=',')
        if hypothetical:
            case_path = 'mwo/data/hypothetical_haiti_parameters.csv'
        else:
            case_path = 'mwo/data/haiti_parameters.csv'
        case_params = pd.read_csv(case_path, delimiter=',')

        if hypothetical:
            with open('mwo/data/hypothetical_epidemic_parameters.json', 'r') as f:
                epidemic_params = json.load(f)
        else:
            with open('mwo/data/epidemic_parameters.json', 'r') as f:
                epidemic_params = json.load(f)

        if resource_path is None:
            resource_path = 'mwo/data/haiti_resource_allocation_parameters.json'

        with open(resource_path, 'r') as f:
            resource_allocation_params = json.load(f)

        initial_demands = []
        locations_available = []
        for index, row in case_params.iterrows():
            if index < self.max_regions:
                S0 = row.N0 - row.A0 - row.I0 - row.R0
                I0 = row.I0
                initial_demands.append([S0, I0])
                locations_available.append([row.ctc_locs, row.ctu_locs, row.orp_locs])

        resource_allocation_params['num_regions'] = len(initial_demands)
        resource_allocation_params['num_facility_types'] = len(resource_allocation_params['facility_types'])
        resource_allocation_params['num_intervention_types'] = len(resource_allocation_params['intervention_types'])
        resource_allocation_params['locations_available'] = np.array(locations_available)

        resource_allocation_params['resources_available'] = np.array(resource_allocation_params['resources_available'])
        resource_allocation_params['facility_intervention_capacity'] = np.array(
            resource_allocation_params['facility_intervention_capacity'])
        resource_allocation_params['intervention_personnel_capacity'] = np.array(
            resource_allocation_params['intervention_personnel_capacity'])
        resource_allocation_params['facility_personnel_capacity'] = np.array(
            resource_allocation_params['facility_personnel_capacity'])
        resource_allocation_params['initial_demands'] = initial_demands
        return epidemic_params, resource_allocation_params, case_params

    @staticmethod
    def pseudo_uniform_test(sims=1, plot=False):
        num_regions = 2
        num_intervention_types = 1
        items = 5
        resources = np.array([items])
        capacity = 3

        counter = np.zeros(shape=(items + 1, items + 1))

        final_decisions = []
        final_attempts = 0

        for iteration in range(sims):
            attempts = 0
            infeasible = True
            while infeasible:
                decision = np.zeros(shape=(num_regions, num_intervention_types))
                low_array = np.zeros(shape=resources.shape)
                resources_available = np.random.randint(low=low_array,
                                                        high=resources + 1,
                                                        size=resources.shape)
                ix = np.arange(num_regions)
                np.random.shuffle(ix)
                for m in range(num_intervention_types):
                    for k in range(ix.size - 1):
                        decision[ix[k], m] = np.random.randint(low=0, high=resources_available[m] + 1)
                        resources_available[m] -= decision[ix[k], m]
                    decision[ix[ix.size - 1], m] = resources_available[m]
                    if ((decision <= capacity) & (decision >= 0)).all():
                        r1 = int(decision[0, 0])
                        r2 = int(decision[1, 0])
                        print(r1)
                        print(r2)
                        counter[r1, r2] += 1
                        print(counter)
                        decision_string = np.array2string(decision.T[0]) \
                            .replace('[', '') \
                            .replace(']', '') \
                            .replace(' ', '') \
                            .replace('.', '')
                        final_decisions.append(decision_string)
                        final_attempts += attempts
                        infeasible = False
                attempts += 1
        print('Average attempts: ', final_attempts / sims)
        if plot:
            x = np.linspace(0, items, items + 1)
            y = np.linspace(0, items, items + 1)

            xx, yy = np.meshgrid(x, y)

            z = xx * 0 + yy * 0 + counter / sims

            ax = Axes3D(plt.figure())

            ax.set_xlabel('Allocated resources to region 1')
            ax.set_ylabel('Allocated resources to region 2')
            ax.set_zlabel('Relative frequency of decision')

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            ax.plot_surface(xx, yy, z, cmap=plt.cm.viridis)
            plt.show()

    def plot_all(self):
        case = self.initialize_case(regions=self.regions)
        case.simulate_epidemic(end=self.horizon)
        case.plot_all()

    def cumulative_plots(self):
        path = '../data/haiti_cumulative_infected.csv'
        df = pd.read_csv(path, delimiter=',')

        case = self.initialize_case(regions=self.regions)

        for i in range(case.number_of_regions):
            case.regions[i].beta = 0.90
            case.regions[i].phi = 1.0

        case.update_params()

        cumulative_infected = case.get_cumulative_infected(begin=0.0, end=self.horizon)
        for i in range(self.num_regions):
            case.regions[i].C = cumulative_infected[:, i]

        haiti_infected = np.sum(cumulative_infected, axis=1)
        artibonite_infected = case.regions[4].C
        centre_infected = case.regions[5].C
        nord_infected = case.regions[2].C
        nord_ouest_infected = case.regions[8].C
        ouest_infected = case.regions[0].C

        dates = [datetime.datetime.strptime(d, "%m/%d/%Y").date() for d in df['Date'].astype(str)]

        #start_date = datetime.date(year=2010, month=10, day=28)
        start_date = datetime.date(year=2010, month=10, day=21)
        end_date = start_date + datetime.timedelta(days=int(self.horizon+1))

        tspan = np.linspace(0, self.horizon, num=int(1 + (self.horizon / self.dt)))
        times = (np.arange(0, self.horizon+1)/self.dt).astype('int64')
        tspan = pd.date_range(start_date, end_date, periods=tspan.size).to_pydatetime()
        tspan_date = pd.date_range(start_date, end_date, periods=tspan.size)

        fc = 'red'
        s = 40
        ec = 'red'
        c = 'blue'
        alpha = 0.4

        plt.scatter(dates, df['Total']/1000, label='Actual', s=s, facecolors=fc, alpha=alpha, edgecolors=ec)
        plt.xlabel('Time')
        plt.ylabel('Cumulative symptomatic infections [Thousands]')
        plt.plot(tspan, haiti_infected/1000, label='Estimated', c=c)
        plt.title('Haiti')
        plt.plot([], [], '-o', c=u'#ff7f0e', label='Haiti')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        fig.text(0.47, 0.04, 'Time', ha='center', va='center')
        fig.text(0.02, 0.5, 'Cumulative symptomatic cases [Thousands]', ha='center', va='center', rotation='vertical')
        ax1.title.set_text('Artibonite')
        ax2.title.set_text('Centre')
        ax3.title.set_text('Nord')
        ax4.title.set_text('Nord-Ouest')

        ax1.scatter(dates, df['Artibonite']/1000, alpha=alpha, facecolors=fc, s=s, edgecolors=ec)
        ax1.plot(tspan, artibonite_infected/1000, c=c)
        ax2.scatter(dates, df['Centre']/1000, alpha=alpha, label='Actual', facecolors=fc, s=s, edgecolors=ec)
        ax2.plot(tspan, centre_infected/1000, label='Estimated', c=c)
        ax3.scatter(dates, df['Nord']/1000,alpha=alpha, facecolors=fc, s=s, edgecolors=ec)
        ax3.plot(tspan, nord_infected/1000, c=c)
        ax4.scatter(dates, df['Nord-Ouest']/1000,alpha=alpha, facecolors=fc, s=s, edgecolors=ec)
        ax4.plot(tspan, nord_ouest_infected/1000, c=c)
        ax2.legend()
        fig.autofmt_xdate()
        plt.show()

        case.map_plot(z='Relative cumulative symptomatic infections')
        case.map_plot(z='Cumulative symptomatic infections')
        case.map_plot(z='At least one symptomatic cholera case')

    def nonlinear_least_squares(self, param_guess=(1.3e8, 1.3e11, 0.03, 100)):
        path = '../data/haiti_cumulative_infected.csv'
        df = pd.read_csv(path, delimiter=',')
        case = self.initialize_case(regions=self.regions)

        for i in range(case.number_of_regions):
            case.regions[i].beta = 0.9
            case.regions[i].phi = 1.0
        case.update_params()

        data = []
        data.append(df['Total'][df['Total'].notnull()].to_numpy())
        data.append(df['Artibonite'][df['Artibonite'].notnull()].to_numpy())
        data.append(df['Centre'][df['Centre'].notnull()].to_numpy())
        data.append(df['Nord'][df['Nord'].notnull()].to_numpy())
        data.append(df['Nord-Ouest'][df['Nord-Ouest'].notnull()].to_numpy())

        ixs = []

        indices = np.array([18, 28, 35, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140])
        indices = indices*(1/self.dt)
        indices = indices.astype('int64')
        ixs.append(indices)

        indices = np.array([18, 28, 35, 49, 56, 63, 70, 77])
        indices = indices*(1/self.dt)
        indices = indices.astype('int64')
        ixs.append(indices)

        indices = np.array([28, 49, 56, 63, 70])
        indices = indices*(1/self.dt)
        indices = indices.astype('int64')
        ixs.append(indices)

        indices = np.array([28, 35, 63, 70, 77])
        indices = indices*(1/self.dt)
        indices = indices.astype('int64')
        ixs.append(indices)

        indices = np.array([35, 49, 56, 63, 70, 77])
        indices = indices*(1/self.dt)
        indices = indices.astype('int64')
        ixs.append(indices)

        params = fmin(self.calibration_score, x0=param_guess, args=(data, case, ixs))
        print(params)

    @staticmethod
    def calibration_score(params, data, case, indices):
        case.regions[4].B[0], case.regions[5].B[0], case.regions[0].B[0], case.regions[2].B[0], case.regions[8].B[0] = params

        case.update_params()
        forecast = case.get_cumulative_infected()
        haiti_forecast = np.sum(forecast, axis=1)
        artibonite_forecast = forecast[:, 4]
        centre_forecast = forecast[:, 5]
        nord_forecast = forecast[:, 2]
        nord_ouest_forecast = forecast[:, 8]

        haiti_forecast = haiti_forecast[indices[0]]
        artibonite_forecast = artibonite_forecast[indices[1]]
        centre_forecast = centre_forecast[indices[2]]
        nord_forecast = nord_forecast[indices[3]]
        nord_ouest_forecast = nord_ouest_forecast[indices[4]]

        sse = 0

        diff = data[0].ravel() - haiti_forecast.ravel()
        sse += np.dot(diff, diff)

        diff = data[1].ravel() - artibonite_forecast.ravel()
        sse += np.dot(diff, diff)

        diff = data[2].ravel() - centre_forecast.ravel()
        sse += np.dot(diff, diff)

        diff = data[3].ravel() - nord_forecast.ravel()
        sse += np.dot(diff, diff)

        diff = data[4].ravel() - nord_ouest_forecast.ravel()
        sse += np.dot(diff, diff)

        return sse

    @staticmethod
    def sse(array1, array2):
        diff = array1.ravel() - array2.ravel()
        sse = np.dot(diff, diff)
        return sse


