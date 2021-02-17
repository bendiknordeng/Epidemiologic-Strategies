__author__ = "Martin Willoch Olstad"
__email__ = "martinwilloch@gmail.com"

import time
from Instance import Instance
import sys

if __name__ == '__main__':
    sys.stdout = open('output.txt', 'w')

    #computational_study = 'Model Training'
    #computational_study = 'Value Function Tuning and Convergence'
    #computational_study = 'Base Case'
    #computational_study = 'Kit Size'
    #computational_study = 'Early Vaccines'
    #computational_study = 'Additional Vaccines'
    #computational_study = 'Additional ORS'
    #computational_study = 'Dispersal Distributions'
    #computational_study = 'Long Horizon'
    #computational_study = 'Epidemic Model Plots'

    if computational_study == 'Model Training':
        st = time.time()
        hypothetical = False
        max_iters = 500
        horizon = 120.0
        decision_period = 7.0
        dt = 0.10
        decomposition = 'stage'
        max_regions = 10
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical)
        instance.run()

    if computational_study == 'Value Function Tuning and Convergence':
        st = time.time()
        hypothetical = False
        horizon = 120.0
        decision_period = 7.0
        dt = 0.10
        max_iters = 200
        max_regions = 10
        decomposition = 'stage'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical)
        instance.run_hyperparameter_tuning(name='lr', lrs=(0.0001, 0.001, 0.01, 0.1, 1.0), run_lrs=True)
        instance.run_hyperparameter_tuning(name='hidden_dims', architectures=([], [10], [20, 10], [30, 20, 10], [40, 30, 20, 10]), run_lrs=False)
        et = time.time()
        print('Run time: ',et-st)

    if computational_study == 'Base Case':
        hypothetical = False
        load_path = '500_iters_base_case'
        iterations = 100
        horizon = 120.0
        decision_period = 7.0
        dt = 0.10
        max_regions = 10
        max_iters = 500
        decomposition = 'stage'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(iterations=iterations, confidence=0.95, load_path=load_path, name='base_case')

    if computational_study == 'Kit Size':
        hypothetical = False
        load_path = '500_iters_base_case'
        iterations = 100
        horizon = 120.0
        decision_period = 7.0
        dt = 0.10
        max_regions = 10
        max_iters = 500
        decomposition = 'stage'

        resource_path = '../data/haiti_resource_allocation_parameters_100_kit_size.json'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(iterations=iterations, confidence=0.95, load_path=load_path, name='100_kit_size')

        resource_path = '../data/haiti_resource_allocation_parameters_500_kit_size.json'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(iterations=iterations, confidence=0.95, load_path=load_path, name='500_kit_size')

    if computational_study == 'Early Vaccines':
        hypothetical = True
        load_path = 'hypothetical_adp_best'
        iterations = 100
        horizon = 120.0
        decision_period = 7.0
        dt = 0.10
        max_regions = 10
        max_iters = 500
        decomposition = 'stage'

        st = time.time()
        resource_path = '../data/haiti_resource_allocation_parameters_one_week_vaccines.json'
        decomposition = 'stage'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(iterations=iterations, confidence=0.95, load_path=load_path,
                                       name='one_week_vaccines')
        print('ONE WEEK VACCINES')
        print('Time: ', time.time() - st)

        st = time.time()
        resource_path = '../data/haiti_resource_allocation_parameters_early_vaccines.json'
        decomposition = 'stage'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(iterations=iterations, confidence=0.95, load_path=load_path,
                                       name='immediate_vaccines')
        print('IMMEDIATE VACCINES')
        print('Time: ', time.time() - st)

    if computational_study == 'Additional Vaccines':
        hypothetical = True
        load_path = 'hypothetical_adp_best'
        iterations = 100
        horizon = 120.0
        decision_period = 7.0
        dt = 0.10
        max_regions = 10
        max_iters = 500
        decomposition = 'stage'

        st = time.time()
        resource_path = '../data/haiti_resource_allocation_parameters_600_vaccines.json'
        decomposition = 'stage'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(iterations=iterations, confidence=0.95, load_path=load_path,
                                       name='600_vaccines')
        print('600 VACCINES')
        print('Time: ', time.time() - st)

        st = time.time()
        resource_path = '../data/haiti_resource_allocation_parameters_800_vaccines.json'
        decomposition = 'stage'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(iterations=iterations, confidence=0.95, load_path=load_path,
                                       name='800_vaccines')
        print('800 VACCINES')
        print('Time: ', time.time() - st)

        st = time.time()
        resource_path = '../data/haiti_resource_allocation_parameters_1000_vaccines.json'
        decomposition = 'stage'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(iterations=iterations, confidence=0.95, load_path=load_path,
                                       name='1000_vaccines')
        print('1000 VACCINES')
        print('Time: ', time.time() - st)

    if computational_study == 'Additional ORS':
        hypothetical = True
        load_path = 'hypothetical_adp_best'
        iterations = 100
        horizon = 120.0
        decision_period = 7.0
        dt = 0.10
        max_regions = 10
        max_iters = 500
        decomposition = 'stage'

        st = time.time()
        resource_path = '../data/haiti_resource_allocation_parameters_300_ors.json'
        decomposition = 'stage'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(iterations=iterations, confidence=0.95, load_path=load_path,
                                       name='300_ors')
        print('300 ORS')
        print('Time: ', time.time() - st)

        st = time.time()
        resource_path = '../data/haiti_resource_allocation_parameters_400_ors.json'
        decomposition = 'stage'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(iterations=iterations, confidence=0.95, load_path=load_path,
                                       name='400_ors')
        print('400 ORS')
        print('Time: ', time.time() - st)

    if computational_study == 'Dispersal Distributions':
        hypothetical = True
        load_path = 'hypothetical_adp_best'
        iterations = 1
        horizon = 120.0
        decision_period = 7.0
        dt = 0.10
        max_regions = 10
        max_iters = 500
        decomposition = 'stage'
        resource_path = '../data/haiti_resource_allocation_parameters.json'

        st = time.time()
        elems = [0.025]
        probs = [1.0]
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(elems=elems, probs=probs, iterations=1, confidence=0.95, load_path=load_path,
                                       name='calibrated_value')
        print('CALIBRATED VALUE')
        print('Time: ', time.time() - st)

        st = time.time()
        elems = [0.0, 0.025, 0.25]
        probs = [0.10, 0.80, 0.10]
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(elems=elems, probs=probs, iterations=iterations, confidence=0.95, load_path=load_path,
                                       name='low_variance')
        print('LOW VARIANCE')
        print('Time: ', time.time() - st)

        st = time.time()
        elems = [0.0, 0.025, 0.25]
        probs = [1.0/3.0, 1.0/3.0, 1.0/3.0]
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(elems=elems, probs=probs, iterations=iterations, confidence=0.95, load_path=load_path,
                                       name='high_variance')
        print('HIGH VARIANCE')
        print('Time: ', time.time() - st)

        st = time.time()
        elems = [0.0, 0.025, 0.050]
        probs = [0.25, 0.50, 0.25]
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(elems=elems, probs=probs, iterations=iterations, confidence=0.95, load_path=load_path,
                                       name='low_max_dispersal')
        print('LOW MAX DISPERSAL')
        print('Time: ', time.time() - st)

    if computational_study == 'Long Horizon':
        hypothetical = True
        load_path = 'hypothetical_adp_best'
        iterations = 100
        horizon = 150.0
        dt = 0.10
        max_regions = 10
        max_iters = 500
        decision_period = 7.0
        decomposition = 'stage'

        st = time.time()
        resource_path = '../data/haiti_resource_allocation_parameters.json'
        decomposition = 'stage'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(iterations=iterations, confidence=0.95, load_path=load_path,
                                       name='final_run_150_horizon')
        print('150 HORIZON')
        print('Time: ', time.time() - st)

        hypothetical = True
        load_path = 'hypothetical_adp_best'
        horizon = 180.0
        dt = 0.10
        max_regions = 10
        decomposition = 'stage'

        st = time.time()
        resource_path = '../data/haiti_resource_allocation_parameters.json'
        decomposition = 'stage'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical,
                            resource_path=resource_path)
        instance.initialize_case(regions=instance.regions)
        instance.run_policy_comparison(iterations=iterations, confidence=0.95, load_path=load_path,
                                       name='final_run_180_horizon')
        print('180 HORIZON')
        print('Time: ', time.time() - st)

    if computational_study == 'Epidemic Model Plots':
        hypothetical = False
        horizon = 150.0
        decision_period = 7.0
        dt = 0.10
        max_regions = 10
        max_iters = 500
        decomposition = 'stage'
        instance = Instance(max_iters=max_iters, horizon=horizon, max_regions=max_regions, dt=dt,
                            decomposition=decomposition,
                            decision_period=decision_period,
                            hypothetical=hypothetical)
        instance.initialize_case(regions=instance.regions)
        instance.cumulative_plots()

    sys.stdout.close()

