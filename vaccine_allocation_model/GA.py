import numpy as np
import scipy
from utils import tcolors, write_pickle, calculate_yll
import pandas as pd
import os
from copy import copy
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from datetime import datetime
import json
from itertools import combinations

class SimpleGeneticAlgorithm:
    def __init__(self, simulations, population_size, process, objective, min_generations, 
                random_individuals, expected_years_remaining, verbose, individuals_from_file=None):
        """ Initializes a simple genetic algorithm instance

        Args:
            simulations (int): number specifying number of simulations
            population_size (int): number of individuals to initialize population with
            process (MarkovDecisionProcess): a process to simulate fitness of individuals
            objective (str): choice of metric to evaluate fitness
            verbose (bool): specify whether or not to print
        """
        self.simulations = simulations
        self.process = process
        if individuals_from_file is None:
            self.population = Population(population_size, verbose, random_individuals)
            self.generation_count = 0 if individuals_from_file is None else individuals_from_file[0]
            self.final_scores = defaultdict(partial(defaultdict, list))
        else:
            self.population = Population(population_size, verbose, random_individuals, individuals_from_file[1])
            self.final_scores = individuals_from_file[2]
            self.generation_count = individuals_from_file[0]
            Individual.GENERATION = individuals_from_file[0]
        self.best_individual = None
        self.generations_since_new_best = 0
        self.expected_years_remaining = expected_years_remaining
        self.objective = objective
        self.random_individuals = random_individuals
        self.min_generations = min_generations
        self.number_of_runs = []
        self._generate_output_dirs()
        self.verbose = verbose

    def get_objective(self, objective):
        return {"deaths": lambda process: np.sum(process.state.D),
                "infected": lambda process: np.sum(process.state.total_infected),
                "weighted": lambda process: np.sum(process.state.total_infected)*0.01 + np.sum(process.state.D),
                "yll": lambda process: calculate_yll(self.expected_years_remaining, process.state.D.sum(axis=0))
                }[objective]

    def run(self):
        """ Function to evaluate current generation, create offsprings, evaluate offsprings, and generate new generations if not converged """
        while True:
            self.run_population()
            self.reset_final_scores()
            self.write_to_file()
            if self.check_convergence():
                break
            self.crossover(self.generation_count)
            self.repair_offsprings()
            self.mutation()
            self.repair_offsprings()
            self.run_population(offsprings=True)
            self.population.new_generation(self.generation_count, self.min_generations)
            self.generation_count += 1
            self.reset_final_scores(new_generation=True)
            
    def run_population(self, offsprings=False):
        """ For current population, simulate

        Args:
            offsprings (bool, optional): if population to be run is offsprings of a generation. Defaults to False.

        Returns:
            bool: True if the two best scores of the population also is significant best
        """
        if self.verbose: print(f"\n\n{tcolors.OKBLUE}Running{' offsprings of ' if offsprings else ' '}generation {self.generation_count}{tcolors.ENDC}")
        self.find_fitness(offsprings)
        count = 0
        significant_best = self.find_best_individual(offsprings)
        while not significant_best and count <= 2:
            if self.verbose: print("Running more simulations...")
            self.find_fitness(offsprings, from_start=False)
            significant_best = self.find_best_individual(offsprings, from_start=False)
            count += 1

        if self.verbose:
            if significant_best:
                print(f"{tcolors.OKGREEN}Significant best {'offspring' if offsprings else 'individual'} found{tcolors.ENDC}")
                if offsprings:
                    print(f"Best offspring of generation {self.generation_count}: {self.population.offsprings[0]}")
                else:
                    print(f"Best individual of generation {self.generation_count}: {self.population.individuals[0]}")
            else:
                if offsprings:
                    print(f"{tcolors.FAIL}Significant best {'offspring' if offsprings else 'individual'} not found.{tcolors.ENDC}")
                    print(f"\n{tcolors.OKGREEN}{self.population.offsprings[0]} added to population.{tcolors.ENDC}")
                else:
                    print(f"{tcolors.FAIL}Significant best {'offspring' if offsprings else 'individual'} not found.{tcolors.ENDC}")

        return significant_best

    def check_convergence(self):
        """ Checks if the best individual of the current population is better than the previous best

        Returns:
            bool: True if new all-time best individual is found
        """
        candidate = self.population.individuals[0]
        if self.best_individual is None:
            print(f"{tcolors.OKGREEN}Setting all-time best individual: {candidate}{tcolors.ENDC}")
            self.best_individual = candidate
        else:
            if candidate == self.best_individual:
                print(f"{tcolors.WARNING}{candidate} already all-time best. Continuing...{tcolors.ENDC}")
                write_pickle(self.best_individual_path+str(self.generation_count)+".pkl", self.best_individual)
                self.generations_since_new_best += 1
                if self.generations_since_new_best > 2 and self.generation_count > self.min_generations:
                    print(f"{tcolors.OKGREEN}Converged. Best individual: {self.best_individual.ID}{tcolors.ENDC}")
                    return True
                return False
            if self.verbose: print(f"{tcolors.HEADER}Testing {candidate} against all-time high {self.best_individual}{tcolors.ENDC}")
            self.population.offsprings = [candidate, self.best_individual]
            new_best = self.find_best_individual(offsprings=True, convergence_test=True)
            count = 0
            while not new_best and count <= 5:
                self.find_fitness(offsprings=True, from_start=False, convergence_test=True)
                new_best = self.find_best_individual(offsprings=True, convergence_test=True)
                count += 1
            if new_best:
                if self.verbose: print(f"{tcolors.OKGREEN}New all-time best: {candidate}{tcolors.ENDC}")
                self.best_individual = candidate
                self.generations_since_new_best = 0
            else:
                if self.verbose: print(f"{tcolors.FAIL}Candidate individual ({candidate}) worse than all-time best ({self.best_individual}){tcolors.ENDC}")
                write_pickle(self.best_individual_path+str(self.generation_count)+".pkl", self.best_individual)
                self.generations_since_new_best += 1
                if self.generations_since_new_best > 2 and self.generation_count > self.min_generations:
                    print(f"{tcolors.OKGREEN}Converged. Best individual: {self.best_individual.ID}{tcolors.ENDC}")
                    return True
        write_pickle(self.best_individual_path+str(self.generation_count)+".pkl", self.best_individual)
        return False

    def find_fitness(self, offsprings=False, from_start=True, convergence_test=False):
        """ Find estimated fitness of every individual through simulation of process

        Args:
            offsprings (bool, optional): if population to be run is offsprings of a generation. Defaults to False.
            from_start (bool, optional): if fitness is to be estimated from scratch. Defaults to True.
        """
        pop = self.population.offsprings if offsprings else self.population.individuals
        if from_start:
            runs = self.simulations
            seeds = [seed for seed in range(runs)]
        else:
            runs = int(self.simulations/2)
            seeds = [np.random.randint(self.simulations, 1e+6) for _ in range(runs)]
            pop = pop[:5]
        if self.verbose: 
            if convergence_test:
                print(f"Finding fitness for convergence test individuals")
            else:
                print(f"Finding fitness for top 5 {'offsprings' if offsprings else 'individuals'}: {pop}")
        if not from_start or self.generation_count == 0 or offsprings:
            self.reset_final_scores()
            for run in range(runs):
                np.random.seed(seeds[run])
                self.process.init()
                self.process.reset()
                if self.verbose: print(f"\n{tcolors.BOLD}Finding score for run {run+1}{tcolors.ENDC}:")
                for individual in pop:
                    self.process.reset(reset_measures=False)
                    self.process.run(weighted_policy_weights=individual.genes)
                    if not convergence_test: individual.update_strategy_count(self.process.state)
                    for obj in ['deaths', 'infected', 'weighted', 'yll']:
                        score = self.get_objective(obj)(self.process)
                        self.final_scores[individual.ID][obj].append(score)
                        if self.verbose and obj == self.objective: print(f"{individual}: {score}")
        if self.verbose: print(f"\n{tcolors.UNDERLINE}Mean scores:{tcolors.ENDC}")
        for individual in sorted(pop, key=lambda i: np.mean(self.final_scores[i.ID][self.objective])):
            mean_score = np.mean(self.final_scores[individual.ID][self.objective])
            if self.verbose: print(f"{individual}: {mean_score:.2f}")
            individual.mean_score = mean_score

    def find_best_individual(self, offsprings=False, from_start=True, convergence_test=False):
        """ Loop through two most promising individuals to check for significance

        Args:
            offsprings (bool, optional): if population to be run is offsprings of a generation. Defaults to False.

        Returns:
            bool: True if both first and second best individuals passes t-test with significance
        """
        pop = self.population.offsprings if offsprings else self.population.individuals
        if not convergence_test: pop = self.population.sort_by_mean(pop, offsprings, from_start)
        if self.verbose: print(f"\nFinding best {'offspring' if offsprings and not convergence_test else 'individual'}...")
        range1, range2 = (1 if offsprings else 2, len(pop))
        for i in range(range1): # test two best
            first = pop[i]
            for j in range(i+1, range2):
                second = pop[j]
                if not self.t_test(first, second, significance=0.25 if convergence_test else 0.5):
                    if self.verbose: print(f"{tcolors.WARNING}Significance not fulfilled between {first} and {second}.{tcolors.ENDC}")
                    return False
        return True

    def t_test(self, first, second, significance):
        """ Performs one-sided t-test to check to variables for significant difference

        Args:
            first (Individual): presumed best individual
            second (Individual): presumed worse individual
            significance (float, optional): level of significance to test against. Defaults to 0.1.

        Returns:
            bool: True if significance is achieved
        """
        significant_best = False
        s1 = np.array(self.final_scores[first.ID][self.objective])
        s2 = np.array(self.final_scores[second.ID][self.objective])
        if not (s1==s2).all():
            z = s1 - s2
            p = scipy.stats.ttest_ind(z, np.zeros(len(s1)), alternative="less").pvalue
            significant_best = p < significance
        return significant_best

    def crossover(self, generation_count):
        """ Creates offspring from two fittest individuals

        Args:
            generation_count (int): what generation that is creating offsprings
        """
        # Select parents for crossover based on current heat
        n = len(self.population.individuals)
        ranks = np.arange(1,n+1)
        c = (n*2*(n-1))/(6*(n-1)+n)
        probs = np.exp(-ranks/c)
        selected = np.random.choice(self.population.individuals, 3, p=probs/sum(probs), replace=False)
        parents = list(combinations(selected, 2))
        new_offsprings = True
        for combination in parents:
            parent1 = combination[0]
            parent2 = combination[1]
            if self.verbose: print(f"{tcolors.OKCYAN}Crossing parents {parent1} and {parent2}{tcolors.ENDC}")
            p1 = parent1.genes
            p2 = parent2.genes
            if (p1 == p2).all(): continue 
            shape = p1.shape
            o1_genes = np.zeros(shape)
            o2_genes = np.zeros(shape)
            unique_child = False
            count = 0
            while not unique_child and count < 10: # don't make copy of parents
                c_row = np.random.randint(0, high=shape[1])
                c_col = np.random.randint(0, high=shape[2])
                vertical_cross = np.random.random() < 0.5
                if vertical_cross:
                    o1_genes[:, :, :c_col] = p1[:, :, :c_col]
                    o1_genes[:, :c_row, c_col] = p1[:, :c_row, c_col]
                    o1_genes[:, c_row:, c_col] = p2[:, c_row:, c_col]
                    o1_genes[:, :, c_col+1:] = p2[:, :, c_col+1:]
                    o2_genes[:, :, :c_col] = p2[:, :, :c_col]
                    o2_genes[:, :c_row, c_col] = p2[:, :c_row, c_col]
                    o2_genes[:, c_row:, c_col] = p1[:, c_row:, c_col]
                    o2_genes[:, :, c_col+1:] = p1[:, :, c_col+1:]
                else:
                    o1_genes[:, :c_row, :] = p1[:, :c_row, :]
                    o1_genes[:, c_row, :c_col] = p1[:, c_row, :c_col]
                    o1_genes[:, c_row, c_col:] = p2[:, c_row, c_col:]
                    o1_genes[:, c_row+1:, :] = p2[:, c_row+1:, :]
                    o2_genes[:, :c_row, :] = p2[:, :c_row, :]
                    o2_genes[:, c_row, :c_col] = p2[:, c_row, :c_col]
                    o2_genes[:, c_row, c_col:] = p1[:, c_row, c_col:]
                    o2_genes[:, c_row+1:, :] = p1[:, c_row+1:, :]
                unique_child = not ((p1 == o1_genes).all() or (p1 == o2_genes).all() or (p2 == o1_genes).all() or (p2 == o2_genes).all())
                count += 1
            if unique_child:
                o1 = Individual(generation=generation_count, offspring=True)
                o1.genes = o1_genes
                o2 = Individual(generation=generation_count, offspring=True)
                o2.genes = o2_genes
                o3 = Individual(generation=generation_count, offspring=True)
                o3.genes = np.divide(p1+p2, 2)
                o4 = Individual(generation=generation_count, offspring=True)
                o4.genes = np.divide(p1+3*p2, 4)
                o5 = Individual(generation=generation_count, offspring=True)
                o5.genes = np.divide(3*p1+p2, 4)
                if new_offsprings: 
                    self.population.offsprings = [o1,o2,o3,o4,o5]
                    new_offsprings = False
                else: 
                    self.population.offsprings += [o1,o2,o3,o4,o5]
            else:
                if self.verbose: print(f"{tcolors.WARNING}Unique child not found between {parent1} and {parent2}{tcolors.ENDC}")
                o1 = Individual(generation=generation_count, offspring=True)
                o1.genes = np.divide(p1+p2, 2)
                o2 = Individual(generation=generation_count, offspring=True)
                o2.genes = np.divide(p1+3*p2, 4)
                o3 = Individual(generation=generation_count, offspring=True)
                o3.genes = np.divide(3*p1+p2, 4)
                if new_offsprings:
                    self.population.offsprings = [o1,o2,o3]
                    new_offsprings = False
                else: 
                    self.population.offsprings += [o1,o2,o3]

    def mutation(self):
        """ Randomly altering the genes of offsprings """
        for offspring in self.population.offsprings:
            shape = offspring.genes.shape
            draw = np.random.random() 
            if draw < 0.1:
                i1 = np.random.randint(0, high=shape[0])
                j1 = np.random.randint(0, high=shape[1])
                k1 = np.random.randint(0, high=shape[2])
                i2 = np.random.randint(0, high=shape[0])
                while i1 == i2: i2 = np.random.randint(0, high=shape[0])
                j2 = np.random.randint(0, high=shape[1])
                while j1 == j2: j2 = np.random.randint(0, high=shape[1])
                k2 = np.random.randint(0, high=shape[2])
                while k1 == k2: k2 = np.random.randint(0, high=shape[2])
                value = copy(offspring.genes[i1, j1, k1])
                offspring.genes[i1, j1, k1] = offspring.genes[i2, j2, k2]
                offspring.genes[i2, j2, k2] = value
                vertical_mutation = np.random.random() > 0.5
                if vertical_mutation:
                    values = copy(offspring.genes[i1, :, k1])
                    offspring.genes[i1, :, k1] = offspring.genes[i2, :, k2]
                    offspring.genes[i2, :, k2] = values
                else:
                    values = copy(offspring.genes[i1, j1, :])
                    offspring.genes[i1, j1, :] = offspring.genes[i2, j2, :]
                    offspring.genes[i2, j2, :] = values
            if draw > 0.9:
                i = np.random.randint(0, high=shape[0])
                j = np.random.randint(0, high=shape[1])
                k = np.random.randint(0, high=shape[2])
                offspring.genes[i, j, k] = int(np.random.random() > 0.5)
            
    def repair_offsprings(self):
        """ Make sure the genes of offsprings are feasible, i.e. normalize. """
        for offspring in self.population.offsprings:
            norm = np.sum(offspring.genes, axis=2, keepdims=True)
            for loc in np.argwhere(norm==0):
                i=loc[0]
                j=loc[1]
                offspring.genes[i,j,:] = np.array([1,0,0,0,0])
                norm = np.sum(offspring.genes, axis=2, keepdims=True)
            offspring.genes = np.divide(offspring.genes, norm)

    def reset_final_scores(self, new_generation=False):
        if new_generation:
            pop_ids = [individual.ID for individual in self.population.individuals]
            individuals_to_remove = []
            for i in self.final_scores.keys():
                if i not in pop_ids:
                    individuals_to_remove.append(i)
            for i in individuals_to_remove:
                del self.final_scores[i]
        else:
            for i, scores in self.final_scores.items():
                for obj in ["deaths", "infected", "weighted", "yll"]:
                    self.final_scores[i][obj] = scores[obj][:self.simulations]

    def _generate_output_dirs(self):
        start_of_run = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_folder = f"/results/GA_{start_of_run}"
        folder_path = os.getcwd()+run_folder
        individuals_path = folder_path + "/individuals"
        final_scores_path = folder_path + "/final_scores"
        best_individuals_path = folder_path + "/best_individuals"
        os.mkdir(folder_path)
        os.mkdir(individuals_path)
        os.mkdir(final_scores_path)
        os.mkdir(best_individuals_path)
        self.overview_path = folder_path + f"/generation_overview.csv"
        self.individuals_path = individuals_path + f"/individuals_"
        self.final_score_path = final_scores_path + f"/final_score_"
        self.best_individual_path = best_individuals_path + f"/best_individual_"

        with open(folder_path + "/run_params.json", "w") as file:
            json.dump(self.__repr__(), file, indent=4)

    def write_to_file(self):
        """ Dump individuals and corresponding scores as pickle """
        write_pickle(self.individuals_path+str(self.generation_count)+".pkl", self.population.individuals)
        write_pickle(self.final_score_path+str(self.generation_count)+".pkl", self.final_scores)
        self.to_pandas()

    def to_pandas(self):
        """ Write summary of genetic algorithm run to csv """
        generation = [self.generation_count for _ in range(len(self.population.individuals))]
        ids = [individual.ID for individual in self.population.individuals]
        mean_score = [individual.mean_score for individual in self.population.individuals]
        gen_df = pd.DataFrame({"generation": generation, "individual": ids, "mean_score": mean_score})
        if self.generation_count == 0 :
            gen_df.to_csv(self.overview_path, header=True, index=False)
        else:
            gen_df.to_csv(self.overview_path, mode='a', header=False, index=False)

    def __str__(self):
        out = f"Time: {datetime.now().strftime('%d.%m.%Y (%H:%M:%S)')}\n"
        out += f"Objective: {self.objective}\n"
        out += f"Number of simulations: {self.simulations}\n"
        process_string = str(self.process).replace('\n', ', ')
        out += f"MDP params: ({process_string})\n"
        out += f"Population size: {len(self.population.individuals)}\n"
        out += f"Random individual genes: {self.random_individuals}\n"
        out += f"Minimum number of generations: {self.min_generations}"
        return out
    
    def __repr__(self):
        out = {}
        out["objective"] = self.objective
        out["simulations"] = self.simulations
        mdp = self.process
        out["process"] = {"horizon": mdp.horizon, "decision_period": mdp.decision_period, "policy": str(mdp.policy)}
        out["population_size"] = len(self.population.individuals)
        out["random_individuals"] = self.random_individuals
        out["min_generations"] = self.min_generations
        return out

class Population: 
    def __init__(self, population_size, verbose, random_individuals, individuals_from_file=None):
        """ Create population object

        Args:
            population_size (int): number of individuals to initialize population with
            verbose (bool): specify whether or not to print
        """
        if individuals_from_file is not None:
            self.individuals = individuals_from_file
        else:
            if random_individuals:
                self.individuals = [Individual() for _ in range(population_size)]
            else:
                self.individuals = [Individual(i) for i in range(1,population_size+1)]
        self.verbose = verbose
        self.least_fittest_index = 0
        self.offsprings = None
        if self.verbose: print(f"{tcolors.OKCYAN}Initial population: {self.individuals}{tcolors.ENDC}")
    
    def new_generation(self, generation_count, min_generations):
        """ Create new generation 

        Args:
            generation_count (int): what number of population this is
        """
        if min_generations//2 < generation_count <= min_generations:
            self.individuals = self.individuals[:-2]
        elif generation_count > min_generations:
            self.individuals = self.individuals[:-1]
        self.individuals.append(self.offsprings[0])
        if self.verbose: print(f"{tcolors.OKCYAN}New generation: {self.individuals}{tcolors.ENDC}") 

    def sort_by_mean(self, pop, offsprings, from_start):
        if from_start:
            if offsprings:
                self.offsprings = sorted(pop, key=lambda x: x.mean_score)
                return self.offsprings
            else:
                self.individuals = sorted(pop, key=lambda x: x.mean_score)
                return self.individuals
        else:
            if offsprings:
                self.offsprings[:5] = sorted(pop[:5], key=lambda x: x.mean_score)
                return self.offsprings[:5]
            else:
                self.individuals[:5] = sorted(pop[:5], key=lambda x: x.mean_score)
                return self.individuals[:5]

class Individual:
    ID_COUNTER=1
    GENERATION=0

    def __init__(self, i=-1, generation=0, offspring=False):
        """ Create individual instance

        Args:
            i (int, optional): what number of individual it is, to determine genes. Defaults to -1.
            generation (int, optional): what generation the individual belongs to. Defaults to 0.
            offspring (bool, optional): True if the individual is an offspring. Defaults to False.
        """
        self.ID = self.get_id(generation, offspring)
        self.mean_score = 0
        self.genetype = i
        self.genes = self.create_genes(i)
        self.strategy_count = defaultdict(partial(defaultdict, int))
    
    def get_id(self, generation, offspring):
        """ Generate id for an individual

        Args:
            generation (int): what generation the individual belongs to
            offspring (bool): True if the individual is an offspring. False otherwise

        Returns:
            str: id of individual
        """
        if offspring and generation == Individual.GENERATION: 
            Individual.GENERATION += 1
            Individual.ID_COUNTER = 1
        id = f"gen_{Individual.GENERATION}"
        id += f"_{Individual.ID_COUNTER:03d}"
        Individual.ID_COUNTER += 1
        return id

    def create_genes(self, i):
        """ Create genes for an individual

        Args:
            i (int): number in order to assign different genes to different individuals

        Returns:
            numpy.ndarray: shape #trend_states, #times_per_state, #number of weights
        """
        genes = np.zeros((3,3,5))
        if 0 <= i < 5:
            genes[:,:,i] = 1
        elif 5 <= i < 11:
            comb = list(combinations(np.arange(4)+1, 2))
            genes[:, :, comb[i-5][0]] = 1/2
            genes[:, :, comb[i-5][1]] = 1/2
        elif 11 <= i < 15:
            comb = list(combinations(np.arange(4)+1, 3))
            genes[:, :, comb[i-11][0]] = 1/3
            genes[:, :, comb[i-11][1]] = 1/3
            genes[:, :, comb[i-11][2]] = 1/3
        elif i==15:
            for j in range(1,5):
                genes[:, :, j] = 1/4
        elif i==16:
            for j in range(5):
                genes[:, :, j] = 1/5
        elif i==17: # Set one weight vector randomly, make for each timestep
            weights = np.zeros(5)
            for j in range(5):
                high = 100 if j > 0 else 50
                weights[j] = np.random.randint(low=0, high=high) # nr trend_states, max nr of occurrences (trendcounts), nr of weights (policies)
            norm = np.sum(weights)
            genes[:, :] = np.divide(weights, norm)
        else:
            genes = np.random.randint(low=0, high=100, size=(3,3,5)) # nr trend_states, max nr of occurrences (trendcounts), nr of weights (policies)
            norm = np.sum(genes, axis=2, keepdims=True)
            genes = np.divide(genes, norm)
        return genes

    def update_strategy_count(self, state):
        for trend, count in state.trend_count.items():
            for i in range(count):
                self.strategy_count[trend][i] += 1

    def __str__(self):
        return self.ID
    
    def __repr__(self):
        return self.ID
