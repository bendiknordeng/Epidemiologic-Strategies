import numpy as np
import copy

class SimpleGeneticAlgorithm:
    def __init__(self, population_size, process):
        self.population_size = population_size
        self.population = Population(population_size)
        self.best_fitness_score = np.infty
        self.second_best_fitness_score = np.infty
        self.generation_count = -1
        self.final_deaths = {}
        self.converged = False # If search for best individual has not converged
        self.process = process

    def find_fitness(self, runs, offsprings=None):
        population = self.population.individuals if not offsprings else offsprings
        self.final_deaths = {}
        seeds = [np.random.randint(0, 1e+6) for _ in range(runs)]
        print(f"\033[1mRunning generation {self.generation_count} \033[0m")
        for i in range(len(population)):
            self.final_deaths[i] = []
            for j in range(runs):
                np.random.seed(seeds[j])
                self.process.reset()
                self.process.run(weighted_policy_weights=population[i].genes)
                deaths = np.sum(self.process.path[-1].D)
                self.final_deaths[i].append(deaths)
            print(f"Agent {i} achieved {np.mean(self.final_deaths[i])} deaths...")

    def evaluate_fitness(self):
        new_fitness_king = False
        for i in range(self.population_size-1):
            for j in range((i+1), self.population_size):
                s1, s2 = self.final_deaths[i], self.final_deaths[j]
                s1_mean, s2_mean = np.mean(s1), np.mean(s2)
                if s1_mean <= s2_mean: # s1 better than s2
                    if s1_mean < self.best_fitness_score:
                        self.population.fittest = copy.deepcopy(self.population.get_individual(i))
                        self.best_fitness_score = s1_mean
                        new_fitness_king = True
                        if s2_mean < self.second_best_fitness_score:
                            self.population.second_fittest = copy.deepcopy(self.population.get_individual(j))
                            self.second_best_fitness_score = s2_mean
                    elif s1_mean < self.second_best_fitness_score:
                        if not s1_mean == self.best_fitness_score: # so that the best score is not added as second best
                            self.population.second_fittest = copy.deepcopy(self.population.get_individual(i))
                            self.second_best_fitness_score = s1_mean
                else: # s2 better than s1
                    if s2_mean < self.best_fitness_score:
                        self.population.fittest = copy.deepcopy(self.population.get_individual(j))
                        self.best_fitness_score = s2_mean
                        new_fitness_king = True
                        if s1_mean < self.second_best_fitness_score:
                            self.population.second_fittest = copy.deepcopy(self.population.get_individual(i))
                            self.second_best_fitness_score = s1_mean
                    elif s2_mean < self.second_best_fitness_score:
                        self.population.second_fittest = copy.deepcopy(self.population.get_individual(j))
                        self.second_best_fitness_score = s2_mean
        self.converged = not new_fitness_king

    def two_sided_t_test(s1,s2):
        """
        nr_obs = 5
        p = 1
        
        while p >= 0.01 and nr_obs < 100:
            nr_obs += 1
            deaths1 = np.zeros(nr_obs)
            deaths2 = np.zeros(nr_obs)
            for i in range(nr_obs):
                deaths1[i] = np.sum(s1[i].D)
                deaths2[i] = np.sum(s2[i].D)
            one = deaths1-deaths2
            p = scipy.stats.ttest_ind(one, np.zeros(nr_obs), axis=0, alternative="two-sided").pvalue
            if p < 0.01:
                found_best = True
        return found_best
        """
        pass

    def one_sided_t_test(s1, s2):
        """
        nr_obs = 5
        p = 1
        while p >= 0.01 and nr_obs < 100:
            nr_obs += 1
            deaths1 = np.zeros(nr_obs)
            deaths2 = np.zeros(nr_obs)
            for i in range(nr_obs):
                deaths1[i] = np.sum(s1[i].D)
                deaths2[i] = np.sum(s2[i].D)
            one = deaths1-deaths2
            p = scipy.stats.ttest_ind(one, np.zeros(nr_obs), axis=0, alternative="two-sided").pvalue
            
        print(f"Found different solutions after {nr_obs} observations")
        print(f"P value is {p}")
        print(f"Diff 1-2: {one}")
        less = scipy.stats.ttest_ind(one, np.zeros(nr_obs), axis=0, alternative="less").pvalue
        greater = scipy.stats.ttest_ind(one, np.zeros(nr_obs), axis=0, alternative="greater").pvalue
        if less < greater:
            print("Solution 1 is best")
        else:
            print("Solution 2 is best")
        """
        pass

    def new_generation(self):
        if self.generation_count == -1: self.generation_count +=1; return
        self.selection()
        self.crossover()
        self.mutation()
        self.repair_offsprings()
        offsprings = [self.population.o1, self.population.o2]
        self.find_fitness(10, offsprings=offsprings)
        self.get_fittest_offspring()
        self.population.new_generation()
        self.generation_count += 1

    def selection(self):
        self
        pass

    def crossover(self):
        p1 = self.population.fittest.genes
        p2 = self.population.second_fittest.genes
        shape = p1.shape
        o1 = np.zeros(shape)
        o2 = np.zeros(shape)
        c_row = np.random.randint(0, high=shape[1])
        c_col = np.random.randint(0, high=shape[2])
        vertical_cross = np.random.random() <= 0.5
        if vertical_cross:
            o1[:, :, :c_col] = p1[:, :, :c_col]
            o1[:, :c_row, c_col] = p1[:, :c_row, c_col]
            o1[:, c_row:, c_col] = p2[:, c_row:, c_col]
            o1[:, :, c_col+1:] = p2[:, :, c_col+1:]

            o2[:, :, :c_col] = p2[:, :, :c_col]
            o2[:, :c_row, c_col] = p2[:, :c_row, c_col]
            o2[:, c_row:, c_col] = p1[:, c_row:, c_col]
            o2[:, :, c_col+1:] = p1[:, :, c_col+1:]
        else:
            o1[:, :c_row, :] = p1[:, :c_row, :]
            o1[:, c_row, :c_col] = p1[:, c_row, :c_col]
            o1[:, c_row, c_col:] = p2[:, c_row, c_col:]
            o1[:, c_row+1:, :] = p2[:, c_row+1:, :]

            o2[:, :c_row, :] = p2[:, :c_row, :]
            o2[:, c_row, :c_col] = p2[:, c_row, :c_col]
            o2[:, c_row, c_col:] = p1[:, c_row, c_col:]
            o2[:, c_row+1:, :] = p1[:, c_row+1:, :]
        self.population.o1 = Individual()
        self.population.o1.genes = o1
        self.population.o2 = Individual()
        self.population.o2.genes = o2 

    def mutation(self):
        o1 = self.population.o1
        o2 = self.population.o2
        shape = o1.genes.shape
        for offspring in [o1, o2]:
            draw = np.random.random() 
            while draw > 0.1:
                i1 = np.random.randint(0, high=shape[0])
                j1 = np.random.randint(0, high=shape[1])
                k1 = np.random.randint(0, high=shape[2])
                i2 = np.random.randint(0, high=shape[0])
                while i1 == i2: i2 = np.random.randint(0, high=shape[0])
                j2 = np.random.randint(0, high=shape[1])
                while j1 == j2: j2 = np.random.randint(0, high=shape[1])
                k2 = np.random.randint(0, high=shape[2])
                while k1 == k2: k2 = np.random.randint(0, high=shape[2])
                value = offspring.genes[i1, j1, k1]
                offspring.genes[i1, j1, k1] = offspring.genes[i2, j2, k2]
                offspring.genes[i2, j2, k2] = value
                draw = np.random.random()
        self.population.o1 = o1 
        self.population.o2 = o2
    
    def repair_offsprings(self):
        o1 = self.population.o1
        o2 = self.population.o2
        for offspring in [o1, o2]:
            norm = np.sum(offspring.genes, axis=2, keepdims=True)
            offspring.genes = np.divide(offspring.genes, norm)
        
    def get_fittest_offspring(self):
        s1, s2 = self.final_deaths[0], self.final_deaths[1]
        s1_mean, s2_mean = np.mean(s1), np.mean(s2)
        if s1_mean <= s2_mean: # s1 better than s2
            return
        else: # s2 better than s1
            self.population.o1 = copy.deepcopy(self.population.o2)
            self.population.o2 = copy.deepcopy(self.population.o1)

    def add_fittest_offspring(self):
        # update fitness values of offspring
        self.fittest.calcFitness()
        self.second_fittest.calcFitness()

        # replace least fittest individual from most fittest offspring
        self.population.individuals[self.least_fittest_index] = self.get_fittest_offspring()

class Population: 
    def __init__(self, population_size):
        self.individuals = [Individual(i) for i in range(population_size)]
        self.fittest = None
        self.second_fittest = None
        self.least_fittest_index = 0
        self.o1 = None
        self.o2 = None

    def get_individual(self, i):
        return self.individuals[i]
    
    def get_fittest(self):
        return self.fittest
    
    def get_second_fittest(self):
        return self.second_fittest
    
    def new_generation(self):
        self.individuals = [self.fittest, self.second_fittest, self.o1]

"""
    def rank_individuals(self):
        max_fit_1 = -1
        max_fit_2 = -1
        max_fit_1_index = 0
        max_fit_2_index = 0
        min_fit = np.infty
        min_fit_index = 0
        for i in range(len(self.individuals)):
            if max_fit_1 < self.individuals[i].fitness:
                max_fit_1 = self.individuals[i].fitness
                max_fit_1_index = i
            elif max_fit_2 < self.individuals[i].fitness:
                max_fit_2 = self.individuals[i].fitness
                max_fit_2_index = i
            elif self.individuals[i].fitness < min_fit:
                min_fit = self.individuals[i].fitness
                min_fit_index = i
        
        self.fittest = max_fit_1
        return self.individuals[max_fit_1_index], self.individuals[max_fit_2_index], min_fit_index
"""
"""
    #def calculate_fitness(self):
        for i in self.individuals:
            i.calculate_fitness()
"""
class Individual:
    def __init__(self, i=0):
        self.fitness = 0
        self.genes = np.zeros(4)
        self.gene_length = 4
        self.create_individual(i)

    def create_individual(self, i):
        # Set genes randomly for each individual
        self.genes = np.random.randint(low=0, high=100, size=(3,4,4)) # nr wave_states, max nr of occurrences (wavecounts), nr of weights (policies)
        norm = np.sum(self.genes, axis=2, keepdims=True)
        self.genes = np.divide(self.genes, norm)
"""
    def calculate_fitness(self):
        self.fitness = 0
        for i in range(5):
            if (self.genes[i] == 1):
                self.fitness += 1
"""
"""    
    def mutate(self):
        np.random.shuffle(self.genes)
"""