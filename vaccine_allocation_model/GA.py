import numpy as np
import scipy
from covid import utils
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from functools import partial

class SimpleGeneticAlgorithm:
    def __init__(self, runs, population_size, process):
        self.population_size = population_size
        self.population = Population(population_size)
        self.best_fitness_score = np.infty
        self.second_best_fitness_score = np.infty
        self.generation_count = 0
        self.final_deaths = defaultdict(list)
        self.best_individual = Individual()
        self.best_deaths = np.inf
        self.converged = False # If search for best individual has not converged
        self.process = process
        self.runs = runs
        self.generations_since_new_best = 0
        start_of_run = datetime.now().strftime("%Y%m%d%H%M%S")
        run_folder = f"/results/GA_{start_of_run}"
        folder_path = os.getcwd()+run_folder
        individuals_path = folder_path + "/individuals"
        final_scores_path = folder_path + "/final_scores"
        os.mkdir(folder_path)
        os.mkdir(individuals_path)
        os.mkdir(final_scores_path)
        self.overview_path = folder_path + f"/generation_overview.csv"
        self.individuals_path = folder_path + f"/individuals/individuals_"
        self.final_score_path = folder_path + f"/final_scores/final_score_"
        self.number_of_runs = []
        self.GA_path = folder_path + f"/GA_object.pkl"

    def find_fitness(self, offsprings=False, from_start=True):
        population = self.population.individuals if not offsprings else self.population.offsprings
        self.final_deaths = defaultdict(list) if from_start else self.final_deaths
        runs = self.runs if from_start else int(self.runs/2)
        seeds = [np.random.randint(0, 1e+6) for _ in range(runs)]
        print(f"\033[1mRunning{' offsprings of ' if offsprings else ' '}generation {self.generation_count}\033[0m")
        for individual in population:
            print(f"Finding score for individual {individual.ID}...")
            for j in tqdm(range(runs), ascii=True):
                np.random.seed(seeds[j])
                self.process.init()
                self.process.run(weighted_policy_weights=individual.genes)
                strategy_count = self.process.path[-1].strategy_count
                for wave_state, count in strategy_count.items():
                    for wave in range(count):
                        individual.strategy_count[wave_state][str(wave+1)] += 1
                deaths = np.sum(self.process.path[-1].D)
                self.final_deaths[individual.ID].append(deaths)
            mean_score = np.mean(self.final_deaths[individual.ID])
            print(f"Mean score: {mean_score:.0f}\n")
            individual.mean_score = mean_score

    def one_sided_t_test(self, s1, s2, significance=0.1):
        significant_best = False
        s1 = np.array(s1)
        s2 = np.array(s2)
        if not (s1==s2).all():
            z = s1 - s2
            p = scipy.stats.ttest_ind(z, np.zeros(len(s1)), alternative="less").pvalue
            significant_best = p < significance
        return significant_best

    def find_best_individual(self, offsprings=False):
        range1, range2 = (2, len(self.population.individuals)) if not offsprings else (1, len(self.population.offsprings))
        for i in range(range1): # test two best
            first = self.population.individuals[i] if not offsprings else self.population.offsprings[i]
            for j in range(i+1, range2):
                second = self.population.individuals[j] if not offsprings else self.population.offsprings[j]
                s1, s2 = self.final_deaths[first.ID], self.final_deaths[second.ID]
                if not self.one_sided_t_test(s1, s2): return False
        return True

    def evaluate_generation(self):
        self.find_fitness()
        count = 0
        while not self.selection() and count <= 2: # returns False if no one is significant best.
            self.find_fitness(from_start=False)
            count += 1
        self.find_runs(count)
        utils.write_pickle(self.individuals_path+str(self.generation_count)+".pkl", self.population.individuals)
        utils.write_pickle(self.final_score_path+str(self.generation_count)+".pkl", self.final_deaths)
        self.to_pandas()
        if np.mean(self.final_deaths[self.population.individuals[0].ID]) < np.mean(self.best_deaths):
            self.population.offsprings = [self.population.individuals[0], self.best_individual]
            self.find_fitness(offsprings=True)
            new_best = self.selection(offsprings=True)
            count = 0
            while not new_best and count <= 2: # returns False if no one is significant best.
                self.find_fitness(offsprings=True, from_start=False)
                count += 1
            if new_best:
                self.best_individual = self.population.individuals[0]
                self.best_deaths = self.final_deaths[self.best_individual.ID]
                self.generations_since_new_best = 0
            else:
                self.generations_since_new_best += 1
                if self.generations_since_new_best > 2:
                    self.converged = True
                    utils.write_pickle(self.GA_path, self)
                    print(f"Converged. Best individual: {self.best_individual.ID}, score: {np.mean(self.best_deaths)}")
                    return
        self.crossover()
        self.mutation()
        self.repair_offsprings()
        offsprings = True
        self.find_fitness(offsprings)
        count = 0
        while not self.selection(offsprings) and count <= 2: # returns False if no one is significant best.
            self.find_fitness(offsprings, from_start=False)
            count += 1
        self.find_runs(count)
        self.population.new_generation(self.generation_count)
        self.generation_count += 1

    def selection(self, offsprings=False):
        self.population.sort_by_mean(offsprings)
        return self.find_best_individual(offsprings)

    def crossover(self):
        p1 = self.population.individuals[0].genes
        p2 = self.population.individuals[1].genes
        shape = p1.shape
        o1_genes = np.zeros(shape)
        o2_genes = np.zeros(shape)
        c_row = np.random.randint(0, high=shape[1])
        c_col = np.random.randint(0, high=shape[2])
        vertical_cross = np.random.random() <= 0.5
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
        o1 = Individual()
        o1.genes = o1_genes
        o2 = Individual()
        o2.genes = o2_genes
        o3 = Individual()
        o3.genes = np.divide(p1+p2, 2)
        self.population.offsprings = [o1,o2,o3]

    def mutation(self):
        for offspring in self.population.offsprings:
            shape = offspring.genes.shape
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
    
    def repair_offsprings(self):
        for offspring in self.population.offsprings:
            norm = np.sum(offspring.genes, axis=2, keepdims=True)
            for loc in np.argwhere(norm==0):
                i=loc[0]
                j=loc[1]
                offspring.genes[i,j,:] = np.array([1,0,0,0])
            norm = np.sum(offspring.genes, axis=2, keepdims=True)
            offspring.genes = np.divide(offspring.genes, norm)

    def to_pandas(self):
        generation = [self.generation_count for _ in range(len(self.population.individuals))]
        ids = [individual.ID for individual in self.population.individuals]
        mean_score = [individual.mean_score for individual in self.population.individuals]
        gen_df = pd.DataFrame({"generation": generation, "individual": ids, "mean_score": mean_score})
        if self.generation_count == 0 :
            gen_df.to_csv(self.overview_path, header=True, index=False)
        else:
            gen_df.to_csv(self.overview_path, mode='a', header=False, index=False)

    def find_runs(self, count):
        number_runs = self.runs
        if count > 0:
            for _ in range(count): number_runs += int(self.runs/2)
        self.number_of_runs.append(number_runs)

class Population: 
    def __init__(self, population_size):
        self.individuals = [Individual(i) for i in range(population_size)]
        self.least_fittest_index = 0
        self.offsprings = None

    def get_individual(self, i):
        return self.individuals[i]
    
    def new_generation(self, generation_count):
        if 10 < generation_count <= 20:
            self.individuals = self.individuals[:-2]
        elif generation_count > 20:
            self.individuals = self.individuals[:-1]
        self.individuals.append(self.offsprings[0])
    
    def sort_by_mean(self, offsprings=False):
        if not offsprings:
            self.individuals = sorted(self.individuals, key=lambda x: x.mean_score)
        else:
            self.offsprings = sorted(self.offsprings, key=lambda x: x.mean_score)    

class Individual:
    def __init__(self, i=-1):
        self.ID = id(self)
        self.mean_score = 0
        self.rank = -1
        self.genes = self.create_genes(i)
        self.strategy_count = defaultdict(partial(defaultdict, int))
    
    def create_genes(self, i):
        genes = np.zeros((3,4,4))
        if 0 <= i <= 3:
            genes[:,:,i] = 1
        elif 4 <= i <= 6:
            j = (i+1)%4
            k = (i+3)%4 if i==6 else (i+2)%4
            genes[:, :, j] = 0.5
            genes[:, :, k] = 0.5
        elif i==7:
            genes[:, :, 1] = 0.33
            genes[:, :, 2] = 0.33
            genes[:, :, 3] = 0.34
        elif i==8:
            for j in range(4):
                genes[:, :, j] = 0.25
        elif 9 <= i < 10: # Set one weight vector randomly, make for each timestep
            weights = np.zeros(4)
            for j in range(4):
                high = 100 if j > 0 else 50
                weights[j] = np.random.randint(low=0, high=high) # nr wave_states, max nr of occurrences (wavecounts), nr of weights (policies)
            norm = np.sum(weights)
            genes[:, :] = np.divide(weights, norm)
        else:
            genes = np.random.randint(low=0, high=100, size=(3,4,4)) # nr wave_states, max nr of occurrences (wavecounts), nr of weights (policies)
            norm = np.sum(genes, axis=2, keepdims=True)
            genes = np.divide(genes, norm)
        return genes