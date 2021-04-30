import numpy as np

#Main class
class SimpleDemoGA:
    def __init__(self):
        self.population = Population()
        self.fittest = None
        self.second_fittest = None
        self.generation_count = 0

    def selection(self):
        self.fittest = self.population.get_fittest()
        self.secondFittest = self.population.get_second_fittest()

    def crossover(self):
        # Implement Two-Dimensional substring crossover
        pass

    def mutation(self):
        # Implement Two-Dimensional Mutation
        pass

    def get_fittest_offspring(self):
        if self.fittest.fitness > self.second_fittest.fitness:
            return self.fittest
        return self.second_fittest

    def add_fittest_offspring(self):

        #Update fitness values of offspring
        self.fittest.calcFitness()
        self.secondFittest.calcFitness()

        #Get index of least fit individual
        least_fittest_index = self.population.get_least_fittest_index()

        #Replace least fittest individual from most fittest offspring
        self.population.individuals[least_fittest_index] = self.get_fittest_offspring()

class Individual:
    def __init__(self):
        self.fitness = 0
        self.genes = np.zeros(5)
        self.gene_length = 5
        self.create_individual()

    def create_individual(self):
        #Set genes randomly for each individual
        for i in range(self.gene_length):
            self.genes[i] = np.abs(np.random.randint(1) % 2)

    def calculate_fitness(self):
        self.fitness = 0
        for i in range(5):
            if (self.genes[i] == 1):
                self.fitness += 1

class Population: 
    def __init__(self, population_size):
        self.pop_size = 10
        self.individuals = [Individual() for _ in range(population_size)]
        self.fittest = 0

    #Get the fittest individual
    def rank_fittest_individual(self):
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
        return self.individuals[max_fit_1_index], self.individuals[max_fit_2_index], self.individuals[min_fit_index]

    def calculate_fitness(self):
        for i in self.individuals:
            i.calculate_fitness()

if __name__=='__main__':
    demo = SimpleDemoGA()
    demo.population.initialize_population(5)
    demo.population.calculate_fitness()

    print("Generation: " + demo.generation_count + " Fittest: " + demo.population.fittest)

    #While population gets an individual with maximum fitness
    while demo.population.fittest < 5:
        demo.generationCount += 1

        #Do selection
        demo.selection()

        #Do crossover
        demo.crossover()

        #Do mutation under a random probability
        if (np.random.randint %7 < 5):
            demo.mutation()

        #Add fittest offspring to population
        demo.addFittestOffspring()

        #Calculate new fitness value
        demo.population.calculateFitness()

        print("Generation: " + demo.generationCount + " Fittest: " + demo.population.fittest)

    print("\nSolution found in generation " + demo.generationCount)
    print("Fitness: "+demo.population.getFittest().fitness)
    gene_string = "Genes: "
    for i in range(5):
        gene_string += (demo.population.getFittest().genes)[i]

    print(gene_string)
    print()
