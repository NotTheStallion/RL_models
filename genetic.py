import numpy as np
import random
from simple.grid_world_env import GridWorldEnvSlow, GridWorldEnv
import matplotlib.pyplot as plt


class GAAgent:
    def __init__(self, env :GridWorldEnv, population_size=50, chromosome_length=20, mutation_rate=0.1, elitism_count=2):
        """
        Initializes the Genetic Algorithm agent with necessary parameters.
        """
        self.env = env
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Initializes the population with random chromosomes.
        Returns:
            A list of randomly generated chromosomes.
        """
        # Generate random chromosomes by moving on the board
        population = []
        for _ in range(self.population_size):
            # use self.possible_actions() to get the possible actions
            self.env.reset()
            chromosome = []
            for _ in range(self.chromosome_length):
                action = random.choice(self.env.possible_actions())
                self.env.step(action)
                chromosome.append(action)
            population.append(chromosome)
        return population

    def evaluate_fitness(self, chromosome):
        """
        Evaluates the fitness of a given chromosome.
        Args:
            chromosome: A list of actions representing a candidate solution.
        Returns:
            Fitness score as a numerical value.
        """
        # the sum of distance of each cell to the goal
        self.env.reset()
        total_distance = 0
        for action in chromosome:
            try:
                next_state, reward, done, _ = self.env.step(action)
                total_distance += abs(next_state[0] - self.env.terminal_state[0]) + abs(next_state[1] - self.env.terminal_state[1])
            except:
                total_distance = np.inf
        return -total_distance

    def select_parents(self, fitness_scores):
        """
        Selects two parents based on their fitness scores using roulette wheel selection.
        Returns:
            Two selected parent chromosomes.
        """
        # Apply softmax to fitness scores
        exp_scores = np.exp(fitness_scores - np.max(fitness_scores))
        selection_probs = exp_scores / np.sum(exp_scores)

        parent1 = self.population[np.random.choice(len(self.population), p=selection_probs)]
        parent2 = self.population[np.random.choice(len(self.population), p=selection_probs)]

        return parent1, parent2

    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent chromosomes.
        Args:
            parent1: First parent chromosome.
            parent2: Second parent chromosome.
        Returns:
            Two offspring chromosomes.
        """
        crossover_point = random.randint(1, self.chromosome_length - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

        return offspring1, offspring2

    def mutate(self, chromosome):
        """
        Mutates a chromosome based on the mutation rate.
        Args:
            chromosome: A chromosome to mutate.
        Returns:
            A mutated chromosome.
        """
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.choice(range(len(self.env.actions)))

        return chromosome

    def run_generation(self):
        """
        Executes one generation of the genetic algorithm.
        Returns:
            The new population for the next generation.
        """
        fitness_scores = [self.evaluate_fitness(c) for c in self.population]
        new_population = []

        # Apply elitism (preserve best chromosomes)
        elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
        elites = [self.population[i] for i in elite_indices]
        new_population.extend(elites)

        # Generate offspring
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(fitness_scores)
            offspring1, offspring2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(offspring1))
            if len(new_population) < self.population_size:
                new_population.append(self.mutate(offspring2))

        self.population = new_population

    def run(self, generations):
        """
        Runs the genetic algorithm for a specified number of generations.
        Args:
            generations: Number of generations to run.
        """
        best_fitness_per_generation = []

        for generation in range(generations):
            fitness_scores = [self.evaluate_fitness(c) for c in self.population]
            best_fitness = max(fitness_scores)
            best_fitness_per_generation.append(best_fitness)

            print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
            self.run_generation()

        # Plot the best fitness per generation
        plt.plot(best_fitness_per_generation)
        plt.title('Best Fitness per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()



if __name__ == "__main__":
    random_seed = 2020
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    height = 4
    width = 4
    number_of_holes = 4

    env = GridWorldEnvSlow(height, width, number_of_holes)

    agent = GAAgent(env, population_size=50, chromosome_length=20, mutation_rate=0.1, elitism_count=2)
    agent.run(generations=100)