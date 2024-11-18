import random
import time
import matplotlib.pyplot as plt
import numpy as np
from SetCoveringProblemCreator import SetCoveringProblemCreator


class Individual:
    """Represents a possible solution to the Set Covering Problem."""

    def __init__(self, state):
        self.state = state
        self.fitness = 0


def initialize_population(population_size, state_length):
    """Generates the initial population."""
    return [Individual([random.choice([0, 1]) for _ in range(state_length)]) for _ in range(population_size)]


def calculate_fitness(individual, subsets, universal_set):
    """Calculates the fitness of an individual."""
    selected_subsets = [subset for i, subset in enumerate(subsets) if individual.state[i] == 1]
    covered_elements = set().union(*selected_subsets)
    individual.fitness = 100 - (
        50 * ((len(universal_set) - len(covered_elements)) / len(universal_set))
        + 50 * (sum(individual.state) / len(subsets))
    )
    return individual.fitness, len(selected_subsets), len(covered_elements)


def select_parents(population, tournament_size):
    """Selects a parent using tournament selection."""
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)


def crossover(parent1, parent2, subsets, universal_set, num_cross_pts=2):
    """Performs multi-point crossover between two parents."""
    cross_pts = sorted(random.sample(range(1, len(parent1.state)), num_cross_pts))
    
    child1_state = parent1.state[:cross_pts[0]]
    child2_state = parent2.state[:cross_pts[0]]
    
    for i in range(len(cross_pts)):
        if i % 2 == 0:
            if i + 1 < len(cross_pts):
                child1_state += parent2.state[cross_pts[i]:cross_pts[i + 1]]
                child2_state += parent1.state[cross_pts[i]:cross_pts[i + 1]]
            else:
                child1_state += parent2.state[cross_pts[i]:]
                child2_state += parent1.state[cross_pts[i]:]
        else:
            if i + 1 < len(cross_pts):
                child1_state += parent1.state[cross_pts[i]:cross_pts[i + 1]]
                child2_state += parent2.state[cross_pts[i]:cross_pts[i + 1]]

    child1 = Individual(child1_state)
    child2 = Individual(child2_state)
    calculate_fitness(child1, subsets, universal_set)
    calculate_fitness(child2, subsets, universal_set)
    return child1, child2


def mutate(individual, mutation_rate, subsets, universal_set):
    """Mutates an individual's state."""
    for i in range(len(individual.state)):
        if random.random() < mutation_rate:
            individual.state[i] = 1 - individual.state[i]
    calculate_fitness(individual, subsets, universal_set)


def get_chosen_subsets(solution, subsets):
    """Returns the subsets chosen in the solution."""
    return [subset for i, subset in enumerate(subsets) if solution.state[i] == 1]


def decaying_rate(initial_rate, min_rate, current_gen, max_gens):
    """Calculates a decaying rate."""
    return max(min_rate, initial_rate - (initial_rate - min_rate) * (current_gen / max_gens))


def genetic_algorithm(subsets, universal_set, population_size=50, generations=50, tournament_size=5, initial_crossover_rate=0.9, min_crossover_rate=0.5, initial_mutation_rate=0.05, min_mutation_rate=0.001):
    """Runs the genetic algorithm and tracks fitness, subsets, and elements covered."""
    state_length = len(subsets)
    population = initialize_population(population_size, state_length)
    best_solution = None

    # Tracking values over generations
    best_fitness_per_gen = []
    subsets_chosen_per_gen = []
    elements_covered_per_gen = []

    for generation in range(1, generations + 1):
        current_crossover_rate = decaying_rate(initial_crossover_rate, min_crossover_rate, generation, generations)
        current_mutation_rate = decaying_rate(initial_mutation_rate, min_mutation_rate, generation, generations)
        
        for individual in population:
            fitness, num_subsets, num_elements = calculate_fitness(individual, subsets, universal_set)
            individual.fitness = fitness

        population.sort(key=lambda ind: ind.fitness, reverse=True)

        if best_solution is None or population[0].fitness > best_solution.fitness:
            best_solution = population[0]

        # Track best values per generation
        best_fitness_per_gen.append(population[0].fitness)
        num_subsets, num_elements = len(get_chosen_subsets(population[0], subsets)), len(set().union(*get_chosen_subsets(population[0], subsets)))
        subsets_chosen_per_gen.append(num_subsets)
        elements_covered_per_gen.append(num_elements)

        num_elites = 10
        new_population = population[:num_elites]

        while len(new_population) < population_size:
            if random.random() < current_crossover_rate:
                parent1 = select_parents(population, tournament_size)
                parent2 = select_parents(population, tournament_size)
                child1, child2 = crossover(parent1, parent2, subsets, universal_set)
                new_population.extend([child1, child2])
            else:
                new_population.append(select_parents(population, tournament_size))

        for individual in new_population[num_elites:]:
            mutate(individual, current_mutation_rate, subsets, universal_set)

        population = new_population

    return best_fitness_per_gen, subsets_chosen_per_gen, elements_covered_per_gen


def main():
    """Main function to run a single experiment and plot results."""
    scp = SetCoveringProblemCreator()
    listOfSubsets = scp.Create(usize=100, totalSets=200)
    universal_set = set().union(*listOfSubsets)

    # Run genetic algorithm for a single run and track values over generations
    best_fitness_per_gen, subsets_chosen_per_gen, elements_covered_per_gen = genetic_algorithm(
        listOfSubsets, universal_set, population_size=50, generations=50
    )

    # Plot Fitness over Generations
    plt.figure()
    plt.plot(range(1, 51), best_fitness_per_gen, label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Best Fitness over Generations")
    plt.legend()
    plt.show()

    # Plot Number of Subsets Chosen over Generations
    plt.figure()
    plt.plot(range(1, 51), subsets_chosen_per_gen, label="Number of Subsets Chosen", color='orange')
    plt.xlabel("Generation")
    plt.ylabel("Number of Subsets")
    plt.title("Number of Subsets Chosen over Generations")
    plt.legend()
    plt.show()

    # Plot Number of Elements Covered over Generations
    plt.figure()
    plt.plot(range(1, 51), elements_covered_per_gen, label="Elements Covered", color='green')
    plt.xlabel("Generation")
    plt.ylabel("Number of Elements Covered")
    plt.title("Elements Covered over Generations")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
