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
    return individual.fitness


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
            else:
                child1_state += parent1.state[cross_pts[i]:]
                child2_state += parent2.state[cross_pts[i]:]

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
    """Runs the genetic algorithm and returns the best solution and fitness history."""
    state_length = len(subsets)
    population = initialize_population(population_size, state_length)
    best_solution = None
    best_fitness_per_gen = []

    for generation in range(1, generations + 1):
        current_crossover_rate = decaying_rate(initial_crossover_rate, min_crossover_rate, generation, generations)
        current_mutation_rate = decaying_rate(initial_mutation_rate, min_mutation_rate, generation, generations)
        
        for individual in population:
            calculate_fitness(individual, subsets, universal_set)

        population.sort(key=lambda ind: ind.fitness, reverse=True)

        if best_solution is None or population[0].fitness > best_solution.fitness:
            best_solution = population[0]

        best_fitness_per_gen.append(population[0].fitness)

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

    return best_solution, best_fitness_per_gen, len(get_chosen_subsets(best_solution, subsets))


def main():
    """Main function to run experiments and plot results."""
    scp = SetCoveringProblemCreator()
    collection_sizes = [50, 150, 250, 350]
    num_runs = 30
    generations = 50

    final_fitness_mean = []
    final_fitness_std = []
    final_num_subsets_mean = []
    fitness_over_generations = {size: [] for size in collection_sizes}

    for size in collection_sizes:
        print(f"Running experiments for |S| = {size}")
        final_fitness = []
        final_num_subsets = []
        fitness_per_gen_runs = []

        for run in range(num_runs):
            listOfSubsets = scp.Create(usize=100, totalSets=size)
            universal_set = set().union(*listOfSubsets)
            best_solution, best_fitness_per_gen, num_subsets = genetic_algorithm(
                listOfSubsets,
                universal_set,
                population_size=50,
                generations=generations,
                tournament_size=5,
                initial_crossover_rate=0.9,
                min_crossover_rate=0.5,
                initial_mutation_rate=0.05,
                min_mutation_rate=0.001,
            )
            final_fitness.append(best_solution.fitness)
            final_num_subsets.append(num_subsets)
            fitness_per_gen_runs.append(best_fitness_per_gen)

        mean_fitness = np.mean(final_fitness)
        std_fitness = np.std(final_fitness)
        final_fitness_mean.append(mean_fitness)
        final_fitness_std.append(std_fitness)

        mean_num_subsets = np.mean(final_num_subsets)
        final_num_subsets_mean.append(mean_num_subsets)

        fitness_per_gen_runs = np.array(fitness_per_gen_runs)
        mean_fitness_per_gen = np.mean(fitness_per_gen_runs, axis=0)
        fitness_over_generations[size] = mean_fitness_per_gen

    # Plot Mean ± Std of Best Fitness and Mean Number of Subsets
    fig, ax1 = plt.subplots()
    ax1.errorbar(
        collection_sizes,
        final_fitness_mean,
        yerr=final_fitness_std,
        fmt='-o',
        color='blue',
        label='Mean Best Fitness',
    )
    ax1.set_xlabel('Number of Subsets |S|')
    ax1.set_ylabel('Mean Best Fitness', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Mean and Std of Best Fitness & Mean Number of Subsets Chosen over 10 runs')

    ax2 = ax1.twinx()
    ax2.plot(
        collection_sizes,
        final_num_subsets_mean,
        '-s',
        color='orange',
        label='Mean Number of Subsets',
    )
    ax2.set_ylabel('Mean Number of Subsets Chosen', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.show()

    # Plot Mean Best Fitness over Generations
    plt.figure()
    for size in collection_sizes:
        plt.plot(
            range(1, generations + 1),
            fitness_over_generations[size],
            label=f'|S|={size}',
        )
    plt.xlabel('Generation')
    plt.ylabel('Mean Best Fitness')
    plt.title('Mean Best Fitness over Generations over 10 runs')
    plt.legend()
    plt.show()

    # Optionally, display the final results
    for size in collection_sizes:
        print(f"\nResults for |S| = {size}:")
        print(f"Mean Best Fitness: {final_fitness_mean[collection_sizes.index(size)]:.2f}")
        print(
            f"Std of Best Fitness: {final_fitness_std[collection_sizes.index(size)]:.2f}"
        )
        print(
            f"Mean Number of Subsets Chosen: {final_num_subsets_mean[collection_sizes.index(size)]:.2f}"
        )


if __name__ == '__main__':
    main()
