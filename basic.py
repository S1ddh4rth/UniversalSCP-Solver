import random
import time
from SetCoveringProblemCreator import *

def generate_individual(num_subsets):
    return [random.choice([0, 1]) for _ in range(num_subsets)]

def calculate_fitness(individual, subsets, universal_set):
    selected_subsets = [subset for i, subset in enumerate(subsets) if individual[i] == 1]
    covered_elements = set().union(*selected_subsets)
    return len(covered_elements) - sum(individual)

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate):
    return [1 - gene if random.random() < mutation_rate else gene for gene in individual]

def genetic_algorithm(subsets, universal_set, population_size=100, generations=1000, mutation_rate=0.01):
    num_subsets = len(subsets)
    population = [generate_individual(num_subsets) for _ in range(population_size)]

    start_time = time.time()

    for generation in range(generations):
        gen_start_time = time.time()

        population = sorted(population, key=lambda x: calculate_fitness(x, subsets, universal_set), reverse=True)

        best_solution = population[0]
        best_fitness = calculate_fitness(best_solution, subsets, universal_set)
        selected_subsets = [subset for i, subset in enumerate(subsets) if best_solution[i] == 1]
        covered_elements = set().union(*selected_subsets)

        gen_end_time = time.time()
        print(f"Generation {generation + 1}:")
        print(f"  Best fitness: {best_fitness}")
        print(f"  Number of elements covered: {len(covered_elements)}")
        print(f"  Number of subsets used: {sum(best_solution)}")
        print(f"  Time taken: {gen_end_time - gen_start_time:.4f} seconds\n")

        new_population = population[:2]  # Keep the two best individuals

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:50], 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        population = new_population

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f} seconds")

    best_solution = max(population, key=lambda x: calculate_fitness(x, subsets, universal_set))
    return best_solution, calculate_fitness(best_solution, subsets, universal_set)

def main():
    scp = SetCoveringProblemCreator()

    choice = "1"

    if choice == "1":
        subsets = scp.Create(usize=100, totalSets=200)
        print(f"Generated a new Set Covering Problem with {len(subsets)} subsets")
    elif choice == "2":
        subsets = scp.ReadSetsFromJson("scp_test.json")
        print(f"Loaded Set Covering Problem from scp_test.json with {len(subsets)} subsets")

    universal_set = set().union(*subsets)
    print(f"Universal set size: {len(universal_set)}")

    best_solution, fitness = genetic_algorithm(subsets, universal_set)
    
    selected_subsets = [subset for i, subset in enumerate(subsets) if best_solution[i] == 1]
    covered_elements = set().union(*selected_subsets)

    print(f"\nBest solution found:")
    print(f"Number of subsets used: {sum(best_solution)}")
    print(f"Number of elements covered: {len(covered_elements)}")
    print(f"Fitness: {fitness}")

if __name__ == '__main__':
    main()
