import random
import time
from SetCoveringProblemCreator import *

class Individual:
    def __init__(self, state):
        self.state = state
        self.fitness = 0

def initialize_population(population_size, state_length):
    return [Individual([random.choice([0, 1]) for _ in range(state_length)]) for _ in range(population_size)]

def calculate_fitness(individual, subsets, universal_set):
    selected_subsets = [subset for i, subset in enumerate(subsets) if individual.state[i] == 1]
    covered_elements = set().union(*selected_subsets)
    individual.fitness = 100 - (90 * ((len(universal_set) - len(covered_elements)) / len(universal_set)) + 20 * (sum(individual.state) / len(listOfSubsets)))
    return individual.fitness

def select_parents(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)

def crossover(parent1, parent2, num_cross_pts=2):
    cross_pts = sorted(random.sample(range(1, len(parent1.state)), num_cross_pts))
    child1_state = parent1.state[:cross_pts[0]]
    child2_state = parent2.state[:cross_pts[0]]

    for i in range(len(cross_pts)):
        if i % 2 == 0:
            if i + 1 < len(cross_pts):
                child1_state += parent2.state[cross_pts[i]:cross_pts[i+1]]
                child2_state += parent1.state[cross_pts[i]:cross_pts[i+1]]
            else:
                child1_state += parent2.state[cross_pts[i]:]
                child2_state += parent1.state[cross_pts[i]:]
        else:
            if i + 1 < len(cross_pts):
                child1_state += parent1.state[cross_pts[i]:cross_pts[i+1]]
                child2_state += parent2.state[cross_pts[i]:cross_pts[i+1]]
            else:
                child1_state += parent1.state[cross_pts[i]:]
                child2_state += parent2.state[cross_pts[i]:]

    child1 = Individual(child1_state)
    child2 = Individual(child2_state)
    calculate_fitness(child1, listOfSubsets, universal_set)
    calculate_fitness(child2, listOfSubsets, universal_set)
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual.state)):
        if random.random() < mutation_rate:
            individual.state[i] = 1 - individual.state[i]
    calculate_fitness(individual, listOfSubsets, universal_set)

def get_chosen_subsets(solution, subsets):
    return [subset for i, subset in enumerate(subsets) if solution.state[i] == 1]

def decaying_rate(initial_rate, min_rate, current_gen, max_gens):
    return max(min_rate, initial_rate - (initial_rate - min_rate) * (current_gen / max_gens))

def local_beam_search(population, beam_size, mutation_rate):
    beam_individuals = population[:beam_size]
    new_individuals = []
    for individual in beam_individuals:
        new_individual = Individual(individual.state[:])
        mutate(new_individual, mutation_rate / 2)
        new_individuals.append(new_individual)
    return new_individuals

def genetic_algorithm(subsets, universal_set, population_size=100, generations=3000, tournament_size=5, initial_crossover_rate=0.9, min_crossover_rate=0.5, initial_mutation_rate=0.05, min_mutation_rate=0.001, initial_beam_search_rate=0.01, max_beam_search_rate=0.1, beam_size=5):
    state_length = len(subsets)
    population = initialize_population(population_size, state_length)
    start_time = time.time()
    best_solution = None

    for generation in range(1, generations + 1):
        current_crossover_rate = decaying_rate(initial_crossover_rate, min_crossover_rate, generation, generations)
        current_mutation_rate = decaying_rate(initial_mutation_rate, min_mutation_rate, generation, generations)
        current_beam_search_rate = decaying_rate(initial_beam_search_rate, max_beam_search_rate, generation, generations)
        for individual in population:
            calculate_fitness(individual, subsets, universal_set)

        population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        if best_solution is None or population[0].fitness > best_solution.fitness:
            best_solution = population[0]

        num_elites = 10
        new_population = population[:num_elites]

        while len(new_population) < population_size:
            if random.random() < current_crossover_rate:
                parent1 = select_parents(population, tournament_size)
                parent2 = select_parents(population, tournament_size)
                child1, child2 = crossover(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                new_population.append(select_parents(population, tournament_size))

        for individual in new_population[num_elites:]:
            mutate(individual, current_mutation_rate)
        
        if random.random() < current_beam_search_rate:
            new_population.extend(local_beam_search(population, beam_size, current_mutation_rate))

        population = new_population

        if time.time() - start_time > 45:
            print(f"Time limit reached at generation {generation}.")
            break

    return best_solution, time.time() - start_time

def output_solution(solution, subsets, universal_set, execution_time):
    chosen_subsets = get_chosen_subsets(solution, subsets)
    covered_elements = set().union(*chosen_subsets)
    min_subsets = len(chosen_subsets)

    print("Roll no : 2022A7PS0569G")
    print(f"Number of subsets in scp_test.json file : {len(subsets)}")
    print("Solution :")

    for i, val in enumerate(solution.state):
        print(f"{i}:{val}", end=", ")
        if (i + 1) % 10 == 0:
            print()  # Print new line after every 10 elements

    print(f"\nFitness value of best state : {solution.fitness}")
    print(f"Minimum number of subsets that can cover the Universe-set : {min_subsets}")
    print(f"Time taken : {execution_time:.2f} seconds")

def main():
    global listOfSubsets, universal_set
    scp = SetCoveringProblemCreator()
    choice = 1

    if choice == 1:
        listOfSubsets = scp.ReadSetsFromJson("scp_test_1.json")
        print("Using problem from JSON file.")
    else:
        listOfSubsets = scp.Create(usize=100, totalSets=200)
        print("Generated new problem.")

    universal_set = set().union(*listOfSubsets)

    best_solution, execution_time = genetic_algorithm(listOfSubsets, universal_set)
    output_solution(best_solution, listOfSubsets, universal_set, execution_time)

if __name__ == '__main__':
    main()
