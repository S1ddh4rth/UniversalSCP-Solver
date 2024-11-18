import random
import time
from SetCoveringProblemCreator import *

class Individual: #represents a possible solution to the SCP (Set Covering Problem)
    def __init__(self, state):
        self.state = state #list of binary values, ith value being 1 indicates that the (i+1)th set is taken
        #size of the list is equal to total number of subsets in the SCP
        self.fitness = 0 #value of custom fitness function

def initialize_population(population_size, state_length): #generates the first population
    return [Individual([random.choice([0, 1]) for _ in range(state_length)]) for _ in range(population_size)]

def calculate_fitness(individual, subsets, universal_set): #
    selected_subsets = [subset for i, subset in enumerate(subsets) if individual.state[i] == 1]
    covered_elements = set().union(*selected_subsets)
    # num_elements_covered = len(covered_elements)
    # num_subsets_used = sum(individual.state)

    #most basic fitness fn
    # individual.fitness = num_elements_covered - num_subsets_used

    #option 2
    #individual.fitness = num_elements_covered* (1-num_subsets_used/len(universal_set))

    #option3
    individual.fitness = 100 - ( 50*((len(universal_set)-len(covered_elements))/(len(universal_set))) + 50*(sum(individual.state)/len(listOfSubsets)) )


    return individual.fitness

def select_parents(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)

def crossover(parent1, parent2): #random crossover point selected, only 1 crossover point
    crossover_point = random.randint(1, len(parent1.state) - 1)
    child1 = Individual(parent1.state[:crossover_point] + parent2.state[crossover_point:])
    calculate_fitness(child1,listOfSubsets,universal_set) #update fitness values
    child2 = Individual(parent2.state[:crossover_point] + parent1.state[crossover_point:])
    calculate_fitness(child2,listOfSubsets,universal_set) #update fitness values
    return child1, child2 #returns the new states


def mutate(individual, mutation_rate): #implementing mutaion
    for i in range(len(individual.state)):
        if random.random() < mutation_rate:
            individual.state[i] = 1-individual.state[i] #inverts one of the elements in the list
    calculate_fitness(individual,listOfSubsets,universal_set) #update fitness after mutation

def get_chosen_subsets(solution,subsets): #return which subsets are currently chosen
    return [subset for i, subset in enumerate(subsets) if solution.state[i] == 1]

def genetic_algorithm(subsets, universal_set, population_size=50, generations=100, tournament_size=5, crossover_rate=0.9, mutation_rate=0.01):
    state_length = len(subsets)
    population = initialize_population(population_size, state_length)
    #Start timer
    start_time = time.time()
    best_solution = None

    for generation in range(1,generations+1):
        #Calculate fitness for each individual in the population
        for individual in population:
            calculate_fitness(individual, subsets, universal_set)

        #Sort population by fitness (higher fitness is better)
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        #Update the best solution found so far
        if best_solution is None or population[0].fitness > best_solution.fitness:
            best_solution = population[0]

        #Implementing Elitism: carry forward the top n individuals
        num_elites=10
        new_population = population[:num_elites]

        while len(new_population) < population_size:
            if random.random() < crossover_rate:
                parent1 = select_parents(population, tournament_size)
                parent2 = select_parents(population, tournament_size)
                child1, child2 = crossover(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                new_population.append(select_parents(population, tournament_size))

        #Mutate individuals (skip the elite ones)
        for individual in new_population[num_elites:]:
            mutate(individual, mutation_rate)
        #updating the population
        population = new_population

        #printing progress every 10 generations
        if generation  == 5000:
            best_in_generation = max(population, key=lambda ind: ind.fitness)
            chosen_subsets = get_chosen_subsets(best_in_generation, subsets)
            covered_elements = set().union(*chosen_subsets)

            print(f"Generation {generation}:")
            print(f" - Best fitness: {best_in_generation.fitness}")
            print(f" - Number of sets used: {len(chosen_subsets)}")
            print(f" - Number of elements covered: {len(covered_elements)}")

    #stopping timer
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

    #Return the best solution found during all generations
    return best_solution

def output_chosen_subsets(chosen_subsets):
    print("\nChosen subsets:")
    for i, subset in enumerate(chosen_subsets, 1):
        print(f"Subset {i}: {subset}")

def main():
    global listOfSubsets, universal_set
    scp = SetCoveringProblemCreator()
    choice = 2

    if choice == 1:
        listOfSubsets = scp.ReadSetsFromJson("scp_test.json")
        print("Using problem from JSON file.")
    else:
        listOfSubsets = scp.Create(usize=100, totalSets=200)
        print("Generated new problem.")

    print(f"Number of subsets: {len(listOfSubsets)}")

    universal_set = set().union(*listOfSubsets)
    print(f"Size of universal set: {len(universal_set)}")

    best_solution = genetic_algorithm(listOfSubsets, universal_set)

    chosen_subsets = get_chosen_subsets(best_solution, listOfSubsets)
    covered_elements = set().union(*chosen_subsets)

    print(f"\nBest solution found:")
    print(f"Number of subsets used: {len(chosen_subsets)}")
    print(f"Number of elements covered: {len(covered_elements)}")
    print(f"Fitness: {best_solution.fitness}")

    output_chosen_subsets(chosen_subsets)

if __name__ == '__main__':
    main()
